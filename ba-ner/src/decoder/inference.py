"""
inference.py — LLM-Inferenz (Qwen3.5) auf dem Test-Set

Unterstuetzt zwei Modi auf demselben Codepfad:

  1. LoRA/QLoRA Mode (Default): Lade Basismodell + trainierten LoRA-Adapter.
     Aktiviert via --adapter <path>.

  2. Zero-Shot Mode: Lade nur das Basismodell ohne Adapter.
     Aktiviert via --zeroshot ODER mode: zeroshot in der Config.

Beide Modi nutzen denselben Prompt, denselben Parser, dieselben Metriken
und denselben Output-Pfad-Konventionen, damit die Ergebnisse direkt
vergleichbar sind.

Greedy Decoding (do_sample=False) wird fuer Reproduzierbarkeit verwendet.

Verwendung:
    # LoRA-Mode
    python -m src.decoder.inference \
        --adapter results/multinerd/qwen35-4b-qlora/best_lora_adapter \
        --config configs/qwen35_4b.yaml

    # Zero-Shot Mode
    python -m src.decoder.inference \
        --zeroshot \
        --config configs/qwen35_4b_zeroshot.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from src.config import DATASET_LANGUAGE, DATASET_NAME, load_experiment_config, output_dir_from_config
from src.data.dataset_loader import load_ner_dataset
from src.data.preprocess_decoder import prepare_test_inputs
from src.decoder.parse_output import (
    entities_to_bio,
    evaluate_llm_predictions,
    parse_llm_output_with_diagnostics,
)
from src.decoder.generation import (
    THINKING_ENABLED,
    prepare_generation_batch_inputs,
    prepare_generation_inputs,
)
from src.evaluate.efficiency import count_parameters, get_vram_peak_mb, reset_vram_tracking
from src.run_metadata import collect_run_metadata

console = Console()
INFERENCE_CHECKPOINT_VERSION = 2
INFERENCE_CHECKPOINT_FILENAME = "inference_checkpoint.jsonl"


def _legacy_checkpoint_header(expected_header: Dict[str, Any]) -> Dict[str, Any] | None:
    """Return the v1 equivalent for an unchanged batch-size-one QLoRA run."""
    if (
        expected_header.get("version") != 2
        or expected_header.get("generation_batch_size") != 1
        or expected_header.get("precision_mode") != "qlora_4bit"
        or expected_header.get("attn_impl") != "sdpa"
    ):
        return None
    legacy_keys = (
        "experiment_name",
        "model_name",
        "regime",
        "thinking_enabled",
        "max_new_tokens",
        "seed",
        "total_samples",
    )
    return {
        "version": 1,
        **{key: expected_header[key] for key in legacy_keys},
    }


def _load_inference_checkpoint(
    path: Path,
    expected_header: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return []
    try:
        header = json.loads(lines[0])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid inference checkpoint header: {path}") from exc
    if header != expected_header and header != _legacy_checkpoint_header(expected_header):
        raise RuntimeError(
            f"Inference checkpoint does not match this run: {path}"
        )

    records = []
    for line in lines[1:]:
        try:
            record = json.loads(line)
        except (TypeError, ValueError):
            break
        if (
            not isinstance(record, dict)
            or record.get("index") != len(records)
            or not isinstance(record.get("pred_entities"), list)
            or not isinstance(record.get("raw_output"), str)
            or not isinstance(record.get("parse_status"), str)
            or not isinstance(record.get("parse_diagnostics"), dict)
            or not isinstance(record.get("latency_ms"), (int, float))
            or not isinstance(record.get("elapsed_seconds"), (int, float))
        ):
            break
        records.append(record)
    return records


def _write_inference_checkpoint(
    path: Path,
    header: Dict[str, Any],
    records: List[Dict[str, Any]],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, ensure_ascii=False) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# Inferenz
# ---------------------------------------------------------------------------

def run_decoder_inference(
    adapter_path:    str | None,
    base_model_name: str | None,
    config_path:     str,
    zeroshot:        bool = False,
) -> Dict[str, Any]:
    """Fuehrt NER-Inferenz mit einem LLM auf dem Test-Set durch.

    Unterstuetzt zwei Modi:
      - LoRA/QLoRA: Basismodell + Adapter werden geladen (Default).
      - Zero-Shot:  Nur das Basismodell wird geladen, ohne Adapter.

    Ablauf pro Sample (identisch in beiden Modi):
      1. Prompt (system + user) mit apply_chat_template() tokenisieren
      2. model.generate() mit greedy decoding aufrufen
      3. Nur die neu generierten Tokens dekodieren (Prompt-Tokens ausschliessen)
      4. JSON aus dem Rohtext parsen (mit Fallback-Strategien)
      5. Entities in BIO-Tags umwandeln fuer seqeval

    Args:
        adapter_path:     Pfad zum LoRA-Adapter (None bei Zero-Shot).
        base_model_name:  Optionaler HuggingFace Model-ID Override; muss zur Config passen.
        config_path:      Pfad zur YAML-Config.
        zeroshot:         True = ohne Adapter laufen lassen.

    Returns:
        Dict mit f1, precision, recall, parse_failure_rate, latency, vram, regime.
    """
    cfg = load_experiment_config(config_path, expected_model_type="decoder")

    # --- Modus aus Config oder CLI bestimmen ---
    cfg_mode = str(cfg.get("mode", "lora")).lower()
    is_zeroshot = cfg_mode == "zeroshot"
    regime_label = "llm_zeroshot" if is_zeroshot else "llm_lora"

    if zeroshot and not is_zeroshot:
        raise ValueError("--zeroshot can only be used with a config that sets mode: zeroshot.")
    if is_zeroshot and adapter_path:
        raise ValueError("Zero-shot inference must not receive --adapter.")
    if not is_zeroshot and not adapter_path:
        raise ValueError(
            "LoRA/QLoRA-Inferenz benoetigt --adapter. "
            "Fuer Zero-Shot bitte eine mode: zeroshot Config verwenden."
        )

    configured_model_name = str(cfg["model_name"])
    if base_model_name is not None and base_model_name != configured_model_name:
        raise ValueError(
            f"--base {base_model_name!r} does not match config model_name {configured_model_name!r}."
        )
    base_model_name = configured_model_name

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    mode_str = "Zero-Shot" if is_zeroshot else "LoRA/QLoRA"
    console.rule(
        f"[bold cyan]Decoder-Inferenz ({mode_str}): {cfg['experiment_name']} on MultiNERD English[/bold cyan]"
    )

    use_qlora:                bool = cfg.get("use_qlora", True)
    attn_impl:                str  = cfg.get("attn_impl", "sdpa")
    max_new_tokens:           int  = int(cfg.get("max_new_tokens", 256))
    generation_batch_size:    int  = int(cfg.get("inference_batch_size", 1))
    checkpoint_sync_interval: int  = int(cfg.get("checkpoint_sync_interval", 100))
    if generation_batch_size < 1:
        raise ValueError("inference_batch_size must be >= 1")
    if checkpoint_sync_interval < 1:
        raise ValueError("checkpoint_sync_interval must be >= 1")
    precision_mode = "qlora_4bit" if use_qlora else "bf16"

    # --- Tokenizer laden ---
    # Bei Zero-Shot direkt vom Basismodell, sonst vom Adapter-Verzeichnis
    tokenizer_source = base_model_name if is_zeroshot else adapter_path
    console.print(f"[cyan]Lade Tokenizer von {tokenizer_source}...[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # --- Basismodell laden ---
    console.print(f"[cyan]Lade Basismodell: {base_model_name}[/cyan]")
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            attn_implementation=attn_impl,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map="auto",
            trust_remote_code=True,
        )

    # --- Modell vorbereiten: LoRA-Adapter laden ODER Basismodell direkt nutzen ---
    if is_zeroshot:
        console.print("[cyan]Zero-Shot Mode: kein Adapter wird geladen.[/cyan]")
        model = base_model
    else:
        from peft import PeftModel

        console.print(f"[cyan]Lade LoRA-Adapter von {adapter_path}...[/cyan]")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    total_params, trainable_params = count_parameters(model)
    console.print(f"Parameter: {total_params:,} gesamt, {trainable_params:,} trainierbar")

    device = next(model.parameters()).device

    # --- Test-Daten laden ---
    console.print("[cyan]Loading MultiNERD English test split...[/cyan]")
    raw_dataset, info = load_ner_dataset()
    raw_test = raw_dataset["test"]
    prompts, gold_entities = prepare_test_inputs(raw_test, info)
    tokens_list: List[List[str]] = [s["tokens"] for s in raw_test]
    valid_types = frozenset(info.entity_types)

    output_dir = output_dir_from_config(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / INFERENCE_CHECKPOINT_FILENAME
    checkpoint_header = {
        "version": INFERENCE_CHECKPOINT_VERSION,
        "experiment_name": cfg["experiment_name"],
        "model_name": base_model_name,
        "regime": regime_label,
        "thinking_enabled": THINKING_ENABLED,
        "max_new_tokens": max_new_tokens,
        "generation_batch_size": generation_batch_size,
        "precision_mode": precision_mode,
        "attn_impl": attn_impl,
        "seed": seed,
        "total_samples": len(prompts),
    }
    checkpoint_records = _load_inference_checkpoint(
        checkpoint_path,
        checkpoint_header,
    )
    _write_inference_checkpoint(
        checkpoint_path,
        checkpoint_header,
        checkpoint_records,
    )
    if checkpoint_records:
        console.print(
            f"[yellow]Setze Inferenz bei Sample {len(checkpoint_records):,}/"
            f"{len(prompts):,} fort: {checkpoint_path}[/yellow]"
        )

    # --- Warmup-Lauf ---
    reset_vram_tracking()
    _warmup(model, tokenizer, prompts[0], device, max_new_tokens)

    # --- Inferenz-Schleife ---
    pred_entities: List[List[Dict]] = [
        record["pred_entities"] for record in checkpoint_records
    ]
    parse_statuses: List[str] = [
        record["parse_status"] for record in checkpoint_records
    ]
    parse_diagnostics: List[Dict[str, int]] = [
        record["parse_diagnostics"] for record in checkpoint_records
    ]
    raw_outputs: List[str] = [
        record["raw_output"] for record in checkpoint_records
    ]
    latencies_ms: List[float] = [
        float(record["latency_ms"]) for record in checkpoint_records
    ]
    previous_elapsed = (
        float(checkpoint_records[-1]["elapsed_seconds"])
        if checkpoint_records
        else 0.0
    )
    start_index = len(checkpoint_records)

    console.print(
        f"\n[cyan]Generiere fuer {len(prompts)} Test-Samples "
        f"(Batch-Groesse {generation_batch_size}, Checkpoint-Sync alle "
        f"{checkpoint_sync_interval} Samples)...[/cyan]"
    )
    inference_started = time.perf_counter()
    total_generation_seconds = sum(latencies_ms)
    if checkpoint_records:
        total_generation_seconds /= 1000.0

    with checkpoint_path.open("a", encoding="utf-8") as checkpoint_handle:
        records_since_sync = 0
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Generiere...",
                total=len(prompts),
                completed=start_index,
            )

            for batch_start in range(
                start_index,
                len(prompts),
                generation_batch_size,
            ):
                batch_end = min(
                    batch_start + generation_batch_size,
                    len(prompts),
                )
                batch_prompts = prompts[batch_start:batch_end]
                model_inputs = prepare_generation_batch_inputs(
                    tokenizer,
                    batch_prompts,
                    device,
                )

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                with torch.no_grad():
                    output_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                batch_generation_seconds = t1 - t0
                total_generation_seconds += batch_generation_seconds
                batch_length = batch_end - batch_start
                amortized_latency_ms = (
                    batch_generation_seconds * 1000.0 / batch_length
                )

                prompt_len = model_inputs["input_ids"].shape[1]
                generated_texts = tokenizer.batch_decode(
                    output_ids[:, prompt_len:],
                    skip_special_tokens=True,
                )

                for offset, generated_text in enumerate(generated_texts):
                    i = batch_start + offset
                    tokens = tokens_list[i]
                    raw_outputs.append(generated_text)

                    entities, status, diagnostics = (
                        parse_llm_output_with_diagnostics(
                            generated_text,
                            tokens,
                            valid_types,
                        )
                    )
                    pred_entities.append(entities)
                    parse_statuses.append(status)
                    parse_diagnostics.append(diagnostics)
                    latencies_ms.append(amortized_latency_ms)

                    completed = i + 1
                    elapsed_seconds = (
                        previous_elapsed + time.perf_counter() - inference_started
                    )
                    record = {
                        "index": i,
                        "pred_entities": entities,
                        "raw_output": generated_text,
                        "parse_status": status,
                        "parse_diagnostics": diagnostics,
                        "latency_ms": amortized_latency_ms,
                        "batch_latency_ms": batch_generation_seconds * 1000.0,
                        "elapsed_seconds": elapsed_seconds,
                    }
                    checkpoint_handle.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
                    records_since_sync += 1

                completed = batch_end
                if (
                    records_since_sync >= checkpoint_sync_interval
                    or completed == len(prompts)
                ):
                    checkpoint_handle.flush()
                    os.fsync(checkpoint_handle.fileno())
                    records_since_sync = 0
                if completed % 100 == 0 or completed == len(prompts):
                    console.print(
                        f"INFERENCE_PROGRESS {completed}/{len(prompts)} "
                        f"elapsed={elapsed_seconds:.3f}s"
                    )
                progress.advance(task, advance=batch_length)

    vram_peak = get_vram_peak_mb()

    # --- Evaluation ---
    metrics = evaluate_llm_predictions(
        tokens_list=tokens_list,
        gold_entities=gold_entities,
        pred_entities=pred_entities,
        parse_statuses=parse_statuses,
        parse_diagnostics=parse_diagnostics,
    )

    latency_mean = float(np.mean(latencies_ms))
    latency_p95  = float(np.percentile(latencies_ms, 95))
    generation_samples_per_second = (
        len(prompts) / total_generation_seconds
        if total_generation_seconds > 0
        else 0.0
    )

    console.print(f"\n[bold green]Test F1: {metrics['f1']:.4f}[/bold green]")
    console.print(f"Precision: {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}")
    console.print(f"Parse-Fehlerrate: {metrics['parse_failure_rate']:.3f}")
    console.print(
        f"  ok={metrics['parse_ok']}  markdown={metrics['parse_markdown_stripped']}"
        f"  regex={metrics['parse_regex_fallback']}  failed={metrics['parse_failed']}"
    )
    console.print(f"Mittlere Latenz: {latency_mean:.2f} ms  (p95: {latency_p95:.2f} ms)")
    console.print(
        f"Generierungsdurchsatz: {generation_samples_per_second:.2f} Samples/s"
    )
    console.print(f"VRAM-Peak: {vram_peak:.1f} MB")

    # --- Ausgaben speichern ---
    saved_samples = [
        {
            "tokens":        tokens,
            "gold_entities": gold,
            "pred_entities": pred,
            "raw_output":    raw,
            "parse_status":  status,
            "parse_diagnostics": diagnostics,
            "gold_bio":      entities_to_bio(tokens, gold),
            "pred_bio":      entities_to_bio(tokens, pred),
        }
        for tokens, gold, pred, raw, status, diagnostics in zip(
            tokens_list, gold_entities, pred_entities, raw_outputs, parse_statuses, parse_diagnostics
        )
    ]
    pred_file = output_dir / "test_predictions.json"
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(saved_samples, f, ensure_ascii=False, indent=2)
    console.print(f"\nVorhersagen gespeichert: {pred_file}")

    full_metrics: Dict[str, Any] = {
        "experiment_name": cfg["experiment_name"],
        "model_name":      base_model_name,
        "model_type":      "decoder",
        "regime":          regime_label,  # llm_zeroshot oder llm_lora
        "dataset":         DATASET_NAME,
        "dataset_language": DATASET_LANGUAGE,
        # seqeval-Metriken sind in der Tabelle als test_f1 etc. erwartet
        "test_f1":         metrics["f1"],
        "test_precision":  metrics["precision"],
        "test_recall":     metrics["recall"],
        **metrics,
        "latency_ms_mean": latency_mean,
        "latency_ms_p95":  latency_p95,
        "latency_measurement": "amortized_per_sample",
        "generation_samples_per_second": generation_samples_per_second,
        "inference_batch_size": generation_batch_size,
        "checkpoint_sync_interval": checkpoint_sync_interval,
        "precision_mode": precision_mode,
        "attn_impl": attn_impl,
        "vram_peak_mb":    vram_peak,
        "total_params":    total_params,
        "thinking_enabled": THINKING_ENABLED,
        "seed":            seed,
        "max_new_tokens":  max_new_tokens,
        "run_metadata":    collect_run_metadata(cfg),
    }
    inf_file = output_dir / "inference_metrics.yaml"
    with open(inf_file, "w") as f:
        yaml.dump(full_metrics, f, default_flow_style=False)
    console.print(f"Inferenz-Metriken gespeichert: {inf_file}")
    checkpoint_path.unlink(missing_ok=True)

    return full_metrics


# ---------------------------------------------------------------------------
# Warmup-Hilfsfunktion
# ---------------------------------------------------------------------------

def _warmup(model, tokenizer, sample_messages: List[Dict], device, max_new_tokens: int) -> None:
    """Fuehrt einen einzelnen Generate-Aufruf zur Aufwaermung der CUDA-Caches durch."""
    console.print("[dim]Waerme CUDA-Caches auf...[/dim]")
    model_inputs = prepare_generation_inputs(tokenizer, sample_messages, device)
    with torch.no_grad():
        model.generate(
            **model_inputs,
            max_new_tokens=16,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# CLI-Einstiegspunkt
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decoder-NER Inferenz (LoRA oder Zero-Shot)")
    parser.add_argument("--adapter",  default=None,  help="Pfad zum LoRA-Adapter (entfaellt bei --zeroshot)")
    parser.add_argument("--base",     default=None, help="Optionaler Basismodell-Name; muss zur Config passen")
    parser.add_argument("--config",   required=True, help="Pfad zur YAML-Config")
    parser.add_argument(
        "--zeroshot",
        action="store_true",
        help="Zero-Shot Mode: Basismodell ohne Adapter laden",
    )
    args = parser.parse_args()

    run_decoder_inference(
        adapter_path=args.adapter,
        base_model_name=args.base,
        config_path=args.config,
        zeroshot=args.zeroshot,
    )
