"""
inference.py — LLM-Inferenz (Qwen3.5 + LoRA) auf dem Test-Set

Laedt das Basismodell und den trainierten LoRA-Adapter, generiert fuer jeden
Test-Satz eine JSON-Entity-Liste, parst die Ausgabe und bewertet sie mit seqeval.

Greedy Decoding (do_sample=False) wird fuer Reproduzierbarkeit verwendet.

Verwendung:
    python -m src.decoder.inference \
        --adapter results/multinerd/qwen35-4b-qlora/best_lora_adapter \
        --base Qwen/Qwen3.5-4B \
        --config configs/qwen35_4b.yaml

    python -m src.decoder.inference \
        --adapter results/wnut_17/qwen35-4b-qlora/best_lora_adapter \
        --base Qwen/Qwen3.5-4B \
        --config configs/qwen35_4b.yaml \
        --dataset wnut_17
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from peft import PeftModel
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data.dataset_loader import load_ner_dataset
from src.data.preprocess_decoder import build_system_prompt, extract_entities_from_bio, prepare_test_inputs
from src.decoder.parse_output import (
    entities_to_bio,
    evaluate_llm_predictions,
    parse_llm_output,
)
from src.evaluate.efficiency import count_parameters, get_vram_peak_mb, reset_vram_tracking

console = Console()


# ---------------------------------------------------------------------------
# Inferenz
# ---------------------------------------------------------------------------

def run_decoder_inference(
    adapter_path:    str,
    base_model_name: str,
    config_path:     str,
    dataset_override: str | None = None,
) -> Dict[str, Any]:
    """Fuehrt NER-Inferenz mit dem LoRA-Fine-Tuned LLM auf dem Test-Set durch.

    Ablauf pro Sample:
      1. Prompt (system + user) mit apply_chat_template() tokenisieren
      2. model.generate() mit greedy decoding aufrufen
      3. Nur die neu generierten Tokens dekodieren (Prompt-Tokens ausschliessen)
      4. JSON aus dem Rohtext parsen (mit Fallback-Strategien)
      5. Entities in BIO-Tags umwandeln fuer seqeval

    Args:
        adapter_path:     Pfad zum gespeicherten LoRA-Adapter-Verzeichnis.
        base_model_name:  HuggingFace Model-ID des Basismodells.
        config_path:      Pfad zur YAML-Config.
        dataset_override: Optionaler Datensatz-Override.

    Returns:
        Dict mit f1, precision, recall, parse_failure_rate, latency, vram.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset_name = dataset_override or cfg.get("dataset", "multinerd")
    dataset_language = cfg.get("dataset_language", "en")

    console.rule(f"[bold cyan]Decoder-Inferenz: {cfg['experiment_name']} auf {dataset_name}[/bold cyan]")

    use_qlora:      bool = cfg.get("use_qlora", True)
    attn_impl:      str  = cfg.get("attn_impl", "sdpa")
    max_new_tokens: int  = 256

    # --- Tokenizer laden ---
    console.print(f"[cyan]Lade Tokenizer von {adapter_path}...[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

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

    # --- LoRA-Adapter laden ---
    console.print(f"[cyan]Lade LoRA-Adapter von {adapter_path}...[/cyan]")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    total_params, trainable_params = count_parameters(model)
    console.print(f"Parameter: {total_params:,} gesamt, {trainable_params:,} trainierbar")

    device = next(model.parameters()).device

    # --- Test-Daten laden ---
    console.print(f"[cyan]Lade {dataset_name} Test-Split...[/cyan]")
    raw_dataset, info = load_ner_dataset(dataset_name, language=dataset_language)
    raw_test = raw_dataset["test"]
    prompts, gold_entities = prepare_test_inputs(raw_test, info)
    tokens_list: List[List[str]] = [s["tokens"] for s in raw_test]
    valid_types = frozenset(info.entity_types)

    # --- Warmup-Lauf ---
    reset_vram_tracking()
    _warmup(model, tokenizer, prompts[0], device, max_new_tokens)

    # --- Inferenz-Schleife ---
    pred_entities:  List[List[Dict]] = []
    parse_statuses: List[str]        = []
    raw_outputs:    List[str]        = []
    latencies_ms:   List[float]      = []

    console.print(f"\n[cyan]Generiere fuer {len(prompts)} Test-Samples...[/cyan]")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generiere...", total=len(prompts))

        for i, (messages, tokens) in enumerate(zip(prompts, tokens_list)):
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
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
            latencies_ms.append((t1 - t0) * 1000)

            prompt_len     = input_ids.shape[1]
            new_token_ids  = output_ids[0][prompt_len:]
            generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)

            raw_outputs.append(generated_text)

            entities, status = parse_llm_output(generated_text, valid_types)
            pred_entities.append(entities)
            parse_statuses.append(status)

            progress.advance(task)

    vram_peak = get_vram_peak_mb()

    # --- Evaluation ---
    metrics = evaluate_llm_predictions(
        tokens_list=tokens_list,
        gold_entities=gold_entities,
        pred_entities=pred_entities,
        parse_statuses=parse_statuses,
    )

    latency_mean = float(np.mean(latencies_ms))
    latency_p95  = float(np.percentile(latencies_ms, 95))

    console.print(f"\n[bold green]Test F1: {metrics['f1']:.4f}[/bold green]")
    console.print(f"Precision: {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}")
    console.print(f"Parse-Fehlerrate: {metrics['parse_failure_rate']:.3f}")
    console.print(
        f"  ok={metrics['parse_ok']}  markdown={metrics['parse_markdown_stripped']}"
        f"  regex={metrics['parse_regex_fallback']}  failed={metrics['parse_failed']}"
    )
    console.print(f"Mittlere Latenz: {latency_mean:.2f} ms  (p95: {latency_p95:.2f} ms)")
    console.print(f"VRAM-Peak: {vram_peak:.1f} MB")

    # --- Ausgaben speichern ---
    output_dir = Path(cfg.get("output_dir", f"results/{dataset_name}/{cfg['experiment_name']}"))
    if dataset_override:
        output_dir = Path(f"results/{dataset_override}/{cfg['experiment_name']}")

    saved_samples = [
        {
            "tokens":        tokens,
            "gold_entities": gold,
            "pred_entities": pred,
            "raw_output":    raw,
            "parse_status":  status,
            "gold_bio":      entities_to_bio(tokens, gold),
            "pred_bio":      entities_to_bio(tokens, pred),
        }
        for tokens, gold, pred, raw, status in zip(
            tokens_list, gold_entities, pred_entities, raw_outputs, parse_statuses
        )
    ]
    pred_file = output_dir / "test_predictions.json"
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(saved_samples, f, ensure_ascii=False, indent=2)
    console.print(f"\nVorhersagen gespeichert: {pred_file}")

    full_metrics: Dict[str, Any] = {
        "experiment_name": cfg["experiment_name"],
        "model_type":      "decoder",
        "dataset":         dataset_name,
        **metrics,
        "latency_ms_mean": latency_mean,
        "latency_ms_p95":  latency_p95,
        "vram_peak_mb":    vram_peak,
        "total_params":    total_params,
    }
    inf_file = output_dir / "inference_metrics.yaml"
    with open(inf_file, "w") as f:
        yaml.dump(full_metrics, f, default_flow_style=False)
    console.print(f"Inferenz-Metriken gespeichert: {inf_file}")

    return full_metrics


# ---------------------------------------------------------------------------
# Warmup-Hilfsfunktion
# ---------------------------------------------------------------------------

def _warmup(model, tokenizer, sample_messages: List[Dict], device, max_new_tokens: int) -> None:
    """Fuehrt einen einzelnen Generate-Aufruf zur Aufwaermung der CUDA-Caches durch."""
    console.print("[dim]Waerme CUDA-Caches auf...[/dim]")
    input_ids = tokenizer.apply_chat_template(
        sample_messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        model.generate(
            input_ids,
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
    parser = argparse.ArgumentParser(description="Decoder-NER Inferenz")
    parser.add_argument("--adapter", required=True, help="Pfad zum LoRA-Adapter-Verzeichnis")
    parser.add_argument("--base",    required=True, help="Basismodell-Name oder -Pfad")
    parser.add_argument("--config",  required=True, help="Pfad zur YAML-Config")
    parser.add_argument("--dataset", default=None,  help="Datensatz-Override (z.B. 'wnut_17')")
    args = parser.parse_args()

    run_decoder_inference(
        adapter_path=args.adapter,
        base_model_name=args.base,
        config_path=args.config,
        dataset_override=args.dataset,
    )
