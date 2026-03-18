"""
inference.py — LLM-Inferenz (Qwen3 / Qwen3.5 + LoRA) auf WNUT-2017 Test-Set

Lädt das Basismodell und den trainierten LoRA-Adapter, generiert für jeden
Test-Satz eine JSON-Entity-Liste, parst die Ausgabe und bewertet sie mit seqeval.

Greedy Decoding (do_sample=False) wird für Reproduzierbarkeit verwendet —
kein Sampling, keine Temperatur-Variation.

Verwendung:
    python -m src.decoder.inference \\
        --adapter results/qwen35-27b-lora/lora_adapter \\
        --base Qwen/Qwen3.5-27B \\
        --config configs/qwen35_27b.yaml
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

from src.data.load_wnut17 import load_wnut17
from src.data.preprocess_decoder import prepare_test_inputs
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
) -> Dict[str, Any]:
    """Führt NER-Inferenz mit dem LoRA-Fine-Tuned LLM auf dem Test-Set durch.

    Ablauf pro Sample:
      1. Prompt (system + user) mit apply_chat_template() tokenisieren
      2. model.generate() mit greedy decoding aufrufen
      3. Nur die neu generierten Tokens dekodieren (Prompt-Tokens ausschließen)
      4. JSON aus dem Rohtext parsen (mit Fallback-Strategien)
      5. Entities in BIO-Tags umwandeln für seqeval

    Args:
        adapter_path:    Pfad zum gespeicherten LoRA-Adapter-Verzeichnis.
        base_model_name: HuggingFace Model-ID des Basismodells.
        config_path:     Pfad zur YAML-Config.

    Returns:
        Dict mit f1, precision, recall, parse_failure_rate, latency, vram.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    console.rule(f"[bold cyan]Decoder-Inferenz: {cfg['experiment_name']}[/bold cyan]")

    use_qlora:      bool = cfg.get("use_qlora", False)
    attn_impl:      str  = cfg.get("attn_impl", "sdpa")
    max_new_tokens: int  = 256  # Maximale Länge der generierten JSON-Antwort

    # --- Tokenizer laden ---
    # Tokenizer aus dem Adapter-Verzeichnis laden (enthält ggf. angepasste Tokens)
    console.print(f"[cyan]Lade Tokenizer von {adapter_path}...[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Basismodell laden ---
    console.print(f"[cyan]Lade Basismodell: {base_model_name}[/cyan]")
    if use_qlora:
        # QLoRA: 4-bit quantisiertes Basismodell (für kleinere GPUs)
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
        # bf16: Basismodell in halber Präzision (für A100 80GB)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map="auto",
            trust_remote_code=True,
        )

    # --- LoRA-Adapter auf Basismodell laden ---
    # PeftModel.from_pretrained() friert die Basisgewichte ein und
    # aktiviert nur den trainierten Adapter für Forward-Passes.
    console.print(f"[cyan]Lade LoRA-Adapter von {adapter_path}...[/cyan]")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()  # Dropout deaktivieren

    total_params, trainable_params = count_parameters(model)
    console.print(f"Parameter: {total_params:,} gesamt, {trainable_params:,} trainierbar")

    # Gerät aus dem ersten Modell-Parameter bestimmen
    device = next(model.parameters()).device

    # --- Test-Daten laden ---
    console.print("[cyan]Lade WNUT-2017 Test-Split...[/cyan]")
    raw_dataset = load_wnut17()
    raw_test    = raw_dataset["test"]
    # prompts = [system + user] ohne Assistent-Turn
    # gold_entities = Gold-Entities pro Satz für die Evaluation
    prompts, gold_entities = prepare_test_inputs(raw_test)
    tokens_list: List[List[str]] = [s["tokens"] for s in raw_test]

    # --- Warmup-Lauf ---
    # Erstes Generate ist langsamer (CUDA JIT, Cache-Aufbau) → Warmup
    # verhindert Verzerrung der Latenz-Messungen
    reset_vram_tracking()
    _warmup(model, tokenizer, prompts[0], device, max_new_tokens)

    # --- Inferenz-Schleife ---
    pred_entities:  List[List[Dict]] = []
    parse_statuses: List[str]        = []
    raw_outputs:    List[str]        = []
    latencies_ms:   List[float]      = []

    console.print(f"\n[cyan]Generiere für {len(prompts)} Test-Samples...[/cyan]")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generiere...", total=len(prompts))

        for i, (messages, tokens) in enumerate(zip(prompts, tokens_list)):
            # Chat-Template anwenden: fügt automatisch <|im_start|>assistant\n hinzu,
            # damit das Modell weiß, dass es jetzt generieren soll
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,  # Assistent-Turn-Beginn einfügen
                return_tensors="pt",
            ).to(device)

            # Latenz mit CUDA-Synchronisierung messen
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,     # Greedy Decoding für Reproduzierbarkeit
                    temperature=None,    # Muss None sein wenn do_sample=False
                    top_p=None,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000)

            # Nur die neu generierten Tokens dekodieren (Prompt-Tokens ausblenden)
            prompt_len     = input_ids.shape[1]
            new_token_ids  = output_ids[0][prompt_len:]
            generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)

            raw_outputs.append(generated_text)

            # JSON parsen — mit Fallback-Strategien
            entities, status = parse_llm_output(generated_text)
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
    output_dir = Path(cfg.get("output_dir", f"results/{cfg['experiment_name']}"))

    # Vollständige Vorhersagen für die Fehleranalyse speichern
    # Enthält: Tokens, Gold-Entities, Pred-Entities, Rohtext, Parse-Status, BIO-Tags
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

    # Inferenz-Metriken als YAML (für compare_all.py)
    full_metrics: Dict[str, Any] = {
        "experiment_name": cfg["experiment_name"],
        "model_type":      "decoder",
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
    """Führt einen einzelnen Generate-Aufruf zur Aufwärmung der CUDA-Caches durch.

    Ohne Warmup wäre der erste Latenz-Messwert durch JIT-Kompilierung
    und Cache-Initialisierung stark verfälscht.
    """
    console.print("[dim]Wärme CUDA-Caches auf...[/dim]")
    input_ids = tokenizer.apply_chat_template(
        sample_messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        model.generate(
            input_ids,
            max_new_tokens=16,      # Nur wenige Tokens für den Warmup
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
    parser = argparse.ArgumentParser(description="Decoder-NER Inferenz auf WNUT-2017 Test-Set")
    parser.add_argument("--adapter", required=True, help="Pfad zum LoRA-Adapter-Verzeichnis")
    parser.add_argument("--base",    required=True, help="Basismodell-Name oder -Pfad")
    parser.add_argument("--config",  required=True, help="Pfad zur YAML-Config")
    args = parser.parse_args()

    run_decoder_inference(
        adapter_path=args.adapter,
        base_model_name=args.base,
        config_path=args.config,
    )
