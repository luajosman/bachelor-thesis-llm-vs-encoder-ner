"""
Decoder (LLM + LoRA) inference for NER on the WNUT-2017 test split.

Loads the LoRA adapter on top of the base model, generates JSON entity
lists for each test sentence, parses them, and evaluates with seqeval.

Usage:
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
# Main inference function
# ---------------------------------------------------------------------------


def run_decoder_inference(
    adapter_path: str,
    base_model_name: str,
    config_path: str,
) -> Dict[str, Any]:
    """Run NER inference with a LoRA-fine-tuned causal LM on WNUT-2017 test.

    Steps:
    1. Load base model (quantized if use_qlora) + LoRA adapter.
    2. Build prompt-only inputs (system + user, no assistant turn).
    3. Generate with greedy decoding (do_sample=False).
    4. Parse JSON output with fallback strategies.
    5. Evaluate with seqeval.
    6. Save raw outputs and parsed predictions for error analysis.

    Parameters
    ----------
    adapter_path:
        Path to the saved LoRA adapter directory.
    base_model_name:
        HuggingFace model id or path for the base model.
    config_path:
        Path to the YAML config.

    Returns
    -------
    Dict[str, Any]
        Metrics dict: f1, precision, recall, parse_failure_rate, latency, vram.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    console.rule(f"[bold cyan]Decoder Inference: {cfg['experiment_name']}[/bold cyan]")

    use_qlora: bool = cfg.get("use_qlora", False)
    attn_impl: str = cfg.get("attn_impl", "sdpa")
    max_new_tokens: int = 256

    # ---- Tokenizer ----
    console.print(f"[cyan]Loading tokenizer from {adapter_path}...[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- Base model ----
    console.print(f"[cyan]Loading base model: {base_model_name}[/cyan]")
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

    # ---- Load LoRA adapter ----
    console.print(f"[cyan]Loading LoRA adapter from {adapter_path}...[/cyan]")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    total_params, trainable_params = count_parameters(model)
    console.print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Determine device from first parameter
    device = next(model.parameters()).device

    # ---- Load dataset ----
    console.print("[cyan]Loading WNUT-2017 test split...[/cyan]")
    raw_dataset = load_wnut17()
    raw_test = raw_dataset["test"]
    prompts, gold_entities = prepare_test_inputs(raw_test)

    tokens_list: List[List[str]] = [s["tokens"] for s in raw_test]

    # ---- Warmup run ----
    reset_vram_tracking()
    _warmup(model, tokenizer, prompts[0], device, max_new_tokens)

    # ---- Inference loop ----
    pred_entities: List[List[Dict]] = []
    parse_statuses: List[str] = []
    raw_outputs: List[str] = []
    latencies_ms: List[float] = []

    console.print(f"\n[cyan]Running inference on {len(prompts)} test samples...[/cyan]")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating...", total=len(prompts))

        for i, (messages, tokens) in enumerate(zip(prompts, tokens_list)):
            # Build input with chat template (no assistant turn yet)
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)

            # Measure latency
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

            # Decode only the new tokens (exclude the prompt)
            prompt_len = input_ids.shape[1]
            new_token_ids = output_ids[0][prompt_len:]
            generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)

            raw_outputs.append(generated_text)

            # Parse JSON
            entities, status = parse_llm_output(generated_text)
            pred_entities.append(entities)
            parse_statuses.append(status)

            progress.advance(task)

    vram_peak = get_vram_peak_mb()

    # ---- Evaluate ----
    metrics = evaluate_llm_predictions(
        tokens_list=tokens_list,
        gold_entities=gold_entities,
        pred_entities=pred_entities,
        parse_statuses=parse_statuses,
    )

    latency_mean = float(np.mean(latencies_ms))
    latency_p95 = float(np.percentile(latencies_ms, 95))

    console.print(f"\n[bold green]Test F1: {metrics['f1']:.4f}[/bold green]")
    console.print(f"Precision: {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}")
    console.print(f"Parse failure rate: {metrics['parse_failure_rate']:.3f}")
    console.print(f"  ok={metrics['parse_ok']}  markdown_stripped={metrics['parse_markdown_stripped']}"
                  f"  regex_fallback={metrics['parse_regex_fallback']}  failed={metrics['parse_failed']}")
    console.print(f"Mean latency: {latency_mean:.2f} ms  (p95: {latency_p95:.2f} ms)")
    console.print(f"VRAM peak: {vram_peak:.1f} MB")

    # ---- Save outputs ----
    output_dir = Path(cfg.get("output_dir", f"results/{cfg['experiment_name']}"))

    # Raw outputs + parsed predictions
    saved_samples = [
        {
            "tokens": tokens,
            "gold_entities": gold,
            "pred_entities": pred,
            "raw_output": raw,
            "parse_status": status,
            "gold_bio": entities_to_bio(tokens, gold),
            "pred_bio": entities_to_bio(tokens, pred),
        }
        for tokens, gold, pred, raw, status in zip(
            tokens_list, gold_entities, pred_entities, raw_outputs, parse_statuses
        )
    ]
    pred_file = output_dir / "test_predictions.json"
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(saved_samples, f, ensure_ascii=False, indent=2)
    console.print(f"\nPredictions saved to {pred_file}")

    # Inference metrics
    full_metrics: Dict[str, Any] = {
        "experiment_name": cfg["experiment_name"],
        "model_type": "decoder",
        **metrics,
        "latency_ms_mean": latency_mean,
        "latency_ms_p95": latency_p95,
        "vram_peak_mb": vram_peak,
        "total_params": total_params,
    }
    inf_file = output_dir / "inference_metrics.yaml"
    with open(inf_file, "w") as f:
        yaml.dump(full_metrics, f, default_flow_style=False)
    console.print(f"Inference metrics saved to {inf_file}")

    return full_metrics


# ---------------------------------------------------------------------------
# Warmup helper
# ---------------------------------------------------------------------------


def _warmup(model, tokenizer, sample_messages: List[Dict], device, max_new_tokens: int) -> None:
    """Run a single generation to warm up CUDA caches."""
    console.print("[dim]Warming up CUDA caches...[/dim]")
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
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decoder NER inference on WNUT-2017 test set")
    parser.add_argument("--adapter", required=True, help="Path to saved LoRA adapter directory")
    parser.add_argument("--base", required=True, help="Base model name or path")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    run_decoder_inference(
        adapter_path=args.adapter,
        base_model_name=args.base,
        config_path=args.config,
    )
