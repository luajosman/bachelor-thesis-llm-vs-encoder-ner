"""Benchmark decoder inference throughput without writing experiment results.

The benchmark uses a deterministic, evenly spaced subset of the test split and
compares generation batch sizes and model-loading precision. It never reads or
modifies the resumable checkpoint used by the real inference job.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

from src.config import load_experiment_config
from src.data.dataset_loader import load_ner_dataset
from src.data.preprocess_decoder import prepare_test_inputs
from src.decoder.generation import prepare_generation_batch_inputs
from src.decoder.parse_output import (
    evaluate_llm_predictions,
    parse_llm_output_with_diagnostics,
)


console = Console()
PRECISION_MODES = ("qlora_4bit", "bf16")


def _load_model(model_name: str, precision_mode: str, attn_impl: str):
    kwargs: dict[str, Any] = {
        "attn_implementation": attn_impl,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if precision_mode == "qlora_4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif precision_mode == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        raise ValueError(f"Unsupported precision mode: {precision_mode}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model


def _generate_subset(
    model,
    tokenizer,
    prompts,
    *,
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    outputs: list[str] = []
    batch_latencies_ms: list[float] = []

    warmup_prompts = prompts[: min(batch_size, len(prompts))]
    warmup_inputs = prepare_generation_batch_inputs(
        tokenizer,
        warmup_prompts,
        device,
    )
    with torch.inference_mode():
        model.generate(
            **warmup_inputs,
            max_new_tokens=min(16, max_new_tokens),
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        model_inputs = prepare_generation_batch_inputs(
            tokenizer,
            batch_prompts,
            device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_started = time.perf_counter()
        with torch.inference_mode():
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
        batch_latencies_ms.append(
            (time.perf_counter() - batch_started) * 1000.0
        )
        prompt_width = model_inputs["input_ids"].shape[1]
        outputs.extend(
            tokenizer.batch_decode(
                output_ids[:, prompt_width:],
                skip_special_tokens=True,
            )
        )
    elapsed_seconds = time.perf_counter() - started
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else 0.0
    )
    return {
        "batch_size": batch_size,
        "sample_count": len(prompts),
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": len(prompts) / elapsed_seconds,
        "amortized_latency_ms": elapsed_seconds * 1000.0 / len(prompts),
        "batch_latency_ms_mean": float(np.mean(batch_latencies_ms)),
        "batch_latency_ms_p95": float(np.percentile(batch_latencies_ms, 95)),
        "peak_vram_mb": peak_vram_mb,
        "outputs": outputs,
    }


def run_benchmark(
    config_path: str,
    output_path: str,
    sample_count: int,
    batch_sizes: list[int],
    precision_modes: list[str],
) -> dict[str, Any]:
    cfg = load_experiment_config(config_path, expected_model_type="decoder")
    if sample_count < 1:
        raise ValueError("sample_count must be >= 1")
    if not batch_sizes or any(batch_size < 1 for batch_size in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers")
    if any(mode not in PRECISION_MODES for mode in precision_modes):
        raise ValueError(f"precision_modes must be selected from {PRECISION_MODES}")

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    model_name = str(cfg["model_name"])
    attn_impl = str(cfg.get("attn_impl", "sdpa"))
    max_new_tokens = int(cfg.get("max_new_tokens", 256))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    raw_dataset, info = load_ner_dataset()
    raw_test = raw_dataset["test"]
    all_prompts, all_gold_entities = prepare_test_inputs(raw_test, info)
    sample_count = min(sample_count, len(all_prompts))
    indices = np.linspace(
        0,
        len(all_prompts) - 1,
        num=sample_count,
        dtype=int,
    ).tolist()
    prompts = [all_prompts[index] for index in indices]
    tokens_list = [raw_test[index]["tokens"] for index in indices]
    gold_entities = [all_gold_entities[index] for index in indices]
    valid_types = frozenset(info.entity_types)

    report: dict[str, Any] = {
        "experiment_name": cfg["experiment_name"],
        "model_name": model_name,
        "sample_count": sample_count,
        "sample_indices": indices,
        "max_new_tokens": max_new_tokens,
        "attn_impl": attn_impl,
        "seed": seed,
        "variants": [],
    }

    for precision_mode in precision_modes:
        console.rule(f"{precision_mode}: {model_name}")
        model = _load_model(model_name, precision_mode, attn_impl)
        precision_results = []
        baseline_outputs: list[str] | None = None
        baseline_entities: list[list[dict[str, Any]]] | None = None
        baseline_throughput: float | None = None

        for batch_size in batch_sizes:
            console.print(
                f"[cyan]Benchmark batch={batch_size}, samples={sample_count}[/cyan]"
            )
            try:
                result = _generate_subset(
                    model,
                    tokenizer,
                    prompts,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                )
            except torch.cuda.OutOfMemoryError:
                console.print(f"[red]Out of memory for batch={batch_size}[/red]")
                torch.cuda.empty_cache()
                precision_results.append({
                    "precision_mode": precision_mode,
                    "batch_size": batch_size,
                    "status": "out_of_memory",
                })
                continue

            outputs = result.pop("outputs")
            pred_entities = []
            parse_statuses = []
            parse_diagnostics = []
            for output, tokens in zip(outputs, tokens_list):
                entities, status, diagnostics = (
                    parse_llm_output_with_diagnostics(
                        output,
                        tokens,
                        valid_types,
                    )
                )
                pred_entities.append(entities)
                parse_statuses.append(status)
                parse_diagnostics.append(diagnostics)
            metrics = evaluate_llm_predictions(
                tokens_list=tokens_list,
                gold_entities=gold_entities,
                pred_entities=pred_entities,
                parse_statuses=parse_statuses,
                parse_diagnostics=parse_diagnostics,
            )
            if baseline_outputs is None:
                baseline_outputs = outputs
                baseline_entities = pred_entities
                baseline_throughput = result["samples_per_second"]
            raw_matches = sum(
                candidate == baseline
                for candidate, baseline in zip(outputs, baseline_outputs)
            )
            entity_matches = sum(
                candidate == baseline
                for candidate, baseline in zip(
                    pred_entities,
                    baseline_entities,
                )
            )
            result.update({
                "precision_mode": precision_mode,
                "status": "ok",
                "speedup_vs_first_batch_size": (
                    result["samples_per_second"] / baseline_throughput
                ),
                "exact_output_match_vs_first_batch_size": (
                    raw_matches / len(baseline_outputs)
                ),
                "parsed_entities_match_vs_first_batch_size": (
                    entity_matches / len(baseline_entities)
                ),
                "subset_f1": metrics["f1"],
                "subset_precision": metrics["precision"],
                "subset_recall": metrics["recall"],
                "parse_failure_rate": metrics["parse_failure_rate"],
            })
            precision_results.append(result)
            console.print(
                f"  {result['samples_per_second']:.3f} samples/s, "
                f"{result['speedup_vs_first_batch_size']:.2f}x, "
                f"raw match {result['exact_output_match_vs_first_batch_size']:.1%}, "
                f"entity match {result['parsed_entities_match_vs_first_batch_size']:.1%}, "
                f"F1 {result['subset_f1']:.3f}"
            )

        report["variants"].extend(precision_results)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    console.print(f"[green]Benchmark saved: {destination}[/green]")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark decoder inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 16, 32],
    )
    parser.add_argument(
        "--precision-modes",
        choices=PRECISION_MODES,
        nargs="+",
        default=list(PRECISION_MODES),
    )
    arguments = parser.parse_args()
    run_benchmark(
        config_path=arguments.config,
        output_path=arguments.output,
        sample_count=arguments.samples,
        batch_sizes=arguments.batch_sizes,
        precision_modes=arguments.precision_modes,
    )
