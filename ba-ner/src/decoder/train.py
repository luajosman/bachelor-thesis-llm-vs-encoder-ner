"""
Decoder-based NER training with LoRA / QLoRA via TRL SFTTrainer.

Supports:
  - Qwen3.5-27B with bf16 LoRA on A100 (use_qlora: false, attn_impl: sdpa)
  - Qwen3-14B with 4-bit QLoRA (use_qlora: true, attn_impl: flash_attention_2)

Usage:
    python -m src.decoder.train configs/qwen35_27b.yaml
    python -m src.decoder.train configs/qwen3_14b.yaml
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTConfig, SFTTrainer

from src.data.preprocess_decoder import prepare_decoder_dataset

console = Console()


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train_decoder(config_path: str) -> Dict[str, Any]:
    """Fine-tune a causal LM for NER using LoRA (or QLoRA) via SFTTrainer.

    Parameters
    ----------
    config_path:
        Path to a YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Results dict with train_runtime, trainable_params, total_params.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    console.rule(f"[bold green]Decoder Training: {cfg['experiment_name']}[/bold green]")

    # ---- Reproducibility ----
    seed: int = cfg.get("seed", 42)
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---- Output dir ----
    output_dir = Path(cfg.get("output_dir", f"results/{cfg['experiment_name']}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "lora_adapter"

    model_name: str = cfg["model_name"]
    use_qlora: bool = cfg.get("use_qlora", False)
    attn_impl: str = cfg.get("attn_impl", "sdpa")
    max_seq_length: int = cfg.get("max_seq_length", 512)

    # ---- Tokenizer ----
    console.print(f"[cyan]Loading tokenizer: {model_name}[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- Model loading ----
    if use_qlora:
        console.print("[cyan]Loading model in 4-bit QLoRA mode...[/cyan]")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
    else:
        console.print(f"[cyan]Loading model in bf16 mode (attn={attn_impl})...[/cyan]")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map="auto",
            trust_remote_code=True,
        )

    # Ensure pad token id is set in model config
    model.config.pad_token_id = tokenizer.pad_token_id

    # ---- LoRA config ----
    target_modules = cfg.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # ---- Dataset ----
    console.print("[cyan]Preparing decoder dataset...[/cyan]")
    dataset = prepare_decoder_dataset()

    # ---- SFT config ----
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.get("epochs", 3),
        per_device_train_batch_size=cfg.get("batch_size", 2),
        per_device_eval_batch_size=cfg.get("batch_size", 2),
        gradient_accumulation_steps=cfg.get("grad_accum", 8),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
        bf16=True,
        fp16=False,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        max_seq_length=max_seq_length,
        seed=seed,
        report_to="wandb" if cfg.get("use_wandb", False) else "none",
        run_name=cfg.get("experiment_name"),
        dataset_text_field=None,  # We use messages format
    )

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    # ---- Train ----
    console.print("[bold yellow]Starting LoRA fine-tuning...[/bold yellow]")
    train_start = time.perf_counter()
    trainer.train()
    train_runtime = time.perf_counter() - train_start

    # ---- Save LoRA adapter and tokenizer ----
    console.print(f"[green]Saving LoRA adapter to {adapter_dir}[/green]")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # ---- Save results ----
    results: Dict[str, Any] = {
        "experiment_name": cfg["experiment_name"],
        "model_name": model_name,
        "model_type": "decoder",
        "use_qlora": use_qlora,
        "lora_r": cfg.get("lora_r", 16),
        "lora_alpha": cfg.get("lora_alpha", 32),
        "train_runtime_seconds": float(train_runtime),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "adapter_dir": str(adapter_dir),
    }

    results_file = output_dir / "results.yaml"
    with open(results_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    console.print(f"Results saved to {results_file}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[red]Usage: python -m src.decoder.train <config.yaml>[/red]")
        sys.exit(1)
    train_decoder(sys.argv[1])
