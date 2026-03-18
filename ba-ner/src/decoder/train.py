"""
train.py — Decoder-Training für NER mit LoRA / QLoRA via TRL SFTTrainer

Das LLM wird durch Supervised Fine-Tuning (SFT) auf dem ChatML-Format
trainiert. Nur der LoRA-Adapter wird aktualisiert; die Basisgewichte
bleiben eingefroren (Parameter-Efficient Fine-Tuning).

Zwei Modi:
  - Qwen3.5-27B: bf16 LoRA auf A100 (use_qlora: false, attn_impl: sdpa)
  - Qwen3-14B:   4-bit QLoRA auf 24GB+ GPU (use_qlora: true, attn_impl: flash_attention_2)

Wichtig für Qwen3.5: attn_implementation="sdpa" verwenden!
flash_attention_2 führt bei Qwen3.5 zu CUDA-Fehlern.

Verwendung:
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
# Training
# ---------------------------------------------------------------------------

def train_decoder(config_path: str) -> Dict[str, Any]:
    """Fine-tuned ein kausales LM für NER via LoRA (oder QLoRA) mit SFTTrainer.

    Ablauf:
      1. Config laden, Seeds setzen
      2. Tokenizer laden; pad_token auf eos_token setzen (Qwen hat keinen eigenen)
      3. Basismodell laden (bf16 oder 4-bit quantisiert)
      4. LoRA-Config bauen und Adapter ans Modell anhängen
      5. WNUT-2017 im ChatML-Format laden
      6. SFTTrainer konfigurieren und Training starten
      7. LoRA-Adapter und Tokenizer speichern

    Args:
        config_path: Pfad zur YAML-Konfigurationsdatei.

    Returns:
        Dict mit train_runtime_seconds, trainable_params, total_params.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    console.rule(f"[bold green]Decoder-Training: {cfg['experiment_name']}[/bold green]")

    # --- Reproduzierbarkeit ---
    seed: int = cfg.get("seed", 42)
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Ausgabeverzeichnisse ---
    output_dir  = Path(cfg.get("output_dir", f"results/{cfg['experiment_name']}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "lora_adapter"  # Hier wird nur der Adapter gespeichert

    model_name:     str = cfg["model_name"]
    use_qlora:      bool = cfg.get("use_qlora", False)
    attn_impl:      str  = cfg.get("attn_impl", "sdpa")
    max_seq_length: int  = cfg.get("max_seq_length", 512)

    # --- Tokenizer laden ---
    console.print(f"[cyan]Lade Tokenizer: {model_name}[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Qwen-Modelle haben kein dediziertes pad_token — eos_token als Ersatz setzen.
    # Ohne pad_token schlägt das Batching im Trainer fehl.
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Basismodell laden ---
    if use_qlora:
        # QLoRA: Modell in 4-bit laden, Berechnungen in bfloat16
        # Spart Speicher auf 24-GB-GPUs (z.B. für Qwen3-14B)
        console.print("[cyan]Lade Modell im 4-bit QLoRA-Modus...[/cyan]")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",         # NormalFloat4: beste Qualität für LLMs
            bnb_4bit_double_quant=True,        # Zusätzliche Kompression der Quantisierungsparameter
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
    else:
        # Kein QLoRA: Modell in bfloat16 laden (passt auf A100 80GB)
        # device_map="auto" verteilt das Modell automatisch auf verfügbare GPUs
        console.print(f"[cyan]Lade Modell in bf16 (attn={attn_impl})...[/cyan]")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map="auto",
            trust_remote_code=True,
        )

    # pad_token_id auch in der Modell-Config setzen (für generate())
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- LoRA-Konfiguration ---
    # Welche Gewichtsmatrizen durch LoRA angepasst werden, steht in der Config.
    # Standard: alle Attention- und MLP-Projektionen der Transformer-Blöcke.
    target_modules = cfg.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.get("lora_r", 16),             # Rang der Niedrigrang-Matrizen
        lora_alpha=cfg.get("lora_alpha", 32), # Skalierungsfaktor (alpha/r bestimmt Lernrate des Adapters)
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias="none",  # Bias-Parameter werden nicht trainiert
    )

    # LoRA-Adapter ans Modell anhängen; Basisgewichte werden eingefroren
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Zeigt trainierbare vs. gesamt Parameter

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Gesamt: {total_params:,} | Trainierbar: {trainable_params:,}")

    # --- Datensatz im ChatML-Format laden ---
    console.print("[cyan]Bereite Decoder-Datensatz vor...[/cyan]")
    dataset = prepare_decoder_dataset()

    # --- SFT-Konfiguration ---
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.get("epochs", 3),
        per_device_train_batch_size=cfg.get("batch_size", 2),
        per_device_eval_batch_size=cfg.get("batch_size", 2),
        # Gradient Accumulation: simuliert größere Batch-Größen mit wenig VRAM
        gradient_accumulation_steps=cfg.get("grad_accum", 8),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
        bf16=True,    # bfloat16 für Training (stabiler als fp16 bei großen Modellen)
        fp16=False,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,  # Bei SFT speichern wir nur den letzten Adapter
        max_seq_length=max_seq_length,
        seed=seed,
        report_to="wandb" if cfg.get("use_wandb", False) else "none",
        run_name=cfg.get("experiment_name"),
        dataset_text_field=None,  # Wir nutzen das messages-Format, kein reines Text-Feld
    )

    # --- SFTTrainer zusammenbauen und Training starten ---
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    console.print("[bold yellow]Starte LoRA-Finetuning...[/bold yellow]")
    train_start = time.perf_counter()
    trainer.train()
    train_runtime = time.perf_counter() - train_start

    # --- Nur den LoRA-Adapter speichern (nicht das gesamte Modell) ---
    # Die Basisgewichte bleiben auf HuggingFace Hub; nur der Adapter (~100-300 MB)
    # muss gespeichert werden.
    console.print(f"[green]Speichere LoRA-Adapter nach {adapter_dir}[/green]")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # --- Ergebnisse speichern ---
    results: Dict[str, Any] = {
        "experiment_name":      cfg["experiment_name"],
        "model_name":           model_name,
        "model_type":           "decoder",
        "use_qlora":            use_qlora,
        "lora_r":               cfg.get("lora_r", 16),
        "lora_alpha":           cfg.get("lora_alpha", 32),
        "train_runtime_seconds": float(train_runtime),
        "trainable_params":     trainable_params,
        "total_params":         total_params,
        "adapter_dir":          str(adapter_dir),
    }

    results_file = output_dir / "results.yaml"
    with open(results_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    console.print(f"Ergebnisse gespeichert: {results_file}")

    return results


# ---------------------------------------------------------------------------
# CLI-Einstiegspunkt
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[red]Verwendung: python -m src.decoder.train <config.yaml>[/red]")
        sys.exit(1)
    train_decoder(sys.argv[1])
