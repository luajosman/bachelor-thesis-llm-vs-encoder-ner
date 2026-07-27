"""
train.py — Decoder-Training fuer NER mit LoRA / QLoRA via TRL SFTTrainer

Das LLM wird durch Supervised Fine-Tuning (SFT) auf dem ChatML-Format
trainiert. Nur der LoRA-Adapter wird aktualisiert; die Basisgewichte
bleiben eingefroren (Parameter-Efficient Fine-Tuning).

Zentrale Neuerung gegenueber reinem eval_loss:
  - Nach jeder Epoche wird eine generative Evaluation auf dem Dev-Set
    durchgefuehrt (model.generate() statt teacher-forced loss).
  - Der Best-Checkpoint wird ueber den generativen Dev-F1 ausgewaehlt.
  - Der beste LoRA-Adapter wird separat gespeichert.

Verwendung:
    python -m src.decoder.train configs/qwen35_4b.yaml
    python -m src.decoder.train configs/qwen35_27b.yaml
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from src.config import DATASET_LANGUAGE, DATASET_NAME, load_experiment_config, output_dir_from_config
from src.data.dataset_loader import load_ner_dataset
from src.data.preprocess_decoder import (
    build_system_prompt,
    extract_entities_from_bio,
    format_numbered_tokens,
    prepare_decoder_dataset,
)
from src.decoder.parse_output import (
    evaluate_llm_predictions,
    parse_llm_output_with_diagnostics,
)
from src.decoder.generation import THINKING_ENABLED, prepare_generation_inputs
from src.run_metadata import collect_run_metadata

console = Console()

GENERATIVE_EVAL_STATE_FILENAME = "generative_eval_state.yaml"
# Version 2 invalidates metrics produced with Qwen thinking implicitly enabled.
GENERATIVE_EVAL_STATE_VERSION = 2


# ---------------------------------------------------------------------------
# Generative Dev-Evaluation
# ---------------------------------------------------------------------------

def _run_generative_eval(
    model,
    tokenizer,
    prompts: List[List[Dict]],
    gold_entities: List[List[Dict[str, Any]]],
    tokens_list: List[List[str]],
    valid_types: frozenset,
    max_new_tokens: int = 256,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Fuehrt generative Evaluation auf einem Datensatz-Split durch.

    Statt teacher-forced eval_loss wird hier tatsaechlich generiert,
    die Ausgabe geparst und entity-level F1 berechnet.

    Args:
        model:          Das (PEFT-)Modell.
        tokenizer:      Der Tokenizer.
        prompts:        Liste von [system, user]-Nachrichtenlisten.
        gold_entities:  Gold-Entity-Dicts pro Satz.
        tokens_list:    Token-Listen pro Satz (fuer BIO-Konvertierung).
        valid_types:    Erlaubte Entity-Typen.
        max_new_tokens: Maximale Laenge der generierten Antwort.
        max_samples:    Maximale Anzahl Samples (None = alle).

    Returns:
        Dict mit precision, recall, f1, parse_failure_rate etc.
    """
    model.eval()
    device = next(model.parameters()).device

    n = min(max_samples or len(prompts), len(prompts))

    pred_entities: List[List[Dict]] = []
    parse_statuses: List[str] = []
    parse_diagnostics: List[Dict[str, int]] = []

    for i in range(n):
        model_inputs = prepare_generation_inputs(tokenizer, prompts[i], device)

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

        prompt_len = model_inputs["input_ids"].shape[1]
        new_token_ids = output_ids[0][prompt_len:]
        generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)

        entities, status, diagnostics = parse_llm_output_with_diagnostics(
            generated_text,
            tokens_list[i],
            valid_types,
        )
        pred_entities.append(entities)
        parse_statuses.append(status)
        parse_diagnostics.append(diagnostics)

    metrics = evaluate_llm_predictions(
        tokens_list=tokens_list[:n],
        gold_entities=gold_entities[:n],
        pred_entities=pred_entities,
        parse_statuses=parse_statuses,
        parse_diagnostics=parse_diagnostics,
    )

    model.train()
    return metrics


class GenerativeDevEvalCallback(TrainerCallback):
    """Callback: Generative Dev-Evaluation nach jeder Epoche.

    Statt sich auf den teacher-forced eval_loss zu verlassen, wird
    nach jeder Evaluation tatsaechlich generiert und der Dev-F1
    aus den erzeugten Entities berechnet.

    Der beste LoRA-Adapter wird separat in best_lora_adapter/ gespeichert.
    """

    def __init__(
        self,
        tokenizer,
        dev_prompts: List[List[Dict]],
        dev_gold_entities: List[List[Dict[str, Any]]],
        dev_tokens: List[List[str]],
        valid_types: frozenset,
        output_dir: Path,
        max_new_tokens: int = 256,
        max_eval_samples: Optional[int] = 200,
    ):
        self.tokenizer = tokenizer
        self.dev_prompts = dev_prompts
        self.dev_gold_entities = dev_gold_entities
        self.dev_tokens = dev_tokens
        self.valid_types = valid_types
        self.output_dir = Path(output_dir)
        self.max_new_tokens = max_new_tokens
        self.max_eval_samples = max_eval_samples
        self.best_f1 = -1.0
        self.best_epoch = -1
        self.epoch_results: List[Dict] = []
        self._trainer = None
        self._load_state()

    def set_trainer(self, trainer):
        """Muss nach Trainer-Erstellung aufgerufen werden."""
        self._trainer = trainer

    @property
    def state_file(self) -> Path:
        return self.output_dir / GENERATIVE_EVAL_STATE_FILENAME

    @staticmethod
    def _metric(value: Any, field_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} muss numerisch sein")
        metric = float(value)
        if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
            raise ValueError(f"{field_name} muss endlich und zwischen 0 und 1 sein")
        return metric

    @classmethod
    def _validated_state(cls, raw_state: Any) -> Dict[str, Any]:
        if not isinstance(raw_state, dict):
            raise ValueError("State muss ein Mapping sein")
        if raw_state.get("version") != GENERATIVE_EVAL_STATE_VERSION:
            raise ValueError("Unbekannte State-Version")

        raw_results = raw_state.get("epoch_results")
        if not isinstance(raw_results, list) or not raw_results:
            raise ValueError("epoch_results muss eine nicht-leere Liste sein")

        epoch_results: List[Dict[str, Any]] = []
        seen_epochs = set()
        for index, raw_result in enumerate(raw_results):
            if not isinstance(raw_result, dict):
                raise ValueError(f"epoch_results[{index}] muss ein Mapping sein")
            epoch = raw_result.get("epoch")
            if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
                raise ValueError(f"epoch_results[{index}].epoch ist ungueltig")
            if epoch in seen_epochs:
                raise ValueError(f"Epoche {epoch} ist doppelt vorhanden")
            seen_epochs.add(epoch)
            epoch_results.append({
                "epoch": epoch,
                "dev_f1": cls._metric(
                    raw_result.get("dev_f1"),
                    f"epoch_results[{index}].dev_f1",
                ),
                "dev_precision": cls._metric(
                    raw_result.get("dev_precision"),
                    f"epoch_results[{index}].dev_precision",
                ),
                "dev_recall": cls._metric(
                    raw_result.get("dev_recall"),
                    f"epoch_results[{index}].dev_recall",
                ),
                "parse_failure_rate": cls._metric(
                    raw_result.get("parse_failure_rate"),
                    f"epoch_results[{index}].parse_failure_rate",
                ),
            })

        best_f1 = cls._metric(raw_state.get("best_f1"), "best_f1")
        best_epoch = raw_state.get("best_epoch")
        if (
            isinstance(best_epoch, bool)
            or not isinstance(best_epoch, int)
            or best_epoch not in seen_epochs
        ):
            raise ValueError("best_epoch ist ungueltig")

        matching_best = any(
            result["epoch"] == best_epoch
            and math.isclose(
                result["dev_f1"],
                best_f1,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for result in epoch_results
        )
        max_f1 = max(result["dev_f1"] for result in epoch_results)
        if not matching_best or not math.isclose(
            best_f1,
            max_f1,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Best-Werte passen nicht zu epoch_results")

        return {
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "epoch_results": epoch_results,
        }

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            with self.state_file.open("r", encoding="utf-8") as handle:
                state = self._validated_state(yaml.safe_load(handle))
        except (
            OSError,
            UnicodeError,
            TypeError,
            ValueError,
            yaml.YAMLError,
        ) as exc:
            console.print(
                f"[yellow]Warnung: Generative-Eval-State in {self.state_file} "
                f"ist ungueltig und wird ignoriert ({exc}).[/yellow]"
            )
            return

        self.best_f1 = state["best_f1"]
        self.best_epoch = state["best_epoch"]
        self.epoch_results = state["epoch_results"]
        console.print(
            f"[cyan]Generative-Eval-State geladen: Best-F1 "
            f"{self.best_f1:.4f} (Epoche {self.best_epoch})[/cyan]"
        )

    def _save_state(self) -> None:
        state = {
            "version": GENERATIVE_EVAL_STATE_VERSION,
            "best_f1": float(self.best_f1),
            "best_epoch": int(self.best_epoch),
            "epoch_results": self.epoch_results,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)

        temp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.output_dir,
                prefix=f".{self.state_file.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                yaml.safe_dump(state, handle, default_flow_style=False, sort_keys=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.state_file)
        except Exception:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise

    def on_evaluate(self, args, state, control, **kwargs):
        """Wird nach jeder Evaluation (= nach jeder Epoche) aufgerufen."""
        if self._trainer is None:
            return

        model = self._trainer.model
        epoch = int(state.epoch) if state.epoch else 0
        if any(result["epoch"] == epoch for result in self.epoch_results):
            console.print(
                f"[cyan]Generative Dev-Evaluation fuer Epoche {epoch} bereits "
                f"gespeichert; ueberspringe Wiederholung.[/cyan]"
            )
            return

        console.print(f"\n[cyan]Generative Dev-Evaluation (Epoche {epoch})...[/cyan]")
        console.print(f"  Evaluiere {self.max_eval_samples or len(self.dev_prompts)} Samples...")

        eval_start = time.perf_counter()
        metrics = _run_generative_eval(
            model=model,
            tokenizer=self.tokenizer,
            prompts=self.dev_prompts,
            gold_entities=self.dev_gold_entities,
            tokens_list=self.dev_tokens,
            valid_types=self.valid_types,
            max_new_tokens=self.max_new_tokens,
            max_samples=self.max_eval_samples,
        )
        eval_time = time.perf_counter() - eval_start

        f1 = metrics["f1"]
        console.print(
            f"  Dev-F1 (generativ): {f1:.4f} "
            f"(P={metrics['precision']:.4f}, R={metrics['recall']:.4f}) "
            f"in {eval_time:.1f}s"
        )

        self.epoch_results.append({
            "epoch":         epoch,
            "dev_f1":        f1,
            "dev_precision": metrics["precision"],
            "dev_recall":    metrics["recall"],
            "parse_failure_rate": metrics["parse_failure_rate"],
        })

        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_epoch = epoch
            best_dir = self.output_dir / "best_lora_adapter"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_dir))
            self.tokenizer.save_pretrained(str(best_dir))
            console.print(f"  [green]Neuer Best-F1! Adapter gespeichert: {best_dir}[/green]")
        else:
            console.print(f"  [dim]Kein Improvement (Best: {self.best_f1:.4f} bei Epoche {self.best_epoch})[/dim]")

        self._save_state()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_decoder(config_path: str) -> Dict[str, Any]:
    """Fine-tuned ein kausales LM fuer NER via LoRA (oder QLoRA) mit SFTTrainer.

    Ablauf:
      1. Config laden, Seeds setzen
      2. Tokenizer laden; pad_token auf eos_token setzen
      3. Basismodell laden (bf16 oder 4-bit quantisiert)
      4. LoRA-Config bauen und Adapter ans Modell anhaengen
      5. Datensatz im ChatML-Format laden
      6. Generative-Eval-Callback mit Dev-Daten vorbereiten
      7. SFTTrainer konfigurieren und Training starten
      8. Besten LoRA-Adapter und Ergebnisse speichern

    Args:
        config_path: Pfad zur YAML-Konfigurationsdatei.

    Returns:
        Dict mit train_runtime_seconds, trainable_params, best_dev_f1 etc.
    """
    config_path = Path(config_path)
    cfg = load_experiment_config(config_path, expected_model_type="decoder", expected_regime="llm_lora")

    console.rule(f"[bold green]Decoder-Training: {cfg['experiment_name']} on MultiNERD English[/bold green]")

    from peft import LoraConfig, TaskType, get_peft_model
    from trl import SFTConfig, SFTTrainer

    # --- Reproduzierbarkeit ---
    seed: int = cfg.get("seed", 42)
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Ausgabeverzeichnisse ---
    output_dir = output_dir_from_config(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name:     str  = cfg["model_name"]
    use_qlora:      bool = cfg.get("use_qlora", True)
    attn_impl:      str  = cfg.get("attn_impl", "sdpa")
    max_seq_length: int  = cfg.get("max_seq_length", 1024)

    # --- Tokenizer laden ---
    console.print(f"[cyan]Lade Tokenizer: {model_name}[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Basismodell laden ---
    if use_qlora:
        console.print("[cyan]Lade Modell im 4-bit QLoRA-Modus...[/cyan]")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            attn_implementation=attn_impl,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        console.print(f"[cyan]Lade Modell in bf16 (attn={attn_impl})...[/cyan]")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map="auto",
            trust_remote_code=True,
        )

    model.config.pad_token_id = tokenizer.pad_token_id

    # --- LoRA-Konfiguration ---
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

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Gesamt: {total_params:,} | Trainierbar: {trainable_params:,}")

    # --- Datensatz im ChatML-Format laden (fuer SFT-Training) ---
    console.print("[cyan]Bereite Decoder-Datensatz vor...[/cyan]")
    dataset, info = prepare_decoder_dataset()

    # --- Rohen Dev-Split fuer generative Evaluation laden ---
    console.print("[cyan]Bereite Dev-Daten fuer generative Evaluation vor...[/cyan]")
    raw_dataset, _ = load_ner_dataset()
    raw_dev = raw_dataset["validation"]

    system_prompt = build_system_prompt(info.entity_types)
    valid_types = frozenset(info.entity_types)

    # Dev-Prompts und Gold-Entities fuer generative Eval vorbereiten
    dev_prompts: List[List[Dict]] = []
    dev_gold_entities: List[List[Dict[str, Any]]] = []
    dev_tokens: List[List[str]] = []

    for sample in raw_dev:
        tokens = sample["tokens"]
        ner_tags = sample["ner_tags"]
        dev_prompts.append([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": format_numbered_tokens(tokens)},
        ])
        dev_gold_entities.append(
            extract_entities_from_bio(tokens, ner_tags, info.id2label)
        )
        dev_tokens.append(tokens)

    # --- Generative-Eval-Callback erstellen ---
    gen_eval_max_samples = cfg.get("gen_eval_max_samples", 200)
    gen_eval_callback = GenerativeDevEvalCallback(
        tokenizer=tokenizer,
        dev_prompts=dev_prompts,
        dev_gold_entities=dev_gold_entities,
        dev_tokens=dev_tokens,
        valid_types=valid_types,
        output_dir=output_dir,
        max_new_tokens=int(cfg.get("max_new_tokens", 256)),
        max_eval_samples=gen_eval_max_samples,
    )

    # --- SFT-Konfiguration ---
    sft_config_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=cfg.get("num_train_epochs", cfg.get("epochs", 3)),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", cfg.get("batch_size", 4)),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", cfg.get("batch_size", 4)),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", cfg.get("grad_accum", 4)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.05)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        bf16=True,
        fp16=False,
        logging_steps=cfg.get("logging_steps", 25),
        eval_strategy=cfg.get("eval_strategy", "epoch"),
        save_strategy=cfg.get("save_strategy", "epoch"),
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 2),
        load_best_model_at_end=False,  # Best-Model wird ueber generative Eval gewaehlt
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=cfg.get("packing", False),
        seed=seed,
        report_to="wandb" if cfg.get("use_wandb", False) else "none",
        run_name=f"{cfg.get('experiment_name')}_{DATASET_NAME}",
        dataset_text_field=None,
    )
    length_arg = (
        "max_length"
        if "max_length" in inspect.signature(SFTConfig).parameters
        else "max_seq_length"
    )
    sft_config_kwargs[length_arg] = max_seq_length
    sft_config = SFTConfig(**sft_config_kwargs)

    # --- SFTTrainer zusammenbauen ---
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        callbacks=[gen_eval_callback],
    )

    # Trainer-Referenz fuer den Callback setzen
    gen_eval_callback.set_trainer(trainer)

    # --- Training starten ---
    console.print("[bold yellow]Starte LoRA-Finetuning...[/bold yellow]")
    last_checkpoint = get_last_checkpoint(str(output_dir))
    if last_checkpoint:
        console.print(
            f"[yellow]Setze Training vom letzten Checkpoint fort: "
            f"{last_checkpoint}[/yellow]"
        )
    train_start = time.perf_counter()
    trainer.train(resume_from_checkpoint=last_checkpoint)
    train_runtime = time.perf_counter() - train_start

    # --- Letzten Adapter speichern (als Fallback) ---
    last_adapter_dir = output_dir / "lora_adapter"
    console.print(f"[green]Speichere letzten LoRA-Adapter nach {last_adapter_dir}[/green]")
    model.save_pretrained(str(last_adapter_dir))
    tokenizer.save_pretrained(str(last_adapter_dir))

    # --- Ergebnisse zusammenfassen ---
    best_adapter_dir = output_dir / "best_lora_adapter"
    if not best_adapter_dir.exists():
        # Fallback: wenn die generative Eval keinen Best gefunden hat
        console.print("[yellow]Kein best_lora_adapter gefunden, verwende letzten Adapter.[/yellow]")
        best_adapter_dir = last_adapter_dir

    results: Dict[str, Any] = {
        "experiment_name":       cfg["experiment_name"],
        "model_name":            model_name,
        "model_type":            "decoder",
        "regime":                "llm_lora",  # LoRA/QLoRA Regime
        "dataset":               DATASET_NAME,
        "dataset_language":      DATASET_LANGUAGE,
        "use_qlora":             use_qlora,
        "lora_r":                cfg.get("lora_r", 16),
        "lora_alpha":            cfg.get("lora_alpha", 32),
        "train_runtime_seconds": float(train_runtime),
        "trainable_params":      trainable_params,
        "total_params":          total_params,
        "best_dev_f1":           gen_eval_callback.best_f1,
        "best_epoch":            gen_eval_callback.best_epoch,
        "epoch_results":         gen_eval_callback.epoch_results,
        "best_adapter_dir":      str(best_adapter_dir),
        "last_adapter_dir":      str(last_adapter_dir),
        "thinking_enabled":      THINKING_ENABLED,
        "seed":                  seed,
        "run_metadata":          collect_run_metadata(cfg),
    }

    results_file = output_dir / "results.yaml"
    with open(results_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    console.print(f"Ergebnisse gespeichert: {results_file}")

    console.print(f"\n[bold green]Training abgeschlossen in {train_runtime:.1f}s[/bold green]")
    console.print(f"Bester Dev-F1: {gen_eval_callback.best_f1:.4f} (Epoche {gen_eval_callback.best_epoch})")

    return results


# ---------------------------------------------------------------------------
# CLI-Einstiegspunkt
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decoder-NER Training mit LoRA/QLoRA")
    parser.add_argument("config", help="Pfad zur YAML-Config")
    args = parser.parse_args()
    train_decoder(args.config)
