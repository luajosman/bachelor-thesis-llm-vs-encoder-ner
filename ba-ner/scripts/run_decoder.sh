#!/bin/bash
#SBATCH --job-name=ner-decoder
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/decoder_%j.log
#SBATCH --error=logs/decoder_%j.err

set -euo pipefail

echo "=== NER LLM Pipeline (Qwen3.5: Zero-Shot + LoRA/QLoRA) ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   ${SLURMD_NODENAME:-localhost}"
echo "Start:  $(date)"
echo ""

# GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Activate conda environment
source activate ba-ner

# Move to submission directory (SLURM) or script directory (local)
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

# Create log directory
mkdir -p logs

# ==========================================================================
# MultiNERD English — der einzige aktive Datensatz
# Sechs LLM-Experimente: 3 Modellgroessen x {Zero-Shot, LoRA/QLoRA}
# ==========================================================================

# --------------------------------------------------------------------------
# Zero-Shot (kein Training, nur Inferenz)
# --------------------------------------------------------------------------

echo "--- Zero-Shot Qwen3.5-0.8B (MultiNERD) ---"
python -m src.decoder.inference \
    --zeroshot \
    --base Qwen/Qwen3.5-0.8B \
    --config configs/qwen35_08b_zeroshot.yaml

echo "--- Zero-Shot Qwen3.5-4B (MultiNERD) ---"
python -m src.decoder.inference \
    --zeroshot \
    --base Qwen/Qwen3.5-4B \
    --config configs/qwen35_4b_zeroshot.yaml

echo "--- Zero-Shot Qwen3.5-27B (MultiNERD) ---"
python -m src.decoder.inference \
    --zeroshot \
    --base Qwen/Qwen3.5-27B \
    --config configs/qwen35_27b_zeroshot.yaml

# --------------------------------------------------------------------------
# LoRA/QLoRA Fine-Tuning + anschliessende Inferenz
# --------------------------------------------------------------------------

echo "--- Training Qwen3.5-0.8B QLoRA (MultiNERD) ---"
python -m src.decoder.train configs/qwen35_08b.yaml

echo "--- Inference Qwen3.5-0.8B LoRA (MultiNERD) ---"
python -m src.decoder.inference \
    --adapter results/multinerd/qwen35-08b-qlora/best_lora_adapter \
    --base Qwen/Qwen3.5-0.8B \
    --config configs/qwen35_08b.yaml

echo "--- Training Qwen3.5-4B QLoRA (MultiNERD) ---"
python -m src.decoder.train configs/qwen35_4b.yaml

echo "--- Inference Qwen3.5-4B LoRA (MultiNERD) ---"
python -m src.decoder.inference \
    --adapter results/multinerd/qwen35-4b-qlora/best_lora_adapter \
    --base Qwen/Qwen3.5-4B \
    --config configs/qwen35_4b.yaml

echo "--- Training Qwen3.5-27B QLoRA (MultiNERD) ---"
python -m src.decoder.train configs/qwen35_27b.yaml

echo "--- Inference Qwen3.5-27B LoRA (MultiNERD) ---"
python -m src.decoder.inference \
    --adapter results/multinerd/qwen35-27b-qlora/best_lora_adapter \
    --base Qwen/Qwen3.5-27B \
    --config configs/qwen35_27b.yaml

echo ""
echo "=== LLM pipeline done: $(date) ==="
