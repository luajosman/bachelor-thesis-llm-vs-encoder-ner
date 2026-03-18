#!/bin/bash
#SBATCH --job-name=ner-decoder
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/decoder_%j.log
#SBATCH --error=logs/decoder_%j.err

set -euo pipefail

echo "=== NER Decoder Training (LLM + LoRA) ==="
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

# ---- Training ----
echo "--- Training Qwen3.5-27B with bf16 LoRA (sdpa) ---"
python -m src.decoder.train configs/qwen35_27b.yaml

echo "--- Training Qwen3-14B with QLoRA (flash_attention_2) ---"
python -m src.decoder.train configs/qwen3_14b.yaml

# ---- Inference ----
echo "--- Inference Qwen3.5-27B ---"
python -m src.decoder.inference \
    --adapter results/qwen35-27b-lora/lora_adapter \
    --base Qwen/Qwen3.5-27B \
    --config configs/qwen35_27b.yaml

echo "--- Inference Qwen3-14B ---"
python -m src.decoder.inference \
    --adapter results/qwen3-14b-qlora/lora_adapter \
    --base Qwen/Qwen3-14B \
    --config configs/qwen3_14b.yaml

echo ""
echo "=== Decoder pipeline done: $(date) ==="
