#!/bin/bash
#SBATCH --job-name=ner-decoder
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/decoder_%j.log
#SBATCH --error=logs/decoder_%j.err

set -euo pipefail

echo "=== NER Decoder Training (Qwen3.5 + LoRA/QLoRA) ==="
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
# Hauptbenchmark: MultiNERD (englisch)
# ==========================================================================

echo "--- Training Qwen3.5-4B QLoRA (MultiNERD) ---"
python -m src.decoder.train configs/qwen35_4b.yaml

echo "--- Training Qwen3.5-27B QLoRA (MultiNERD) ---"
python -m src.decoder.train configs/qwen35_27b.yaml

echo "--- Inference Qwen3.5-4B (MultiNERD) ---"
python -m src.decoder.inference \
    --adapter results/multinerd/qwen35-4b-qlora/best_lora_adapter \
    --base Qwen/Qwen3.5-4B \
    --config configs/qwen35_4b.yaml

echo "--- Inference Qwen3.5-27B (MultiNERD) ---"
python -m src.decoder.inference \
    --adapter results/multinerd/qwen35-27b-qlora/best_lora_adapter \
    --base Qwen/Qwen3.5-27B \
    --config configs/qwen35_27b.yaml

# ==========================================================================
# Zusatzbenchmark: WNUT-2017
# ==========================================================================

echo "--- Training Qwen3.5-4B QLoRA (WNUT-17) ---"
python -m src.decoder.train configs/qwen35_4b.yaml --dataset wnut_17

echo "--- Inference Qwen3.5-4B (WNUT-17) ---"
python -m src.decoder.inference \
    --adapter results/wnut_17/qwen35-4b-qlora/best_lora_adapter \
    --base Qwen/Qwen3.5-4B \
    --config configs/qwen35_4b.yaml \
    --dataset wnut_17

# Optional: Qwen3.5-27B auch auf WNUT-17 (deutlich teurer, nur wenn GPU-Budget reicht)
# echo "--- Training Qwen3.5-27B QLoRA (WNUT-17) ---"
# python -m src.decoder.train configs/qwen35_27b.yaml --dataset wnut_17
#
# echo "--- Inference Qwen3.5-27B (WNUT-17) ---"
# python -m src.decoder.inference \
#     --adapter results/wnut_17/qwen35-27b-qlora/best_lora_adapter \
#     --base Qwen/Qwen3.5-27B \
#     --config configs/qwen35_27b.yaml \
#     --dataset wnut_17

echo ""
echo "=== Decoder pipeline done: $(date) ==="
