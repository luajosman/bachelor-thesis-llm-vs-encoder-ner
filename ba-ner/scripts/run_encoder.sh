#!/bin/bash
#SBATCH --job-name=ner-encoder
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/encoder_%j.log
#SBATCH --error=logs/encoder_%j.err

set -euo pipefail

echo "=== NER Encoder Training (DeBERTa-v3) ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   ${SLURMD_NODENAME:-localhost}"
echo "Start:  $(date)"
echo ""

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

echo "--- Training DeBERTa-v3-base (MultiNERD) ---"
python -m src.encoder.train configs/deberta_base.yaml

echo "--- Training DeBERTa-v3-large (MultiNERD) ---"
python -m src.encoder.train configs/deberta_large.yaml

echo "--- Inference DeBERTa-v3-base (MultiNERD) ---"
python -m src.encoder.inference \
    --model results/multinerd/deberta-v3-base/best_model \
    --config configs/deberta_base.yaml

echo "--- Inference DeBERTa-v3-large (MultiNERD) ---"
python -m src.encoder.inference \
    --model results/multinerd/deberta-v3-large/best_model \
    --config configs/deberta_large.yaml

# ==========================================================================
# Zusatzbenchmark: WNUT-2017
# ==========================================================================

echo "--- Training DeBERTa-v3-base (WNUT-17) ---"
python -m src.encoder.train configs/deberta_base.yaml --dataset wnut_17

echo "--- Inference DeBERTa-v3-base (WNUT-17) ---"
python -m src.encoder.inference \
    --model results/wnut_17/deberta-v3-base/best_model \
    --config configs/deberta_base.yaml \
    --dataset wnut_17

# Optional: DeBERTa-v3-large auch auf WNUT-17 (wenn Zeit reicht)
# echo "--- Training DeBERTa-v3-large (WNUT-17) ---"
# python -m src.encoder.train configs/deberta_large.yaml --dataset wnut_17
#
# echo "--- Inference DeBERTa-v3-large (WNUT-17) ---"
# python -m src.encoder.inference \
#     --model results/wnut_17/deberta-v3-large/best_model \
#     --config configs/deberta_large.yaml \
#     --dataset wnut_17

echo ""
echo "=== Encoder pipeline done: $(date) ==="
