#!/usr/bin/env bash
#SBATCH --job-name=ctrl
#SBATCH --partition=H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# MultiNERD control sample: one canonical Qwen QLoRA run per job.
# Usage: sbatch [--mem=64G] scripts/cluster/job_multinerd_control_sample.sh <qwen_config> [extra arguments]
# Extra arguments go to src.evaluate.multinerd_control_sample, e.g. --sample-size 20 or --resume-failed.
# Prefer scripts/cluster/submit_multinerd_control_sample.sh, which sets memory, job names, the frozen
# commit and the output root.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch $0 <qwen_config> [extra multinerd_control_sample arguments]" >&2
    exit 2
fi

CONFIG="$1"
shift
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/cluster"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_ba_ner_env
print_runtime_info
require_path "${CONFIG}"
: "${BA_NER_RESULTS_ROOT:?Set BA_NER_RESULTS_ROOT to the results root that holds multinerd/<experiment>}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
OUTPUT_ROOT="${BA_NER_CONTROL_ROOT:-${BA_NER_RESULTS_ROOT}/multinerd_control_1024}"
echo "Source results root: ${BA_NER_RESULTS_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "HF_HOME: ${HF_HOME}"

python -m src.evaluate.multinerd_control_sample \
    --config "${CONFIG}" \
    --source-results-root "${BA_NER_RESULTS_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    "$@"
