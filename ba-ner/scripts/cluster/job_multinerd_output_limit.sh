#!/usr/bin/env bash
#SBATCH --job-name=mnol
#SBATCH --partition=H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# MultiNERD output-limit re-generation: one canonical Qwen QLoRA run per job.
# Usage: sbatch [--mem=64G] scripts/cluster/job_multinerd_output_limit.sh <qwen_config> [extra arguments]
# Extra arguments go to src.evaluate.multinerd_output_limit, e.g. --smoke-limit 2 --control-limit 5,
# --resume-failed or --max-new-tokens 2048. Prefer scripts/cluster/submit_multinerd_output_limit.sh,
# which sets memory, job names, the frozen commit and the output root.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch $0 <qwen_config> [extra multinerd_output_limit arguments]" >&2
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
OUTPUT_ROOT="${BA_NER_OUTPUT_LIMIT_ROOT:-${BA_NER_RESULTS_ROOT}/multinerd_output_limit_1024}"
echo "Source results root: ${BA_NER_RESULTS_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "HF_HOME: ${HF_HOME}"

python -m src.evaluate.multinerd_output_limit \
    --config "${CONFIG}" \
    --source-results-root "${BA_NER_RESULTS_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    "$@"
