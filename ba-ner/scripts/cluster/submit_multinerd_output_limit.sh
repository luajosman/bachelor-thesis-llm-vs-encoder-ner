#!/usr/bin/env bash
# Submit MultiNERD output-limit re-generation jobs for canonical Qwen QLoRA configs.
#
# Usage:
#   scripts/cluster/submit_multinerd_output_limit.sh [--smoke-limit N] [--control-limit K] \
#       [--resume-failed] [--max-new-tokens T] <config>...
#
# Requires a clean committed checkout and BA_NER_RESULTS_ROOT. Memory follows the model size
# (27B: 160G, 4B and 0.8B: 64G); job names are mnol-{27b|4b|08b}-s<seed>[-smoke]. Smoke runs
# write below <output root>/smoke/. BA_NER_OUTPUT_LIMIT_ROOT overrides the output root;
# MNOL_TIME (default 03:00:00), MNOL_PARTITION (H100) and MEM_QWEN_{27B,4B,08B} override resources.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

usage() {
    sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

SMOKE_LIMIT=""
CONTROL_LIMIT=""
RESUME_FAILED=0
MAX_NEW_TOKENS="${BA_NER_OUTPUT_LIMIT_TOKENS:-1024}"
CONFIGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --smoke-limit) SMOKE_LIMIT="${2:?--smoke-limit needs a value}"; shift 2 ;;
        --control-limit) CONTROL_LIMIT="${2:?--control-limit needs a value}"; shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="${2:?--max-new-tokens needs a value}"; shift 2 ;;
        --resume-failed) RESUME_FAILED=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) CONFIGS+=("$1"); shift ;;
    esac
done
if [ "${#CONFIGS[@]}" -eq 0 ]; then
    usage >&2
    exit 2
fi
if [ -n "${CONTROL_LIMIT}" ] && [ -z "${SMOKE_LIMIT}" ]; then
    echo "--control-limit requires --smoke-limit" >&2
    exit 2
fi
case "${MAX_NEW_TOKENS}" in
    ''|*[!0-9]*) echo "--max-new-tokens must be an integer" >&2; exit 2 ;;
esac
if [ "${MAX_NEW_TOKENS}" -le 256 ]; then
    echo "--max-new-tokens must be greater than 256" >&2
    exit 2
fi

: "${BA_NER_RESULTS_ROOT:?Set BA_NER_RESULTS_ROOT, e.g. /netscratch/losman/ba-ner/results}"
if [ -n "$(git status --porcelain=v1 --untracked-files=all)" ]; then
    echo "Commit the tested implementation and keep the repository clean before submission" >&2
    exit 1
fi
COMMIT="$(git rev-parse HEAD)"
OUTPUT_ROOT="${BA_NER_OUTPUT_LIMIT_ROOT:-${BA_NER_RESULTS_ROOT}/multinerd_output_limit_${MAX_NEW_TOKENS}}"
if [ -n "${SMOKE_LIMIT}" ]; then
    OUTPUT_ROOT="${OUTPUT_ROOT}/smoke"
fi
for value in "${BA_NER_RESULTS_ROOT}" "${OUTPUT_ROOT}" "${HF_HOME:-}"; do
    case "${value}" in
        *,*) echo "SLURM export values cannot contain commas: ${value}" >&2; exit 1 ;;
    esac
done
mkdir -p logs

EXPORTS="ALL,BA_NER_EXPECTED_GIT_COMMIT=${COMMIT},BA_NER_RESULTS_ROOT=${BA_NER_RESULTS_ROOT}"
EXPORTS="${EXPORTS},BA_NER_OUTPUT_LIMIT_ROOT=${OUTPUT_ROOT},HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"
EXPORTS="${EXPORTS},HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1},PYTHONUNBUFFERED=1"
if [ -n "${HF_HOME:-}" ]; then
    EXPORTS="${EXPORTS},HF_HOME=${HF_HOME}"
fi

echo "Commit: ${COMMIT}"
echo "Source results root: ${BA_NER_RESULTS_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"

for CONFIG in "${CONFIGS[@]}"; do
    if [ ! -f "${CONFIG}" ]; then
        echo "Config not found: ${CONFIG}" >&2
        exit 1
    fi
    SEED="$(sed -n 's/^seed:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "${CONFIG}" | head -n 1)"
    EXPERIMENT="$(sed -n 's/^experiment_name:[[:space:]]*\([^[:space:]#]*\).*/\1/p' "${CONFIG}" | head -n 1)"
    if [ -z "${SEED}" ] || [ -z "${EXPERIMENT}" ]; then
        echo "Cannot read seed/experiment_name from ${CONFIG}" >&2
        exit 1
    fi
    case "$(basename "${CONFIG}")" in
        qwen35_27b_3ep*) SIZE=27b; MEM="${MEM_QWEN_27B:-160G}" ;;
        qwen35_4b_canonical*|qwen35_4b_seed*) SIZE=4b; MEM="${MEM_QWEN_4B:-64G}" ;;
        qwen35_08b_canonical*|qwen35_08b_seed*) SIZE=08b; MEM="${MEM_QWEN_08B:-64G}" ;;
        *) echo "Not a canonical Qwen QLoRA config: ${CONFIG}" >&2; exit 1 ;;
    esac
    NAME="mnol-${SIZE}-s${SEED}"
    EXTRA=()
    if [ -n "${SMOKE_LIMIT}" ]; then
        NAME="${NAME}-smoke"
        EXTRA+=(--smoke-limit "${SMOKE_LIMIT}")
    fi
    if [ -n "${CONTROL_LIMIT}" ]; then
        EXTRA+=(--control-limit "${CONTROL_LIMIT}")
    fi
    if [ "${MAX_NEW_TOKENS}" != "1024" ]; then
        EXTRA+=(--max-new-tokens "${MAX_NEW_TOKENS}")
    fi
    if [ "${RESUME_FAILED}" -eq 1 ]; then
        EXTRA+=(--resume-failed)
    fi
    DIRECTORY="${OUTPUT_ROOT}/${EXPERIMENT}"
    if [ -e "${DIRECTORY}/.multinerd_output_limit.lock" ] || { [ -f "${DIRECTORY}/status.json" ] \
            && grep -Eq '"status":[[:space:]]*"(RUNNING|FAILED)"' "${DIRECTORY}/status.json"; }; then
        if [ "${RESUME_FAILED}" -ne 1 ]; then
            echo "${EXPERIMENT}: lock or RUNNING/FAILED status in ${DIRECTORY}; confirm that the job is dead (squeue), then resubmit with --resume-failed" >&2
            exit 1
        fi
    fi
    if [ -f "${DIRECTORY}/inference_metrics.yaml" ]; then
        echo "${EXPERIMENT}: final metrics already exist; the job verifies the contract and skips if it is identical" >&2
    fi
    JOB_ID="$(sbatch --parsable --requeue "--job-name=${NAME}" "--mem=${MEM}" "--time=${MNOL_TIME:-03:00:00}" \
        "--partition=${MNOL_PARTITION:-H100}" "--export=${EXPORTS}" \
        scripts/cluster/job_multinerd_output_limit.sh "${CONFIG}" ${EXTRA[@]+"${EXTRA[@]}"})"
    JOB_ID="${JOB_ID%%;*}"
    echo "SUBMITTED ${EXPERIMENT} job=${JOB_ID} name=${NAME} mem=${MEM} args=${EXTRA[*]:-}"
done
