#!/usr/bin/env bash
# Submit MultiNERD control-sample jobs for canonical Qwen QLoRA configs.
#
# Usage:
#   scripts/cluster/submit_multinerd_control_sample.sh [--smoke N] [--resume-failed] [--after JOBID] <config>...
#
# Requires a clean committed checkout and BA_NER_RESULTS_ROOT. Memory follows the model size (27B: 160G,
# 4B and 0.8B: 64G); job names are ctrl-{27b|4b|08b}-s<seed>[-smoke]. --smoke N samples N sentences and
# writes below <output root>/smoke/; --after makes every job wait for JOBID to finish successfully.
# BA_NER_CONTROL_ROOT overrides the output root; CTRL_PARTITION (H100-SLT,H100), CTRL_TIME (03:00:00)
# and CTRL_TIME_27B (04:00:00) override resources.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

usage() {
    sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

SMOKE=""
RESUME_FAILED=0
AFTER=""
CONFIGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --smoke) SMOKE="${2:?--smoke needs a sample size}"; shift 2 ;;
        --after) AFTER="${2:?--after needs a job id}"; shift 2 ;;
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

: "${BA_NER_RESULTS_ROOT:?Set BA_NER_RESULTS_ROOT, e.g. /netscratch/losman/ba-ner/results}"
if [ -n "$(git status --porcelain=v1 --untracked-files=all)" ]; then
    echo "Commit the tested implementation and keep the repository clean before submission" >&2
    exit 1
fi
COMMIT="$(git rev-parse HEAD)"
OUTPUT_ROOT="${BA_NER_CONTROL_ROOT:-${BA_NER_RESULTS_ROOT}/multinerd_control_1024}"
if [ -n "${SMOKE}" ]; then
    OUTPUT_ROOT="${OUTPUT_ROOT}/smoke"
fi
for value in "${BA_NER_RESULTS_ROOT}" "${OUTPUT_ROOT}" "${HF_HOME:-}"; do
    case "${value}" in
        *,*) echo "SLURM export values cannot contain commas: ${value}" >&2; exit 1 ;;
    esac
done
mkdir -p logs

EXPORTS="ALL,BA_NER_EXPECTED_GIT_COMMIT=${COMMIT},BA_NER_RESULTS_ROOT=${BA_NER_RESULTS_ROOT}"
EXPORTS="${EXPORTS},BA_NER_CONTROL_ROOT=${OUTPUT_ROOT},HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"
EXPORTS="${EXPORTS},HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1},PYTHONUNBUFFERED=1"
if [ -n "${HF_HOME:-}" ]; then
    EXPORTS="${EXPORTS},HF_HOME=${HF_HOME}"
fi
DEPENDENCY=()
if [ -n "${AFTER}" ]; then
    DEPENDENCY=(--dependency "afterok:${AFTER}")
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
    if [ -z "${SEED}" ]; then
        echo "Cannot read seed from ${CONFIG}" >&2
        exit 1
    fi
    case "$(basename "${CONFIG}")" in
        qwen35_27b_3ep*) SIZE=27b; MEM=160G; TIME="${CTRL_TIME_27B:-04:00:00}" ;;
        qwen35_4b_canonical*|qwen35_4b_seed*) SIZE=4b; MEM=64G; TIME="${CTRL_TIME:-03:00:00}" ;;
        qwen35_08b_canonical*|qwen35_08b_seed*) SIZE=08b; MEM=64G; TIME="${CTRL_TIME:-03:00:00}" ;;
        *) echo "Not a canonical Qwen QLoRA config: ${CONFIG}" >&2; exit 1 ;;
    esac
    NAME="ctrl-${SIZE}-s${SEED}"
    EXTRA=()
    if [ -n "${SMOKE}" ]; then
        NAME="${NAME}-smoke"
        EXTRA+=(--sample-size "${SMOKE}")
    fi
    if [ "${RESUME_FAILED}" -eq 1 ]; then
        EXTRA+=(--resume-failed)
    fi
    JOB_ID="$(sbatch --parsable --requeue "--job-name=${NAME}" "--mem=${MEM}" "--time=${TIME}" \
        "--partition=${CTRL_PARTITION:-H100-SLT,H100}" ${DEPENDENCY[@]+"${DEPENDENCY[@]}"} "--export=${EXPORTS}" \
        scripts/cluster/job_multinerd_control_sample.sh "${CONFIG}" ${EXTRA[@]+"${EXTRA[@]}"})"
    JOB_ID="${JOB_ID%%;*}"
    echo "SUBMITTED ${CONFIG} job=${JOB_ID} name=${NAME} mem=${MEM} time=${TIME} args=${EXTRA[*]:-}"
done
