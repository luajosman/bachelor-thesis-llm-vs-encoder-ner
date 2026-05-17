#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_DIR}"
mkdir -p logs results

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

activate_ba_ner_env() {
    local env_name="${BA_NER_CONDA_ENV:-ba-ner}"

    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "${env_name}"
        return
    fi

    if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
        eval "$("${CONDA_EXE}" shell.bash hook)"
        conda activate "${env_name}"
        return
    fi

    if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck source=/dev/null
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
        conda activate "${env_name}"
        return
    fi

    if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck source=/dev/null
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
        conda activate "${env_name}"
        return
    fi

    echo "Could not initialize conda. Set BA_NER_CONDA_ENV or load conda first." >&2
    exit 1
}

print_runtime_info() {
    echo "Project: ${PROJECT_DIR}"
    echo "Job ID: ${SLURM_JOB_ID:-local}"
    echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
    echo "Start: $(date)"
    echo "Python: $(command -v python)"
    python --version
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
    else
        echo "nvidia-smi not found"
    fi
}

require_path() {
    local path="$1"
    if [ ! -e "${path}" ]; then
        echo "Required path does not exist: ${path}" >&2
        exit 1
    fi
}
