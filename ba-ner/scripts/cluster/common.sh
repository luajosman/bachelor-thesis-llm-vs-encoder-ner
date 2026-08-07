#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_DIR}"

verify_frozen_repository() {
    local expected_commit="${BA_NER_EXPECTED_GIT_COMMIT:-}"
    if [ -z "${expected_commit}" ]; then
        return
    fi
    if ! command -v git >/dev/null 2>&1; then
        echo "Git is required to verify the frozen training snapshot." >&2
        exit 1
    fi

    local actual_commit
    actual_commit="$(git rev-parse HEAD)"
    if [ "${actual_commit}" != "${expected_commit}" ]; then
        echo "Frozen repository mismatch: expected ${expected_commit}, got ${actual_commit}." >&2
        exit 1
    fi

    local worktree_status
    worktree_status="$(git status --porcelain=v1 --untracked-files=all)"
    if [ -n "${worktree_status}" ]; then
        echo "Frozen repository has uncommitted changes; refusing to run:" >&2
        echo "${worktree_status}" >&2
        exit 1
    fi
}

verify_frozen_repository
mkdir -p logs results

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [ -n "${BA_NER_SCRATCH:-}" ]; then
    mkdir -p "${BA_NER_SCRATCH}/hf-cache" "${BA_NER_SCRATCH}/results" "${BA_NER_SCRATCH}/tmp"
    export HF_HOME="${HF_HOME:-${BA_NER_SCRATCH}/hf-cache}"
    export BA_NER_RESULTS_ROOT="${BA_NER_RESULTS_ROOT:-${BA_NER_SCRATCH}/results}"
    export TMPDIR="${TMPDIR:-${BA_NER_SCRATCH}/tmp}"
    export TEMP="${TEMP:-${TMPDIR}}"
    export TMP="${TMP:-${TMPDIR}}"
else
    export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf_cache}"
fi
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
# Xet's parallel reconstruction is unreliable on the cluster's shared filesystem.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-3600}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${BA_NER_RESULTS_ROOT:-results}"

activate_ba_ner_env() {
    local env_name="${BA_NER_CONDA_ENV:-ba-ner}"
    local venv_path="${BA_NER_VENV:-${PROJECT_DIR}/.venv}"

    if [ -x "${venv_path}/bin/python" ]; then
        # shellcheck source=/dev/null
        source "${venv_path}/bin/activate"
        return
    fi

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

    echo "Could not activate ${venv_path} or initialize conda." >&2
    echo "Set BA_NER_VENV, set BA_NER_CONDA_ENV, or load conda first." >&2
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
