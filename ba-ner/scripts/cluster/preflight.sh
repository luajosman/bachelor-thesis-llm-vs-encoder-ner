#!/usr/bin/env bash
#SBATCH --job-name=ner-preflight
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_ba_ner_env
print_runtime_info

echo "=== Package availability ==="
python - <<'PY'
import importlib.util
import sys

required = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "trl",
    "bitsandbytes",
    "seqeval",
    "yaml",
    "rich",
    "numpy",
]

missing = []
for name in required:
    ok = importlib.util.find_spec(name) is not None
    print(f"{name}: {ok}")
    if not ok:
        missing.append(name)

if missing:
    print(f"Missing packages: {', '.join(missing)}", file=sys.stderr)
    sys.exit(1)
PY

echo "=== Config validation ==="
python - <<'PY'
from src.config import FINAL_EXPERIMENTS, load_experiment_config

for spec in FINAL_EXPERIMENTS.values():
    load_experiment_config(spec.config_path)
    print(f"ok: {spec.key} -> {spec.config_path}")
PY

echo "=== CUDA visibility ==="
python - <<'PY'
import os
import sys
import torch

require_cuda = os.environ.get("REQUIRE_CUDA", "0") == "1"
print(f"torch: {torch.__version__}")
print(f"cuda_available: {torch.cuda.is_available()}")
print(f"cuda_device_count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"cuda_device_{i}: {torch.cuda.get_device_name(i)}")
elif require_cuda:
    print("CUDA is required but not visible.", file=sys.stderr)
    sys.exit(1)
PY

echo "=== CLI help checks ==="
python scripts/run_all.py --help >/dev/null
python -m src.encoder.train --help >/dev/null
python -m src.encoder.inference --help >/dev/null
python -m src.decoder.train --help >/dev/null
python -m src.decoder.inference --help >/dev/null
python -m src.evaluate.compare_all --help >/dev/null
python -m src.evaluate.error_analysis --help >/dev/null

echo "=== Unit tests ==="
python -m pytest

echo "Preflight completed successfully."
