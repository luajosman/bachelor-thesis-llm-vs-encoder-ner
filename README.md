# Bachelor Thesis NER Experiments

This repository compares encoder-based token classification against decoder-only
LLM-based NER on one benchmark:

- Dataset: `Babelscape/multinerd`
- Subset: English only, selected by `lang == "en"`
- Evaluation: strict span/entity-level seqeval metrics over the same MultiNERD
  English test split

The active experiment scope is intentionally fixed to eight runs.

## Experiment Matrix

| Key | Config | Model | Regime |
| --- | --- | --- | --- |
| `deberta_base` | `ba-ner/configs/deberta_base.yaml` | `microsoft/deberta-v3-base` | encoder fine-tuning |
| `deberta_large` | `ba-ner/configs/deberta_large.yaml` | `microsoft/deberta-v3-large` | encoder fine-tuning |
| `qwen35_08b_zs` | `ba-ner/configs/qwen35_08b_zeroshot.yaml` | `Qwen/Qwen3.5-0.8B` | LLM zero-shot |
| `qwen35_4b_zs` | `ba-ner/configs/qwen35_4b_zeroshot.yaml` | `Qwen/Qwen3.5-4B` | LLM zero-shot |
| `qwen35_27b_zs` | `ba-ner/configs/qwen35_27b_zeroshot.yaml` | `Qwen/Qwen3.5-27B` | LLM zero-shot |
| `qwen35_08b` | `ba-ner/configs/qwen35_08b.yaml` | `Qwen/Qwen3.5-0.8B` | LLM LoRA/QLoRA fine-tuning |
| `qwen35_4b` | `ba-ner/configs/qwen35_4b.yaml` | `Qwen/Qwen3.5-4B` | LLM LoRA/QLoRA fine-tuning |
| `qwen35_27b` | `ba-ner/configs/qwen35_27b.yaml` | `Qwen/Qwen3.5-27B` | LLM LoRA/QLoRA fine-tuning |

The central registry is `ba-ner/src/config.py`. Runtime entry points validate
that configs match the final model IDs, dataset name, language, regime, and
output directory.

## Repository Layout

```text
ba-ner/
  configs/              # eight final experiment configs
  scripts/              # orchestration and local helper scripts
  scripts/cluster/      # SLURM jobs, preflight checks, submit orchestration
  src/config.py         # final experiment registry and config validation
  src/data/             # MultiNERD English loading and preprocessing
  src/encoder/          # DeBERTa training and inference
  src/decoder/          # Qwen zero-shot, LoRA/QLoRA training, parsing
  src/evaluate/         # shared metrics, comparison, error analysis
  tests/                # focused regression tests
  results/              # generated experiment artifacts
```

## Setup

From the repository root:

```bash
cd ba-ner
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For QLoRA runs, use a CUDA environment compatible with `torch`,
`bitsandbytes`, and the selected Qwen model size.

## GPU Cluster Setup

The repository is prepared for SLURM without requiring local training or
inference. All cluster scripts assume a Conda environment named `ba-ner`.

On the cluster, create and populate the environment once:

```bash
cd ba-ner
conda create -n ba-ner python=3.11 -y
conda activate ba-ner
pip install -r requirements.txt
```

Run the preflight check before submitting experiments. It validates package
availability, configs, CLI entry points, CUDA visibility, and tests; it does
not load the dataset or any model.

```bash
mkdir -p logs
sbatch scripts/cluster/preflight.sh
```

To submit the full eight-experiment matrix as separate SLURM jobs with
dependencies:

```bash
mkdir -p logs
scripts/cluster/submit_all.sh
```

The submit script uses `sbatch --parsable` and wires dependencies as follows:

- encoder inference waits for the corresponding encoder training job
- LoRA inference waits for the corresponding LoRA training job
- zero-shot jobs start directly
- final comparison waits for all eight inference jobs

Resource defaults are editable through environment variables before calling
`submit_all.sh`, for example:

```bash
GPU_QWEN=gpu:a100:1 MEM_QWEN_27B=160G TIME_QWEN_27B_TRAIN=48:00:00 \
  scripts/cluster/submit_all.sh
```

Individual job scripts are also available:

```bash
sbatch scripts/cluster/job_encoder_train.sh configs/deberta_base.yaml
sbatch scripts/cluster/job_encoder_infer.sh \
  configs/deberta_base.yaml results/multinerd/deberta-v3-base/best_model
sbatch scripts/cluster/job_decoder_zeroshot.sh configs/qwen35_4b_zeroshot.yaml
sbatch scripts/cluster/job_decoder_lora_train.sh configs/qwen35_4b.yaml
sbatch scripts/cluster/job_decoder_lora_infer.sh \
  configs/qwen35_4b.yaml results/multinerd/qwen35-4b-qlora/best_lora_adapter
sbatch scripts/cluster/job_compare.sh results
```

## Running Experiments

Run the full matrix:

```bash
cd ba-ner
python scripts/run_all.py
```

Useful subsets:

```bash
python scripts/run_all.py --encoder-only
python scripts/run_all.py --decoder-only
python scripts/run_all.py --zeroshot-only
python scripts/run_all.py --finetuned-only
python scripts/run_all.py --model qwen35_4b_zs
python scripts/run_all.py --eval-only
```

Encoder commands:

```bash
python -m src.encoder.train configs/deberta_base.yaml
python -m src.encoder.inference \
  --model results/multinerd/deberta-v3-base/best_model \
  --config configs/deberta_base.yaml
```

Decoder zero-shot command:

```bash
python -m src.decoder.inference \
  --zeroshot \
  --config configs/qwen35_4b_zeroshot.yaml
```

Decoder LoRA/QLoRA commands:

```bash
python -m src.decoder.train configs/qwen35_4b.yaml
python -m src.decoder.inference \
  --adapter results/multinerd/qwen35-4b-qlora/best_lora_adapter \
  --config configs/qwen35_4b.yaml
```

The optional `--base` argument in decoder inference is only a validation
override. If provided, it must match the `model_name` in the config.

## Outputs

All active outputs use this structure:

```text
ba-ner/results/multinerd/<experiment_name>/
  results.yaml              # training summary, where applicable
  inference_metrics.yaml    # final test metrics
  test_predictions.json     # per-sample predictions
  best_model/               # encoder checkpoints
  best_lora_adapter/        # decoder LoRA adapters
```

Training writes validation metrics only. Final test metrics are written by the
inference scripts so all eight experiments are evaluated through the same final
test-stage contract.

## Evaluation

Shared strict span-level metrics live in `ba-ner/src/evaluate/metrics.py`.
Encoder predictions are evaluated from BIO tags. LLM predictions are parsed
from JSON entities with token-based offsets, converted back to BIO tags directly
from those offsets, and then evaluated with the same seqeval wrapper.

LLM outputs must be JSON arrays:

```json
[
  {
    "start_token": 0,
    "end_token": 2,
    "text": "Barack Obama",
    "type": "PER"
  }
]
```

`start_token` is inclusive and `end_token` is exclusive. Both indices refer to
the numbered MultiNERD token list shown in the prompt, not to character offsets.
The `text` field is kept for readability and validation, but BIO reconstruction
uses `start_token` and `end_token` as the source of truth. If no entity is
present, the decoder must return exactly `[]`.

The parser handles direct JSON, fenced JSON blocks, extracted JSON arrays, and
Qwen-style `<think>...</think>` blocks. Parser diagnostics count failed parses,
wrong top-level schemas, invalid list items, missing fields, unknown entity
types, invalid token offsets, text/token mismatches, and overlapping spans.

Aggregate reports:

```bash
python -m src.evaluate.compare_all
python -m src.evaluate.error_analysis \
  --encoder-preds results/multinerd/deberta-v3-large/test_predictions.json \
  --decoder-preds results/multinerd/qwen35-27b-qlora/test_predictions.json
```

`compare_all.py` produces a console table, F1 plot, per-entity heatmap, and
LaTeX table from `results/multinerd/`.

## Tests

Run the focused regression tests from `ba-ner/`:

```bash
python -m pytest
```

The tests cover final config validation, the MultiNERD dataset contract, LLM
JSON parsing diagnostics, BIO conversion, shared metrics, and result
aggregation.

## Reproducibility Notes

- Configs set deterministic seeds where supported.
- Inference uses greedy decoding for LLMs.
- Dataset and model IDs are centralized in `src/config.py`.
- Result files include experiment name, model ID, dataset, language, regime,
  seed, final metrics, and `run_metadata`.
- `run_metadata` records git state, Python/platform details, package versions,
  CUDA visibility, and GPU names when the run executes.
