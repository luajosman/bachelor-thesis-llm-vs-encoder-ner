# Encoder vs. LLMs for Named Entity Recognition

Bachelor thesis repository comparing encoder token classification with
decoder-only LLM structured extraction for Named Entity Recognition (NER).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-red)
![Transformers](https://img.shields.io/badge/Transformers-4.48%2B-yellow)
![PEFT](https://img.shields.io/badge/PEFT-LoRA%2FQLoRA-purple)
![Dataset](https://img.shields.io/badge/Dataset-MultiNERD%20English-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Project Overview

This project supports the Bachelor thesis **"Comparison of Encoder-Based Models
and Large Language Models with Fine-Tuning for Named Entity Recognition."** It
compares supervised DeBERTa token-classification models with Qwen3.5
decoder-only LLMs that perform structured NER extraction. The benchmark is the
English subset of `Babelscape/multinerd`, so all models are evaluated on the
same task and language setting. The LLM side includes both zero-shot evaluation
and LoRA/QLoRA fine-tuning.

## Research Question

How do fine-tuned encoder models compare with decoder-only LLMs, both zero-shot
and LoRA-fine-tuned, for English NER in terms of entity-level quality,
computational efficiency, structured-output robustness, and implementation
complexity?

## Experimental Scope

The implemented experiment registry in `ba-ner/src/config.py` defines exactly
eight runs.

| Model family | Hugging Face model ID | Implemented mode |
| --- | --- | --- |
| DeBERTa-v3-base | `microsoft/deberta-v3-base` | fine-tuned encoder token classification |
| DeBERTa-v3-large | `microsoft/deberta-v3-large` | fine-tuned encoder token classification |
| Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` | zero-shot decoder inference; LoRA/QLoRA fine-tuning |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | zero-shot decoder inference; LoRA/QLoRA fine-tuning |
| Qwen3.5-27B | `Qwen/Qwen3.5-27B` | zero-shot decoder inference; LoRA/QLoRA fine-tuning |

There are no zero-shot encoder experiments in the active study design.

## Core Architecture

The encoder pipeline uses MultiNERD word-level tokens and BIO labels, aligns
labels to tokenizer subwords, and trains Hugging Face token-classification
models. Continuation subwords and special tokens are assigned `-100` so they are
ignored by the loss and evaluation.

The decoder pipeline presents each example as a numbered token list and asks
the LLM to generate a machine-readable JSON array of entity spans. Outputs are
parsed, validated, converted back to BIO tags, and evaluated with the same
strict entity-level metric code as the encoders.

LoRA/QLoRA fine-tuning is implemented through PEFT and TRL. The Qwen base
weights remain frozen while adapter parameters are trained; the best adapter is
selected using generative validation F1.

## Dataset and Representation

The repository intentionally supports one dataset setup:

- Dataset: `Babelscape/multinerd`
- Language: English only, filtered with `lang == "en"`
- Splits: `train`, `validation`, `test`
- Entity types: `PER`, `ORG`, `LOC`, `ANIM`, `BIO`, `CEL`, `DIS`, `EVE`,
  `FOOD`, `INST`, `MEDIA`, `MYTH`, `PLANT`, `TIME`, `VEHI`

Encoder models consume BIO labels directly. Decoder models use a token-offset
JSON schema:

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
the numbered MultiNERD token list shown in the prompt, not to character
offsets. The `text` field is retained for readability and validation, but BIO
reconstruction uses the token offsets as the source of truth. If no entity is
present, the expected decoder output is exactly `[]`.

### LLM Output Format Rationale

The LLM pipeline is generative, but the final evaluation requires entity spans
that can be mapped back to the original MultiNERD token sequence without
ambiguity. For that reason, each decoder prompt presents the sentence as a
numbered token list:

```text
Tokens:
0: Barack
1: Obama
2: visited
3: Berlin
4: .
```

For this input, the expected LLM output is:

```json
[
  {
    "start_token": 0,
    "end_token": 2,
    "text": "Barack Obama",
    "type": "PER"
  },
  {
    "start_token": 3,
    "end_token": 4,
    "text": "Berlin",
    "type": "LOC"
  }
]
```

The fields have the following meaning:

- `start_token`: inclusive start index in the numbered token list.
- `end_token`: exclusive end index in the numbered token list.
- `text`: human-readable entity span, used for consistency checks.
- `type`: one of the allowed MultiNERD entity types.

This representation was chosen instead of a flatter format such as
`{"entity": "Berlin", "type": "LOC"}` because the entity text alone is not
always enough to identify the intended span. If the same surface form appears
multiple times in one sentence, string matching cannot reliably determine which
occurrence the model predicted. Token offsets make the prediction explicit:
the span points directly to `tokens[start_token:end_token]`.

The `text` field is therefore not the primary alignment source. It is kept so
the JSON remains readable and so the parser can detect inconsistent outputs,
for example offsets that point to `Barack Obama` while the generated `text`
claims only `Obama`. The actual BIO reconstruction is deterministic:

```text
start_token = 0, end_token = 2, type = PER  ->  B-PER I-PER
start_token = 3, end_token = 4, type = LOC  ->  B-LOC
```

For the example above, the resulting BIO sequence is:

```text
B-PER I-PER O B-LOC O
```

This design avoids ambiguous string matching, makes repeated entities
distinguishable, reduces punctuation and tokenization alignment errors, and
makes malformed LLM outputs measurable through parser diagnostics. It also
keeps the comparison with encoder models methodologically clean: both model
families are ultimately evaluated as BIO tag sequences with the same strict
entity-level `seqeval` metrics.

## Evaluation Strategy

The primary comparison uses strict entity-level metrics through `seqeval`:

- Precision
- Recall
- F1-score

Encoder predictions are evaluated directly from BIO tags. Decoder predictions
are validated as JSON spans and then converted deterministically to BIO tags.
This keeps the final metric contract shared across model families.

Decoder-specific robustness diagnostics include:

- parse failure rate
- wrong top-level schema
- invalid list items or missing fields
- unknown entity types
- invalid token offsets
- text/token mismatches
- overlapping spans

Efficiency-related outputs include training runtime where applicable, mean and
p95 inference latency, peak VRAM usage, and total/trainable parameter counts.

## Repository Structure

```text
ba-ner/
  configs/              # historical matrix plus canonical seed-study configs/manifest
  scripts/              # local orchestration scripts
  scripts/cluster/      # SLURM preflight, jobs, and dependency submission
  src/config.py         # experiment registry and config validation
  src/data/             # MultiNERD English loading and preprocessing
  src/encoder/          # DeBERTa training and inference
  src/decoder/          # Qwen zero-shot, LoRA/QLoRA, parsing, inference
  src/evaluate/         # shared metrics, comparison, error analysis, efficiency
  tests/                # focused regression tests
  results/              # generated experiment artifacts
```

## Setup

Python `>=3.10` is required. From the repository root:

```bash
cd ba-ner
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Qwen and QLoRA runs, use a CUDA environment compatible with `torch`,
`bitsandbytes`, and the selected model size.

## Running Experiments

Run the full local orchestration pipeline:

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

Encoder training and inference:

```bash
python -m src.encoder.train configs/deberta_base.yaml
python -m src.encoder.inference \
  --model results/multinerd/deberta-v3-base/best_model \
  --config configs/deberta_base.yaml
```

LLM zero-shot inference:

```bash
python -m src.decoder.inference \
  --zeroshot \
  --config configs/qwen35_4b_zeroshot.yaml
```

LLM LoRA/QLoRA fine-tuning and inference:

```bash
python -m src.decoder.train configs/qwen35_4b.yaml
python -m src.decoder.inference \
  --adapter results/multinerd/qwen35-4b-qlora/best_lora_adapter \
  --config configs/qwen35_4b.yaml
```

Aggregate comparison and qualitative error analysis:

```bash
python -m src.evaluate.compare_all
python -m src.evaluate.error_analysis \
  --encoder-preds results/multinerd/deberta-v3-large/test_predictions.json \
  --decoder-preds results/multinerd/qwen35-27b-qlora/test_predictions.json
```

## Canonical Multi-Seed Study

Training variance is measured with seeds `42`, `123`, and `456` for five
fine-tuned model configurations. Every group is trained as a fresh managed
cohort with seeds 42, 123, and 456, giving exactly fifteen new training runs.
This uniform design avoids mixing legacy Seed-42 results with newly managed
runs. All five earlier Seed-42 outputs remain read-only historical evidence and
are excluded from the primary seed aggregates.

The original Qwen3.5-0.8B seed-42 run is preserved at
`results/multinerd/qwen35-08b-qlora` as read-only historical evidence. Its
initial worktree is not fully recoverable, so it is excluded from the primary
seed aggregate. The fresh managed cohort writes to
`results/multinerd/qwen35-08b-qlora-canonical{,-seed123,-seed456}` and starts
all three runs from the pinned base model without reusing the historical path.

The original Qwen3.5-27B seed-42 run used two epochs and is preserved at
`results/multinerd/qwen35-27b-qlora` as a read-only historical/exploratory
run. It is never continued into the canonical setup and is excluded from the
primary 27B seed aggregate. Its cosine schedule already reached its configured
endpoint; adding an epoch from that checkpoint would change or restart the
scheduler trajectory and would therefore be a separate continued-training
ablation. The canonical replacement starts from the unchanged base model with
new QLoRA adapters, optimizer, scheduler, and RNG state at
`results/multinerd/qwen35-27b-qlora-3ep`.
The read-only inventory was refreshed on 2026-08-07: training job `3219771`
and inference job `3226684` are both completed.

Within each canonical group, only the seed and operational identity/path fields
may differ. `configs/seed_study_manifest.yaml` records canonical and historical
roles, legacy paths, reference provenance, and SLURM resources. Recursive
validation blocks missing, extra, type, list, or value differences outside the
allowlist. Each run records:

- `scientific_config_hash`: SHA-256 of the fully resolved scientific/technical
  config plus immutable model/dataset revisions, preprocessing/evaluation code,
  prompt, parser, thinking/decoding policy, and checkpoint-selection contract.
  Only seed and operational identity are removed; the hash must match within a
  seed group.
- `full_run_config_hash`: SHA-256 of the complete resolved run config; it must
  be unique per run.

New run directories contain source/resolved configs, run metadata, environment
versions, an atomic `status.json`, a guarded run lock, training/evaluation
artifacts, predictions, and both hashes. Existing result directories are not
migrated or modified. Cross-seed and cross-variant resumes are blocked. A
resume is possible only for an interrupted instance of the exact same run with
matching model, seed, epoch count, output path, and both hashes.

Decoder model selection uses the highest generative validation F1 and test
inference accepts only `best_lora_adapter`; encoders use the best validation-F1
`best_model`. The test split is not used for checkpoint selection. For the
canonical decoder setup: “Models were trained for up to three epochs, with the
checkpoint achieving the highest generative validation F1 selected for test
evaluation.” DeBERTa retains its existing reference epoch limits (five for base
and four for large).

The three zero-shot Qwen experiments are not repeated. Their inference is
greedy with `do_sample=false`, so seed count and standard deviation are reported
as `not_applicable`, not as an artificial zero.

### Validation and orchestration commands

Run from `ba-ner/` with the project environment active:

```bash
# Full local preflight (includes pytest)
python scripts/run_seed_matrix.py --preflight

# Complete fifteen-run dry run
python scripts/run_seed_matrix.py --dry-run

# Subset dry runs
python scripts/run_seed_matrix.py --encoder-only --dry-run
python scripts/run_seed_matrix.py --decoder-only --dry-run
python scripts/run_seed_matrix.py --model qwen35_27b_3ep --dry-run

# Submit all fifteen train -> inference -> evaluation pipelines.
# This first waits for a successful CUDA compute-node preflight.
python scripts/run_seed_matrix.py --submit

# Submit seed/model subsets
python scripts/run_seed_matrix.py --seeds 123 --submit
python scripts/run_seed_matrix.py --seeds 456 --submit
python scripts/run_seed_matrix.py --model qwen35_27b_3ep --seeds 42 --submit
python scripts/run_seed_matrix.py --model qwen35_27b_3ep --seeds 42 123 456 --submit

# Status and non-mutating live monitor
python scripts/run_seed_matrix.py --status
python scripts/live_monitor.py --refresh 10
python scripts/live_monitor.py --model qwen35-27b --all
python scripts/live_monitor.py --canonical
python scripts/live_monitor.py --variant 3ep --seed 42

# Re-run downstream phases without training
python scripts/run_seed_matrix.py --inference-only --submit
python scripts/run_seed_matrix.py --evaluation-only --submit

# Aggregate one group or every group (sample standard deviation, ddof=1)
python scripts/run_seed_matrix.py --aggregate-only --group deberta-v3-base
python scripts/run_seed_matrix.py --aggregate-only

# Explicit recovery of failed runs only; hashes and checkpoint ownership are
# revalidated before any same-run resume/restart is allowed.
python scripts/run_seed_matrix.py --resume-failed --submit
```

For subset submissions, the orchestrator passes the exact selected config paths
to the CUDA compute-node preflight. Provenance gates from unselected groups do
not block a verified subset, and the selection cannot expand into an unintended
model-by-seed cross product.

Completed and currently running runs are skipped by default. A failed or
unknown existing path is never deleted or silently reused. Inference depends on
successful training, evaluation depends on successful inference, and the final
aggregation uses SLURM `afterany` across selected evaluation jobs. This allows
an explicitly incomplete aggregate to report dependency-cancelled or failed
runs instead of silently producing no report.

Aggregation writes `seed_summary.yaml`, `seed_summary.csv`,
`seed_metrics.json`, and `missing_or_failed_runs.yaml` per group below
`results/seed_studies/multinerd/<group>/aggregate/`, plus wide and long-format
cross-model tables. It reports mean, sample standard deviation (`ddof=1`), min,
max, successful/expected counts, and explicit missing/failed seeds. With fewer
than two successful runs, standard deviation is `null` rather than zero. The
five historical Seed-42 results remain visible in separate historical sections
but are never part of the primary means.

A fixed seed improves reproducibility but does not guarantee bitwise-identical
results across GPU models, CUDA kernels, library versions, or other
nondeterministic hardware paths. Run metadata records the actual environment so
remaining differences can be reported.

### Frozen canonical execution

The historical provenance gaps no longer block the study because none of the
legacy outputs is used in a canonical aggregate. The new fifteen-run cohort is
fully managed from Seed 42 onward. `--submit` additionally requires a clean,
committed worktree. The orchestrator exports the exact Git commit to every
SLURM job; the compute preflight and every downstream phase reject a changed
HEAD or dirty worktree before executing scientific code. Local preflight also
requires at least 500 GiB free in the configured scratch filesystem.

Consequently, commit the complete study definition first, leave the worktree
clean, and only then run `python scripts/run_seed_matrix.py --submit`. The
evidence inventory and design decision are recorded in
[the seed-study provenance audit](ba-ner/docs/seed-study-provenance.md).

## GPU Cluster

The repository includes SLURM scripts for running the matrix on a GPU cluster.
They support either a project-local `.venv` or a Conda environment named
`ba-ner`.

```bash
cd ba-ner
conda create -n ba-ner python=3.11 -y
conda activate ba-ner
pip install -r requirements.txt
```

Preflight validates imports, configs, CLI entry points, CUDA visibility, and
tests without loading datasets or models:

```bash
mkdir -p logs
sbatch scripts/cluster/preflight.sh
```

Submit the full eight-experiment matrix with train-to-inference dependencies:

```bash
mkdir -p logs
scripts/cluster/submit_all.sh
```

Resource defaults can be adjusted through environment variables before calling
`submit_all.sh`.

For day-to-day commands, job monitoring, log inspection, cancellation, and
troubleshooting, see [Cluster job guide](ba-ner/docs/cluster-jobs.md).

## Outputs and Reproducibility

Experiment artifacts are written below:

```text
results/multinerd/<experiment_name>/
  results.yaml              # training summary, where applicable
  inference_metrics.yaml    # final test metrics
  test_predictions.json     # per-sample predictions
  best_model/               # encoder checkpoints
  best_lora_adapter/        # decoder LoRA adapters
```

The codebase uses config-validated runs, fixed seeds where supported, and
greedy decoding for LLM inference. Result metadata records the experiment name,
model ID, dataset, language, regime, seed, metrics, git state, Python/platform
details, package versions, CUDA visibility, and GPU names.

The repository provides scripts and configs for the final study. Completed
experimental results are only present after running the relevant training and
inference jobs.

## Tests

Run the regression tests from `ba-ner/`:

```bash
python -m pytest
```

The tests cover config validation, the MultiNERD dataset contract, decoder
prompt formatting, token-offset JSON parsing, BIO conversion, result
aggregation, and reproducibility metadata collection.

## Author and Thesis Context

Author: **Luaj Osman**

Bachelor thesis project: **Comparison of Encoder-Based Models and Large
Language Models with Fine-Tuning for Named Entity Recognition**

## License and Academic Use

This repository is released under the MIT License. It is intended for academic
research and reproducible experimentation in the context of the Bachelor thesis.
