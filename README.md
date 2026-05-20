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
  configs/              # eight final experiment configs
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

## GPU Cluster

The repository includes SLURM scripts for running the matrix on a GPU cluster.
They assume a Conda environment named `ba-ner`.

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
