# Cluster Job Guide

This guide covers environment setup, submitting training and inference jobs,
monitoring them, finding outputs, and diagnosing common failures on the SLURM
cluster. Run all commands from the `ba-ner/` directory unless stated otherwise.

## 1. Create the Python environment

The cluster login node has an older virtual CPU. NumPy 2.4 and newer require
the x86-64-v2 CPU baseline and cannot be imported there. The repository pins
NumPy below 2.4 for that reason.

### Project-local virtual environment

Use this option when Conda is not installed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify that the environment is active and importable:

```bash
command -v python
python --version
python -c 'import numpy; print(numpy.__version__)'
```

The Python path should end in `ba-ner/.venv/bin/python`, and NumPy should be
older than 2.4. Do not use `--break-system-packages`; that modifies the
operating system's protected Python installation.

### Conda environment

If Conda is available:

```bash
conda create -n ba-ner python=3.11 -y
conda activate ba-ner
python -m pip install -r requirements.txt
```

The cluster scripts use the environment selected with `BA_NER_VENV` first,
then a Conda environment named by `BA_NER_CONDA_ENV`, and finally the local
`.venv` as a fallback.

## 2. Run a preflight job

Do not run training directly on the login node. Submit a preflight job to check
the Python packages, configuration files, tests, and CUDA visibility on a
compute node:

```bash
mkdir -p logs
sbatch scripts/cluster/preflight.sh
```

If the virtual environment is stored somewhere other than `ba-ner/.venv`,
submit with its absolute path:

```bash
export BA_NER_VENV=/absolute/path/to/venv
sbatch --export=ALL,BA_NER_VENV scripts/cluster/preflight.sh
```

## 3. Submit individual jobs

The submission command prints a numeric job ID. Keep that ID; it identifies the
job in the queue, accounting records, and log filenames.

### Encoder training

```bash
sbatch scripts/cluster/job_encoder_train.sh configs/deberta_base.yaml
sbatch scripts/cluster/job_encoder_train.sh configs/deberta_large.yaml
```

### Encoder inference

Only run inference after the matching training job succeeds:

```bash
sbatch scripts/cluster/job_encoder_infer.sh \
  configs/deberta_base.yaml \
  results/multinerd/deberta-v3-base/best_model
```

### Decoder QLoRA training

```bash
sbatch scripts/cluster/job_decoder_lora_train.sh configs/qwen35_08b.yaml
sbatch scripts/cluster/job_decoder_lora_train.sh configs/qwen35_4b.yaml
sbatch scripts/cluster/job_decoder_lora_train.sh configs/qwen35_27b.yaml
```

The decoder scripts request an A100 by default. If the cluster uses a different
GPU resource name, inspect the available generic resources with:

```bash
sinfo -o '%P %G'
```

### Decoder zero-shot inference

```bash
sbatch scripts/cluster/job_decoder_zeroshot.sh configs/qwen35_08b_zeroshot.yaml
sbatch scripts/cluster/job_decoder_zeroshot.sh configs/qwen35_4b_zeroshot.yaml
sbatch scripts/cluster/job_decoder_zeroshot.sh configs/qwen35_27b_zeroshot.yaml
```

### Full experiment matrix

This command submits all eight experiments with train-to-inference dependencies
and runs the comparison job after all inference jobs succeed:

```bash
scripts/cluster/submit_all.sh
```

### Canonical seed study

The historical eight-experiment command above remains available and is not
used to duplicate zero-shot runs. The canonical seed study has its own guarded
orchestrator:

```bash
python scripts/run_seed_matrix.py --preflight
python scripts/run_seed_matrix.py --dry-run
python scripts/run_seed_matrix.py --submit
```

`--submit` runs tests and config/hash/path checks locally, then submits and
waits for a CUDA compute-node preflight. Only after that job succeeds are the
fifteen training pipelines submitted. Each run uses `afterok` dependencies:

```text
training -> inference -> strict saved-prediction evaluation
```

The aggregate job uses `afterany` across evaluation jobs. If a run fails and
its downstream jobs are dependency-cancelled, aggregation still writes an
explicitly incomplete report rather than silently producing nothing.

Job and log names contain model, 27B variant where applicable, seed, phase, and
SLURM job ID. Logs are stored below
`logs/seed-studies/<group>/seed-<seed>/<phase>-%j.{out,err}`. The successful
reference resources are reused. Qwen3.5-27B uses the historical H100-Trails,
160G, three-day allocation with SLURM requeue; any continuation is allowed only
at a checkpoint owned by the exact same seed/run/hash. The historical 2ep path
is read-only and cannot be a resume source for the 3ep variant.

Useful subsets and recovery commands:

```bash
python scripts/run_seed_matrix.py --encoder-only --dry-run
python scripts/run_seed_matrix.py --decoder-only --dry-run
python scripts/run_seed_matrix.py --model qwen35_27b_3ep --dry-run
python scripts/run_seed_matrix.py --model qwen35_08b --model qwen35_4b --model qwen35_27b_3ep --dry-run
python scripts/run_seed_matrix.py --seeds 123 --submit
python scripts/run_seed_matrix.py --seeds 456 --submit
python scripts/run_seed_matrix.py --resume-failed --submit
python scripts/run_seed_matrix.py --inference-only --submit
python scripts/run_seed_matrix.py --evaluation-only --submit
python scripts/run_seed_matrix.py --aggregate-only
```

Subset submissions carry their exact resolved config list into the compute-node
preflight. Its provenance, path, cache, resource, and CUDA checks therefore
apply only to the selected runs; this does not bypass any validation and cannot
expand the request into a model-by-seed cross product.

Do not use `--resume-failed` to change a seed, variant, model, output path, epoch
limit, or config hash; the runtime guard rejects all such attempts.

All five model groups use fresh managed runs for Seeds 42, 123, and 456. Their
five earlier Seed-42 outputs are historical, read-only, and excluded from the
primary aggregates, so gaps in legacy worktree evidence cannot contaminate or
block the canonical cohort.

Submission requires a clean committed repository. `--submit` records the
current HEAD in `BA_NER_EXPECTED_GIT_COMMIT`; the CUDA preflight and every
training, inference, evaluation, and aggregation job verify that exact commit
and reject any dirty worktree before running. Commit and freeze the experiment
definition before submission, and do not edit the checkout while the matrix is
active. The configured scratch filesystem must also have at least 500 GiB free.

## 4. Monitor jobs

List all of your queued and running jobs:

```bash
squeue -u "$USER"
```

Inspect one job:

```bash
squeue -j JOB_ID
scontrol show job JOB_ID
```

The repository monitor includes all 15 canonical fine-tuning entries plus five
historical Seed-42 entries:

```bash
python scripts/live_monitor.py --refresh 10
python scripts/live_monitor.py --model qwen35-27b --all
python scripts/live_monitor.py --variant 3ep
python scripts/live_monitor.py --canonical
python scripts/live_monitor.py --historical
python scripts/run_seed_matrix.py --status
```

It reads structured status/metadata first, then trainer state, manifests, SLURM,
and logs. Missing/corrupt files or unavailable SLURM commands do not stop the
monitor, and it never starts, stops, or modifies training.

Typical SLURM states are:

- `PD`: pending and waiting for resources or a dependency.
- `R`: running.
- `CG`: completing.
- `CD`: completed successfully; normally visible through `sacct`, not `squeue`.
- `F`: failed.
- `OOM`: exceeded the requested memory.
- `CA`: cancelled.

A job disappearing from `squeue` means it finished, failed, or was cancelled.
Check its final status with:

```bash
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS
```

## 5. Follow logs

Logs are written under `logs/`. Replace `JOB_ID` with the actual numeric ID
printed by `sbatch`; do not type the word `JOB_ID` literally.

```bash
tail -F logs/JOB_NAME_JOB_ID.out logs/JOB_NAME_JOB_ID.err
```

For example, if `ner-enc-base` was submitted as job `3199655`:

```bash
tail -F logs/ner-enc-base_3199655.out logs/ner-enc-base_3199655.err
```

Press `Ctrl+C` to stop following the logs. This does not stop the SLURM job.
Batch-mode Python output can be buffered, so an empty `.out` file during early
startup does not necessarily indicate a problem. Always inspect `.err` too.

Find the most recently modified logs:

```bash
ls -lt logs | head
```

## 6. Cancel jobs

Cancel one job:

```bash
scancel JOB_ID
```

Cancel all of your jobs only when that is really intended:

```bash
scancel -u "$USER"
```

## 7. Find results

Completed runs write artifacts below:

```text
results/multinerd/<experiment_name>/
  results.yaml
  inference_metrics.yaml
  test_predictions.json
  best_model/
  best_lora_adapter/
```

Encoder training produces `best_model/`. QLoRA training produces
`best_lora_adapter/`. Inference produces the final metrics and predictions.

## 8. Common failures

### `externally-managed-environment`

The active `python` is the protected system Python. Activate `.venv` before
installing anything:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### `NumPy was built with baseline optimizations: (X86_V2)`

NumPy 2.4 or newer was installed on the old login CPU. Repair the virtual
environment with:

```bash
source .venv/bin/activate
python -m pip install --force-reinstall 'numpy>=1.26,<2.4'
```

### `source: not found` from `sbatch --wrap`

`sbatch --wrap` may execute through `/bin/sh`, where `source` is unavailable.
Use the POSIX dot command instead:

```bash
--wrap='cd /absolute/path/to/ba-ner && . .venv/bin/activate && python ...'
```

The repository's regular job scripts use Bash and handle environment activation
automatically, so prefer them over long `--wrap` commands.

### Job vanishes from `squeue`

Check accounting and both log files:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,MaxRSS
tail -n 200 logs/JOB_NAME_JOB_ID.out
tail -n 200 logs/JOB_NAME_JOB_ID.err
```

### Out of memory

Confirm `OUT_OF_MEMORY` or `OOM` with `sacct`, then reduce the configured batch
size or request more memory/VRAM. Do not immediately resubmit the unchanged job.

### Model or dataset download failure

Check whether compute nodes have internet access and whether the Hugging Face
cache is writable. The job scripts place caches under `ba-ner/.hf_cache` by
default. If authentication is required, log in before submission and ensure the
compute job can access the stored token.
