# Seed-study provenance audit

Audit date: 2026-08-07 (Europe/Berlin)

This document records the hard submission decision for the canonical MultiNERD
seed study. It supplements the machine-readable
`configs/seed_study_manifest.yaml`; it does not modify any historical output.

## Immutable upstream revisions

| Artifact | Revision |
|---|---|
| `Babelscape/multinerd` | `2814b78e7af4b5a1f1886fe7ad49632de4d9dd25` |
| `microsoft/deberta-v3-base` | `8ccc9b6f36199bec6961081d44eb72fb3f7353f3` |
| `microsoft/deberta-v3-large` | `64a8c8eab3e352a784c658aef62be1662607476f` |
| `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` |
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| `Qwen/Qwen3.5-27B` | `fc05daec18b0a78c049392ed2e771dde82bdf654` |

All six cached `refs/main` values and their snapshot directories were observed
at these revisions. New managed runs pass the immutable revisions explicitly to
the Hugging Face loaders; runtime dataset fingerprints are stored per split.

## Historical reference evidence

| Canonical group | Recorded commit | Worktree evidence | Decision |
|---|---|---|---|
| DeBERTa-v3-base | `5c5b74f5f1fe17e317680871ce51373c3ce60f79` | Dirty; exact binary diff and untracked-file snapshot unavailable | Historical only; verified fresh replacement |
| DeBERTa-v3-large | `5c5b74f5f1fe17e317680871ce51373c3ce60f79` | Dirty; exact binary diff and untracked-file snapshot unavailable | Historical only; verified fresh replacement |
| Qwen3.5-0.8B QLoRA | fresh canonical group; original final resume at `cdf10db7914f70b28f37823a8a6152e46bd2e7ca` | Original initial worktree is incomplete; original output is now historical/read-only | Verified fresh replacement |
| Qwen3.5-4B QLoRA | `137bffdd9b4351b35a31e4550bde0c15b3c53a19` | Clean commit available in the evidence clone | Historical only; fresh replacement for a uniform design |
| Qwen3.5-27B 3ep | fresh canonical group derived from the preserved 2ep config | Recursive config diff permits exactly the epoch and identity changes | Verified |

The evidence clone inspected read-only is
`/home/losman/bachelor-thesis-llm-vs-encoder-ner-1`. The Qwen3.5-0.8B cached
tokenizer template was also inspected: omitting `enable_thinking` selects the
same non-thinking branch as explicitly passing `false` for the pinned revision.
That semantic check does not recover the missing initial worktree. To apply one
uniform rule to every model size and family, the study establishes fresh
managed Seeds 42, 123, and 456 for all five groups at separate canonical paths.
All earlier Seed-42 outputs are excluded from the primary aggregates.

The current scientific contract hashes the resolved config, upstream
revisions, dataset split contract, preprocessing, model-family training and
inference paths, evaluation code, prompt, parser, decoding/thinking policy, and
checkpoint-selection policy. A separate execution snapshot hashes every Python,
shell, config, requirements, and setup file used by a managed run, including
untracked files. The exact snapshot is written into each new run directory.

## Historical Qwen3.5-27B status

The original two-epoch run remains read-only and excluded from the primary
three-seed aggregate.

| Phase | Job | State | Start | End | Elapsed | Exit |
|---|---:|---|---|---|---|---|
| Training | `3219771` | `COMPLETED` | 2026-07-31 17:01:52 | 2026-08-03 16:00:54 | 2-22:59:02 | 0:0 |
| Inference | `3226684` | `COMPLETED` | 2026-08-03 18:54:06 | 2026-08-04 18:29:20 | 23:35:14 | 0:0 |

No seed-study job was submitted. At audit time the only listed user job was the
unrelated pending monitor job `3245114` (`ner-training-monitor`).

## Validation outcome and frozen execution

The full dry run resolves exactly fifteen unique planned output paths: three
fresh runs for each of DeBERTa-v3-base, DeBERTa-v3-large, Qwen3.5-0.8B,
Qwen3.5-4B, and Qwen3.5-27B-3ep. It prints the corresponding training,
inference, and evaluation `sbatch` commands. No historical output is a resume
source or canonical observation. Login-node CUDA absence is an expected warning
and CUDA is enforced by the compute-node preflight before any training
submission. At least 500 GiB of free scratch space is required.

The final login-node preflight and the exact fifteen-config compute-preflight
simulation both passed on 2026-08-07. The regression suite passed all 94 tests;
all fifteen destination paths were free, and no path, scientific-hash, resume,
or canonical-group conflict was reported. No training job was submitted during
validation.

The complete matrix is started only from the committed, clean study definition:

```bash
cd ba-ner
python scripts/run_seed_matrix.py --preflight
python scripts/run_seed_matrix.py --dry-run
python scripts/run_seed_matrix.py --submit
```

`--submit` cannot bypass the local gates. It first submits a CUDA compute-node
preflight, waits for success, then records every submitted job atomically. It
exports the submitting repository HEAD as `BA_NER_EXPECTED_GIT_COMMIT`; the
preflight and all downstream jobs verify that HEAD and require a clean worktree
before they execute. The complete phase-specific commands can always be
regenerated without submission with `--dry-run`; no placeholder IDs are used.
