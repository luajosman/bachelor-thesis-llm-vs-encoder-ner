# Encoder vs. LLMs for Named Entity Recognition

Code, configurations, evaluation scripts and LaTeX sources of the Bachelor
thesis

> **Fine-Tuning für Named Entity Recognition: Ein Vergleich von BERT-basierten
> Modellen und LLMs mit LoRA**
> Luaj Osman, Technische Universität Berlin, Fakultät IV, Fachgebiet Quality &
> Usability, in cooperation with DFKI, 26 November 2026.

The thesis (in German) compares fine-tuned DeBERTa-v3 encoders (token
classification) with Qwen3.5 decoder-only LLMs (0.8B, 4B, 27B; zero-shot and
QLoRA fine-tuned, JSON span extraction) on the English part of MultiNERD, with
a domain-transfer evaluation on CrossNER.

## Submitted version

- The submitted thesis is the PDF built from the LaTeX sources in `thesis/`
  (`thesis/main.tex`), 104 pages, dated 26 November 2026 on the title page.
- The built PDF (`thesis/main.pdf`) is **not** versioned; it is listed in
  `.gitignore` and is submitted separately.
- Earlier drafts are not the submitted version: tag
  `betreuerentwurf-2026-11-08` (draft for the supervisor) and branch
  `fassung-lang-2026-11-12` (long version before shortening).
- Tag `zeroshot-runs-137bffd` marks the code state of the zero-shot runs
  evaluated in the thesis (see below). The commits of all other runs are
  listed in Appendix A.1 of the thesis.

## Repository layout

| Path | Content |
| --- | --- |
| `thesis/main.tex`, `chapters/`, `frontmatter/`, `figures/`, `references.bib` | LaTeX sources of the thesis |
| `thesis/generated/` | Table bodies (`tab-*.tex`), figure data (`fig-*.dat`) and all reported figures (`summary.json`), written by the aggregation script; not edited by hand |
| `thesis/scripts/aggregate_results.py` | Aggregation of the run artifacts into `thesis/generated/`, with tests in `test_aggregate_results.py` |
| `thesis/scripts/check/` | Consistency checks for the manuscript (numbers, page layout, wording, text overlap with sources) |
| `ba-ner/src/` | Experiment code: data loading (`data/`), DeBERTa (`encoder/`), Qwen zero-shot and QLoRA (`decoder/`), metrics and evaluations (`evaluate/`) |
| `ba-ner/configs/` | One YAML configuration per run, plus `seed_study_manifest.yaml` and `crossner_overlap_exclusions.yaml` |
| `ba-ner/scripts/` | Orchestration (`run_seed_matrix.py`, `run_crossner.py`) and statistics helpers; `scripts/cluster/` holds the SLURM job and submit scripts |
| `ba-ner/tests/` | Tests of the experiment code |
| `ba-ner/docs/` | Cluster job guide and seed-study provenance audit |
| `ba-ner/results/` | Local copy of the run artifacts (only `.gitkeep` is versioned) |
| `sync_from_cluster.sh` | Fetches cluster commits and small result files (YAML/JSON/CSV) into `ba-ner/results/` |
| `.claude/agents/` | Claude Code agent definitions used for AI-assisted revision (AI use is declared in the thesis) |

## Configurations, run artifacts and evaluation

**Configurations** (`ba-ner/configs/`): the runs of the main study use the
`*_canonical.yaml` files for seed 42 and `*_seed123.yaml` / `*_seed456.yaml`
for the other seeds; the 27B model uses `qwen35_27b_3ep*.yaml`. Zero-shot runs
use `*_zeroshot.yaml`. Files without a suffix (e.g. `deberta_base.yaml`,
`qwen35_27b.yaml`) belong to the historical seed-42 runs (exploratory
pre-runs, roles in `seed_study_manifest.yaml`) that are not part of the
reported seed means. Appendix A.1 of the thesis maps every run to its
configuration file, output directory and commit.

**Run artifacts** (predictions, metrics, run metadata, adapters; about 2 GB
without model weights) are not in Git. They were produced on the DFKI cluster
below `/netscratch/losman/ba-ner/results` and are available to the examiners
on request. A local copy is expected in `ba-ner/results/` with the
subdirectories `multinerd/`, `crossner/`, `crossner_output_limit_1024/`,
`multinerd_output_limit_1024/`, `multinerd_control_1024/` and the SLURM export
`slurm_accounting_2026-08.csv`.

**Evaluation**: `thesis/scripts/aggregate_results.py` reads these artifacts
and `ba-ner/configs/crossner_overlap_exclusions.yaml`, recomputes the strict
entity-level scores, stops if they differ from the stored values, and writes
`thesis/generated/`. The bootstrap uses a fixed seed, so the output is
byte-reproducible.

**Zero-shot note**: the evaluated zero-shot runs (tag `zeroshot-runs-137bffd`)
decoded with sampling and the glossary prompt; the current code decodes
greedily. The thesis reports this deviation in Appendix A.1.

## Commands

The commands below are taken from the project files (`.latexmkrc`, script
headers and argument parsers). **They were not run when this README was
written.**

Build the thesis (pdflatex + biber; auxiliary files go to `thesis/build/`, the
PDF to `thesis/main.pdf`):

```bash
cd thesis && latexmk -pdf main.tex
# without a local TeX installation:
cd thesis && docker run --rm -v "$PWD":/work -w /work texlive/texlive:latest latexmk -pdf main.tex
```

Rebuild the tables and figures from the run artifacts (needs PyYAML and NumPy
and the artifacts in `ba-ner/results/`; overwrites `thesis/generated/`), and
test the aggregation:

```bash
python3 thesis/scripts/aggregate_results.py
python3 -m unittest discover -s thesis/scripts -p 'test_*.py'
```

Set up and test the experiment code:

```bash
cd ba-ner
pip install -r requirements.txt
python -m pytest
```

Experiment entry points (submitting requires SLURM and a clean, committed
checkout; see `ba-ner/docs/cluster-jobs.md`):

| Study | Entry point |
| --- | --- |
| Main study (5 fine-tuned models × 3 seeds) | `python scripts/run_seed_matrix.py --preflight \| --dry-run \| --submit \| --status \| --aggregate-only` |
| CrossNER transfer (256 / 1,024 output tokens) | `python scripts/run_crossner.py [--max-new-tokens 1024] --preflight \| --dry-run \| --submit \| --status \| --aggregate-only` |
| MultiNERD re-generation of truncated answers | `python -m src.evaluate.multinerd_output_limit --preflight \| --aggregate` and `scripts/cluster/submit_multinerd_output_limit.sh` |
| Control sample with 1,024 tokens | `scripts/cluster/submit_multinerd_control_sample.sh` |

All paths are relative to `ba-ner/`. The previous, detailed README with the
full option lists is available with `git show aaaca2b:README.md`.

## License

MIT License (see `LICENSE`). Datasets and models are subject to their own
licenses (MultiNERD, CrossNER, DeBERTa-v3, Qwen3.5).
