"""Token lengths of the JSON target answers of MultiNERD (English) under the shared Qwen3.5
tokenizer: how many sentences have a reference answer longer than 256 / 512 / 1,024 tokens
(thesis Section 4.4.3, justification of the 256-token output limit). Read-only, CPU only;
needs the cached dataset and tokenizer (run on the cluster with HF offline variables set).

    python scripts/answer_lengths.py

Result on 28.10.2026 (revision-pinned data and tokenizer):
  train      n=131280 mean=47.2 p50=32 p90=87 p99=200 max=1047 >256: 576  >512: 49  >1024: 1
  validation n=16410  mean=48.7 p50=32 p90=89 p99=217 max=951  >256: 103  >512: 11  >1024: 0
  test       n=16454  mean=47.7 p50=32 p90=88 p99=186 max=553  >256: 55   >512: 1   >1024: 0
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess_decoder import prepare_decoder_dataset  # noqa: E402
from src.seed_provenance import MODEL_REVISIONS  # noqa: E402

MODEL = "Qwen/Qwen3.5-0.8B"  # all three sizes share the tokenizer
LIMITS = (128, 200, 256, 512, 1024)


def main() -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=MODEL_REVISIONS[MODEL], trust_remote_code=True, local_files_only=True
    )
    formatted, _ = prepare_decoder_dataset()
    report = {}
    for split in ("train", "validation", "test"):
        if split not in formatted:
            continue
        lengths = []
        for example in formatted[split]:
            answers = [m for m in example["messages"] if m["role"] == "assistant"]
            if answers:
                lengths.append(len(tokenizer(answers[-1]["content"], add_special_tokens=False)["input_ids"]))
        lengths.sort()
        n = len(lengths)

        def quantile(p: float) -> int:
            return lengths[min(n - 1, int(p * n))]

        report[split] = {
            "n": n, "mean": statistics.fmean(lengths), "p50": quantile(0.5), "p90": quantile(0.9),
            "p99": quantile(0.99), "max": lengths[-1],
            **{f"gt{limit}": sum(length > limit for length in lengths) for limit in LIMITS},
        }
        print(split, json.dumps(report[split]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
