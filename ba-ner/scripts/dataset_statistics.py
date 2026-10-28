"""Recompute the MultiNERD split statistics (thesis Table 4.3) and the number of
SFT examples whose rendered chat prompt exceeds the 1,024-token training limit
(thesis Section 4.4.3). Read-only; needs the cached dataset and the pinned
Qwen3.5 tokenizer (run on the cluster with HF offline variables set).

    python scripts/dataset_statistics.py [--skip-truncation]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_loader import load_ner_dataset  # noqa: E402

TRAIN_MAX_LENGTH = 1024


def split_statistics(raw, info) -> dict:
    stats = {}
    for split in ("train", "validation", "test"):
        tokens = 0
        entities: collections.Counter[str] = collections.Counter()
        for example in raw[split]:
            tokens += len(example["tokens"])
            for tag in example["ner_tags"]:
                label = info.id2label[tag]
                if label.startswith("B-"):
                    entities[label[2:]] += 1
        stats[split] = {
            "sentences": len(raw[split]),
            "tokens": tokens,
            "entities": sum(entities.values()),
            "per_type": dict(sorted(entities.items())),
        }
    return stats


def truncation_counts(max_length: int = TRAIN_MAX_LENGTH) -> dict:
    from transformers import AutoTokenizer

    from src.data.preprocess_decoder import prepare_decoder_dataset
    from src.seed_provenance import MODEL_REVISIONS

    model = "Qwen/Qwen3.5-0.8B"  # all three sizes share the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model, revision=MODEL_REVISIONS[model], trust_remote_code=True, local_files_only=True
    )
    formatted, _ = prepare_decoder_dataset()

    def length(messages) -> int:
        out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if hasattr(out, "keys"):
            out = out["input_ids"]
        return len(out)

    result = {}
    for split in ("train", "validation"):
        started = time.time()
        lengths = [length(example["messages"]) for example in formatted[split]]
        result[split] = {
            "examples": len(lengths),
            "over_limit": sum(1 for n in lengths if n > max_length),
            "max_length": max(lengths),
            "seconds": round(time.time() - started),
        }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-truncation", action="store_true")
    args = parser.parse_args(argv)
    raw, info = load_ner_dataset()
    report = {"splits": split_statistics(raw, info)}
    if not args.skip_truncation:
        report["truncation_at_%d" % TRAIN_MAX_LENGTH] = truncation_counts()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
