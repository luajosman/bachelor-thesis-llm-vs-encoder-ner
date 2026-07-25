"""
MultiNERD English dataset loader for the final NER experiments.

The repository intentionally supports only:
  - Hugging Face dataset: Babelscape/multinerd
  - Language subset: English (lang == "en")
  - Label schema: 15 MultiNERD entity types, 31 BIO labels including "O"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from rich.console import Console

from src.config import DATASET_LANGUAGE, DATASET_NAME

console = Console()

MULTINERD_HF_NAME = "Babelscape/multinerd"
MULTINERD_ENGLISH_SPLIT_SIZES = {
    "train": 131_280,
    "validation": 16_410,
    "test": 16_454,
}

MULTINERD_LABEL_LIST: List[str] = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-ANIM", "I-ANIM",
    "B-BIO", "I-BIO",
    "B-CEL", "I-CEL",
    "B-DIS", "I-DIS",
    "B-EVE", "I-EVE",
    "B-FOOD", "I-FOOD",
    "B-INST", "I-INST",
    "B-MEDIA", "I-MEDIA",
    "B-MYTH", "I-MYTH",
    "B-PLANT", "I-PLANT",
    "B-TIME", "I-TIME",
    "B-VEHI", "I-VEHI",
]


@dataclass
class DatasetInfo:
    """Static metadata for the MultiNERD English NER task."""

    name: str
    hf_name: str
    label_list: List[str]
    language: str = DATASET_LANGUAGE
    id2label: Dict[int, str] = field(default_factory=dict)
    label2id: Dict[str, int] = field(default_factory=dict)
    entity_types: List[str] = field(default_factory=list)
    num_labels: int = 0

    def __post_init__(self) -> None:
        if not self.id2label:
            self.id2label = {i: label for i, label in enumerate(self.label_list)}
        if not self.label2id:
            self.label2id = {label: i for i, label in enumerate(self.label_list)}
        if not self.entity_types:
            self.entity_types = [label[2:] for label in self.label_list if label.startswith("B-")]
        if self.num_labels == 0:
            self.num_labels = len(self.label_list)


MULTINERD_INFO = DatasetInfo(
    name=DATASET_NAME,
    hf_name=MULTINERD_HF_NAME,
    label_list=MULTINERD_LABEL_LIST,
)


def get_dataset_info(dataset_name: str = DATASET_NAME) -> DatasetInfo:
    """Return static metadata for MultiNERD English."""
    _ensure_multinerd(dataset_name)
    return MULTINERD_INFO


def load_ner_dataset(
    dataset_name: str = DATASET_NAME,
    language: str = DATASET_LANGUAGE,
) -> Tuple[Any, DatasetInfo]:
    """Load MultiNERD, filter the English subset, and validate the schema."""
    _ensure_multinerd(dataset_name)
    _ensure_english(language)

    from datasets import DatasetDict, load_dataset

    console.print(f"[cyan]Loading dataset: {MULTINERD_HF_NAME}[/cyan]")
    # The upstream Hub metadata expects every source row twice, while the
    # repository files match the 164.1K English sentences reported in the paper.
    raw: DatasetDict = load_dataset(
        MULTINERD_HF_NAME,
        verification_mode="no_checks",
    )
    _validate_raw_schema(raw)

    console.print("[cyan]Filtering language: en[/cyan]")
    filtered = raw.filter(lambda x: x["lang"] == DATASET_LANGUAGE)
    if "lang" in filtered["train"].column_names:
        filtered = filtered.remove_columns(["lang"])

    _validate_processed_schema(filtered)
    _validate_english_split_sizes(filtered)

    for split_name in ("train", "validation", "test"):
        console.print(f"  {split_name}: {len(filtered[split_name]):,} sentences")

    return filtered, MULTINERD_INFO


def _ensure_multinerd(dataset_name: str) -> None:
    if dataset_name != DATASET_NAME:
        raise ValueError(f"Unsupported dataset {dataset_name!r}; only {DATASET_NAME!r} is supported.")


def _ensure_english(language: str) -> None:
    if language != DATASET_LANGUAGE:
        raise ValueError(f"Unsupported language {language!r}; only {DATASET_LANGUAGE!r} is supported.")


def _validate_raw_schema(raw: Any) -> None:
    expected_splits = {"train", "validation", "test"}
    actual_splits = set(raw.keys())
    if not expected_splits.issubset(actual_splits):
        raise ValueError(f"MultiNERD is missing required splits: {expected_splits - actual_splits}")

    train_columns = set(raw["train"].column_names)
    expected_columns = {"tokens", "ner_tags", "lang"}
    if not expected_columns.issubset(train_columns):
        raise ValueError(f"MultiNERD train split has columns {train_columns}; expected {expected_columns}")

    labels = _extract_label_names(raw["train"].features.get("ner_tags"))
    if labels is not None and labels != MULTINERD_LABEL_LIST:
        raise ValueError(
            "MultiNERD label schema does not match the expected final label list. "
            f"Expected {MULTINERD_LABEL_LIST}, got {labels}."
        )


def _validate_processed_schema(dataset: Any) -> None:
    expected_columns = {"tokens", "ner_tags"}
    for split_name in ("train", "validation", "test"):
        columns = set(dataset[split_name].column_names)
        if not expected_columns.issubset(columns):
            raise ValueError(f"{split_name} split has columns {columns}; expected {expected_columns}")
        if "lang" in columns:
            raise ValueError(f"{split_name} split still contains a lang column after English filtering")


def _validate_english_split_sizes(dataset: Any) -> None:
    actual = {split: len(dataset[split]) for split in MULTINERD_ENGLISH_SPLIT_SIZES}
    if actual != MULTINERD_ENGLISH_SPLIT_SIZES:
        raise ValueError(
            "MultiNERD English split sizes do not match the source release. "
            f"Expected {MULTINERD_ENGLISH_SPLIT_SIZES}, got {actual}."
        )


def _extract_label_names(feature: Any) -> List[str] | None:
    if feature is None:
        return None
    nested = getattr(feature, "feature", None)
    names = getattr(nested, "names", None)
    return list(names) if names else None
