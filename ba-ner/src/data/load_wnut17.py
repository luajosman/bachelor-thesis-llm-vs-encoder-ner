"""
WNUT-2017 dataset loader with statistics and example display.

Usage:
    python -m src.data.load_wnut17
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

from datasets import DatasetDict, load_dataset
from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------

LABEL_LIST: List[str] = [
    "O",
    "B-corporation",
    "I-corporation",
    "B-creative-work",
    "I-creative-work",
    "B-group",
    "I-group",
    "B-location",
    "I-location",
    "B-person",
    "I-person",
    "B-product",
    "I-product",
]

ID2LABEL: Dict[int, str] = {i: label for i, label in enumerate(LABEL_LIST)}
LABEL2ID: Dict[str, int] = {label: i for i, label in enumerate(LABEL_LIST)}

ENTITY_TYPES: List[str] = [
    "corporation",
    "creative-work",
    "group",
    "location",
    "person",
    "product",
]

console = Console()


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------


def load_wnut17() -> DatasetDict:
    """Load WNUT-2017 from HuggingFace Hub.

    Returns
    -------
    DatasetDict
        Dataset with splits: train, validation, test.
    """
    dataset: DatasetDict = load_dataset("wnut_17")
    return dataset


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _count_entities(split) -> Counter:
    """Count entity types in a dataset split."""
    counts: Counter = Counter()
    for sample in split:
        tokens: List[str] = sample["tokens"]
        tags: List[int] = sample["ner_tags"]
        in_entity = False
        current_type = None
        for token, tag in zip(tokens, tags):
            label = ID2LABEL[tag]
            if label.startswith("B-"):
                current_type = label[2:]
                counts[current_type] += 1
                in_entity = True
            elif label.startswith("I-") and in_entity:
                pass  # continuation
            else:
                in_entity = False
                current_type = None
    return counts


def print_stats(dataset: DatasetDict) -> None:
    """Print dataset split sizes and entity distribution using rich tables.

    Parameters
    ----------
    dataset:
        The WNUT-2017 DatasetDict.
    """
    # ---- Split overview ----
    split_table = Table(title="WNUT-2017 Dataset Statistics", show_header=True, header_style="bold cyan")
    split_table.add_column("Split", style="bold")
    split_table.add_column("Sentences", justify="right")
    split_table.add_column("Tokens", justify="right")
    split_table.add_column("Entities", justify="right")

    for split_name in ["train", "validation", "test"]:
        split = dataset[split_name]
        n_sentences = len(split)
        n_tokens = sum(len(s["tokens"]) for s in split)
        entity_counts = _count_entities(split)
        n_entities = sum(entity_counts.values())
        split_table.add_row(split_name, str(n_sentences), str(n_tokens), str(n_entities))

    console.print(split_table)

    # ---- Entity type distribution (train split) ----
    train_counts = _count_entities(dataset["train"])
    entity_table = Table(title="Entity Type Distribution (Train)", show_header=True, header_style="bold magenta")
    entity_table.add_column("Entity Type", style="bold")
    entity_table.add_column("Count", justify="right")
    entity_table.add_column("Share", justify="right")

    total = sum(train_counts.values())
    for etype in ENTITY_TYPES:
        count = train_counts.get(etype, 0)
        share = f"{count / total * 100:.1f}%" if total > 0 else "0.0%"
        entity_table.add_row(etype, str(count), share)
    entity_table.add_row("[bold]Total[/bold]", str(total), "100.0%")

    console.print(entity_table)


def show_examples(dataset: DatasetDict, n: int = 3) -> None:
    """Print n example sentences from the train split with their entities.

    Parameters
    ----------
    dataset:
        The WNUT-2017 DatasetDict.
    n:
        Number of examples to show.
    """
    console.print(f"\n[bold cyan]--- {n} Example Sentences (Train) ---[/bold cyan]")
    for i, sample in enumerate(dataset["train"]):
        if i >= n:
            break
        tokens: List[str] = sample["tokens"]
        tags: List[int] = sample["ner_tags"]

        # Extract entities from BIO tags
        entities: List[Dict[str, str]] = _extract_entities(tokens, tags)

        sentence = " ".join(tokens)
        console.print(f"\n[bold]Example {i + 1}:[/bold]")
        console.print(f"  Text : {sentence}")
        if entities:
            console.print(f"  Entities:")
            for ent in entities:
                console.print(f"    - [{ent['type']}] \"{ent['entity']}\"")
        else:
            console.print("  Entities: (none)")


def _extract_entities(tokens: List[str], tags: List[int]) -> List[Dict[str, str]]:
    """Extract entity dicts from token list and integer BIO tags.

    Parameters
    ----------
    tokens:
        List of word tokens.
    tags:
        Corresponding BIO integer tags (using LABEL_LIST indices).

    Returns
    -------
    List[Dict[str, str]]
        Each dict has keys ``entity`` (surface string) and ``type``.
    """
    entities: List[Dict[str, str]] = []
    current_tokens: List[str] = []
    current_type: str | None = None

    for token, tag in zip(tokens, tags):
        label = ID2LABEL[tag]
        if label.startswith("B-"):
            if current_type is not None:
                entities.append({"entity": " ".join(current_tokens), "type": current_type})
            current_type = label[2:]
            current_tokens = [token]
        elif label.startswith("I-") and current_type is not None:
            current_tokens.append(token)
        else:
            if current_type is not None:
                entities.append({"entity": " ".join(current_tokens), "type": current_type})
            current_type = None
            current_tokens = []

    if current_type is not None:
        entities.append({"entity": " ".join(current_tokens), "type": current_type})

    return entities


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    console.print("[bold green]Loading WNUT-2017...[/bold green]")
    dataset = load_wnut17()
    print_stats(dataset)
    show_examples(dataset, n=3)
    console.print("\n[bold green]Done.[/bold green]")
