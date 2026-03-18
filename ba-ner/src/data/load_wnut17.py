"""
load_wnut17.py — WNUT-2017 Datensatz laden und inspizieren

WNUT-2017 (Workshop on Noisy User-generated Text) ist der Datensatz dieser
Bachelorarbeit. Er enthält Social-Media-Texte (Twitter, Reddit) mit 6 Entity-Typen:
person, location, corporation, creative-work, group, product.

Der Datensatz ist deutlich schwieriger als CoNLL-2003, weil Entities oft neu,
selten und orthographisch uneinheitlich sind ("emerging entities").

Verwendung:
    python -m src.data.load_wnut17
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

from datasets import DatasetDict, load_dataset
from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Label-Definitionen für WNUT-2017
# ---------------------------------------------------------------------------

# Vollständige BIO-Label-Liste in der Reihenfolge des HuggingFace-Datensatzes.
# "O" = kein Entity (Outside), "B-" = Beginn einer Entity, "I-" = Fortsetzung.
# Diese Liste legt num_labels für den Klassifikationskopf der Encoder-Modelle fest.
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

# Integer-ID → Label-String (für die Dekodierung von Modellvorhersagen)
ID2LABEL: Dict[int, str] = {i: label for i, label in enumerate(LABEL_LIST)}

# Label-String → Integer-ID (wird beim Laden des Modells übergeben)
LABEL2ID: Dict[str, int] = {label: i for i, label in enumerate(LABEL_LIST)}

# Die 6 reinen Entity-Typen ohne BIO-Präfix (für Statistiken und Visualisierungen)
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
# Datensatz laden
# ---------------------------------------------------------------------------

def load_wnut17() -> DatasetDict:
    """Lädt WNUT-2017 direkt von HuggingFace Datasets.

    Beim ersten Aufruf wird der Datensatz heruntergeladen und danach
    automatisch aus dem lokalen Cache geladen.

    Returns:
        DatasetDict mit 'train', 'validation' und 'test'.
        Jede Zeile enthält 'id', 'tokens' (Wortliste) und
        'ner_tags' (Integer-Labels gemäß LABEL_LIST).
    """
    dataset: DatasetDict = load_dataset("wnut_17")
    return dataset


# ---------------------------------------------------------------------------
# Interne Hilfsfunktionen
# ---------------------------------------------------------------------------

def _extract_entities(tokens: List[str], tags: List[int]) -> List[Dict[str, str]]:
    """Konvertiert BIO-Integer-Tags in eine Liste von Entity-Dicts.

    Logik: Ein "B-"-Tag beginnt eine neue Entity. "I-"-Tags setzen sie fort.
    Ein "O"-Tag oder ein neues "B-"-Tag beendet die laufende Entity.
    Am Satzende wird die letzte offene Entity noch abgeschlossen.

    Args:
        tokens: Wortliste eines Satzes.
        tags:   Integer-BIO-Tags (Index in LABEL_LIST).

    Returns:
        Liste von Dicts mit Schlüsseln 'entity' (Oberflächentext) und 'type'.
    """
    entities: List[Dict[str, str]] = []
    current_tokens: List[str] = []
    current_type: str | None = None

    for token, tag in zip(tokens, tags):
        label = ID2LABEL[tag]

        if label.startswith("B-"):
            # Vorherige Entity abschließen, falls noch offen
            if current_type is not None:
                entities.append({"entity": " ".join(current_tokens), "type": current_type})
            # Neue Entity beginnen: "B-person" → type = "person"
            current_type = label[2:]
            current_tokens = [token]

        elif label.startswith("I-") and current_type is not None:
            # Laufende Entity um das nächste Token erweitern
            current_tokens.append(token)

        else:
            # "O"-Tag: offene Entity schließen und Zustand zurücksetzen
            if current_type is not None:
                entities.append({"entity": " ".join(current_tokens), "type": current_type})
            current_type = None
            current_tokens = []

    # Letzte Entity am Satzende abschließen (falls kein abschließendes "O")
    if current_type is not None:
        entities.append({"entity": " ".join(current_tokens), "type": current_type})

    return entities


def _count_entities(split) -> Counter:
    """Zählt die Häufigkeit jedes Entity-Typs in einem Datensatz-Split.

    Args:
        split: Ein einzelner HuggingFace-Split (z.B. dataset['train']).

    Returns:
        Counter mit Entity-Typ als Key und Anzahl als Value.
    """
    counts: Counter = Counter()
    for sample in split:
        for ent in _extract_entities(sample["tokens"], sample["ner_tags"]):
            counts[ent["type"]] += 1
    return counts


# ---------------------------------------------------------------------------
# Statistiken ausgeben
# ---------------------------------------------------------------------------

def print_stats(dataset: DatasetDict) -> None:
    """Gibt Datensatz-Statistiken als formatierte rich-Tabellen aus.

    Tabelle 1: Sätze, Tokens und Entity-Anzahl pro Split.
    Tabelle 2: Entity-Typ-Verteilung im Train-Split.

    Args:
        dataset: Das WNUT-2017 DatasetDict.
    """
    # --- Tabelle 1: Überblick über die drei Splits ---
    split_table = Table(
        title="WNUT-2017 Datensatz-Statistiken",
        show_header=True,
        header_style="bold cyan",
    )
    split_table.add_column("Split", style="bold")
    split_table.add_column("Sätze", justify="right")
    split_table.add_column("Tokens", justify="right")
    split_table.add_column("Entities", justify="right")

    for split_name in ["train", "validation", "test"]:
        split = dataset[split_name]
        n_sentences = len(split)
        n_tokens = sum(len(s["tokens"]) for s in split)
        n_entities = sum(_count_entities(split).values())
        split_table.add_row(split_name, str(n_sentences), str(n_tokens), str(n_entities))

    console.print(split_table)

    # --- Tabelle 2: Entity-Typ-Verteilung im Train-Split ---
    train_counts = _count_entities(dataset["train"])
    entity_table = Table(
        title="Entity-Typ-Verteilung (Train-Split)",
        show_header=True,
        header_style="bold magenta",
    )
    entity_table.add_column("Entity-Typ", style="bold")
    entity_table.add_column("Anzahl", justify="right")
    entity_table.add_column("Anteil", justify="right")

    total = sum(train_counts.values())
    for etype in ENTITY_TYPES:
        count = train_counts.get(etype, 0)
        # Prozentualen Anteil berechnen; Sonderfall total == 0 absichern
        share = f"{count / total * 100:.1f}%" if total > 0 else "0.0%"
        entity_table.add_row(etype, str(count), share)

    entity_table.add_row("[bold]Gesamt[/bold]", str(total), "100.0%")
    console.print(entity_table)


# ---------------------------------------------------------------------------
# Beispielsätze anzeigen
# ---------------------------------------------------------------------------

def show_examples(dataset: DatasetDict, n: int = 3) -> None:
    """Gibt n Beispielsätze aus dem Train-Split mit ihren Entities aus.

    Dient zur schnellen Überprüfung, ob Datensatz und BIO-Extraktion
    korrekt funktionieren.

    Args:
        dataset: Das WNUT-2017 DatasetDict.
        n:       Anzahl der anzuzeigenden Beispiele.
    """
    console.print(f"\n[bold cyan]--- {n} Beispielsätze (Train-Split) ---[/bold cyan]")

    for i, sample in enumerate(dataset["train"]):
        if i >= n:
            break

        tokens: List[str] = sample["tokens"]
        tags: List[int] = sample["ner_tags"]
        entities = _extract_entities(tokens, tags)

        console.print(f"\n[bold]Beispiel {i + 1}:[/bold]")
        console.print(f"  Text     : {' '.join(tokens)}")

        if entities:
            console.print("  Entities :")
            for ent in entities:
                console.print(f"    [{ent['type']}]  \"{ent['entity']}\"")
        else:
            console.print("  Entities : (keine)")


# ---------------------------------------------------------------------------
# Direktaufruf: Datensatz laden und Überblick ausgeben
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    console.print("[bold green]Lade WNUT-2017...[/bold green]")
    dataset = load_wnut17()
    print_stats(dataset)
    show_examples(dataset, n=3)
    console.print("\n[bold green]Fertig.[/bold green]")
