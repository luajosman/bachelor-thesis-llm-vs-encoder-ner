"""
dataset_loader.py — Einheitlicher Datensatz-Loader fuer NER-Experimente

Unterstuetzte Datensaetze:
  - multinerd:  Babelscape/multinerd (englische Teilmenge, 15 Entity-Typen)
  - wnut_17:    WNUT-2017 (Social-Media-Texte, 6 Entity-Typen)

Jeder Datensatz wird ueber eine DatasetInfo-Instanz beschrieben, die Label-Listen,
Mappings und Metadaten enthaelt. So koennen alle nachgelagerten Module
(Preprocessing, Training, Evaluation) datensatzunabhaengig arbeiten.

Verwendung:
    from src.data.dataset_loader import load_ner_dataset
    dataset, info = load_ner_dataset("multinerd")
    dataset, info = load_ner_dataset("wnut_17")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from datasets import DatasetDict, load_dataset
from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# DatasetInfo: Alles was nachgelagerte Module ueber einen Datensatz wissen muessen
# ---------------------------------------------------------------------------

@dataclass
class DatasetInfo:
    """Beschreibt einen NER-Datensatz vollstaendig.

    Wird von Preprocessing, Training und Evaluation genutzt, um
    datensatzunabhaengig zu arbeiten.

    Attributes:
        name:         Kurzname ("multinerd" oder "wnut_17").
        hf_name:      HuggingFace Dataset-ID.
        label_list:   Vollstaendige BIO-Label-Liste in korrekter Reihenfolge.
        id2label:     Integer-ID → Label-String.
        label2id:     Label-String → Integer-ID.
        entity_types: Reine Entity-Typen ohne BIO-Praefix.
        num_labels:   Anzahl der Labels (len(label_list)).
    """
    name:         str
    hf_name:      str
    label_list:   List[str]
    id2label:     Dict[int, str] = field(default_factory=dict)
    label2id:     Dict[str, int] = field(default_factory=dict)
    entity_types: List[str]      = field(default_factory=list)
    num_labels:   int            = 0

    def __post_init__(self):
        if not self.id2label:
            self.id2label = {i: l for i, l in enumerate(self.label_list)}
        if not self.label2id:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
        if not self.entity_types:
            # Entity-Typen aus B-Tags extrahieren (ohne "O")
            self.entity_types = [
                l[2:] for l in self.label_list if l.startswith("B-")
            ]
        if self.num_labels == 0:
            self.num_labels = len(self.label_list)


# ---------------------------------------------------------------------------
# Datensatz-Definitionen
# ---------------------------------------------------------------------------

# WNUT-2017: 6 Entity-Typen, 13 BIO-Labels
WNUT17_LABEL_LIST: List[str] = [
    "O",
    "B-corporation",  "I-corporation",
    "B-creative-work", "I-creative-work",
    "B-group",         "I-group",
    "B-location",      "I-location",
    "B-person",        "I-person",
    "B-product",       "I-product",
]

# MultiNERD: 15 Entity-Typen, 31 BIO-Labels
MULTINERD_LABEL_LIST: List[str] = [
    "O",
    "B-PER",   "I-PER",
    "B-ORG",   "I-ORG",
    "B-LOC",   "I-LOC",
    "B-ANIM",  "I-ANIM",
    "B-BIO",   "I-BIO",
    "B-CEL",   "I-CEL",
    "B-DIS",   "I-DIS",
    "B-EVE",   "I-EVE",
    "B-FOOD",  "I-FOOD",
    "B-INST",  "I-INST",
    "B-MEDIA", "I-MEDIA",
    "B-MYTH",  "I-MYTH",
    "B-PLANT", "I-PLANT",
    "B-TIME",  "I-TIME",
    "B-VEHI",  "I-VEHI",
]


def _build_info(name: str, hf_name: str, label_list: List[str]) -> DatasetInfo:
    """Baut eine DatasetInfo-Instanz aus Name und Label-Liste."""
    return DatasetInfo(name=name, hf_name=hf_name, label_list=label_list)


# Registry: Kurzname → DatasetInfo
_DATASET_REGISTRY: Dict[str, DatasetInfo] = {
    "wnut_17":   _build_info("wnut_17",   "wnut_17",            WNUT17_LABEL_LIST),
    "multinerd": _build_info("multinerd", "Babelscape/multinerd", MULTINERD_LABEL_LIST),
}


# ---------------------------------------------------------------------------
# Oeffentliche API
# ---------------------------------------------------------------------------

def get_dataset_info(dataset_name: str) -> DatasetInfo:
    """Gibt die DatasetInfo fuer einen Datensatz zurueck (ohne Daten zu laden).

    Args:
        dataset_name: "multinerd" oder "wnut_17".

    Returns:
        DatasetInfo mit allen Label-Mappings und Metadaten.

    Raises:
        ValueError: Wenn der Datensatzname unbekannt ist.
    """
    if dataset_name not in _DATASET_REGISTRY:
        available = ", ".join(_DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unbekannter Datensatz: '{dataset_name}'. "
            f"Verfuegbar: {available}"
        )
    return _DATASET_REGISTRY[dataset_name]


def load_ner_dataset(
    dataset_name: str,
    language: str = "en",
) -> Tuple[DatasetDict, DatasetInfo]:
    """Laedt einen NER-Datensatz und gibt ihn mit Metadaten zurueck.

    Fuer MultiNERD wird standardmaessig die englische Teilmenge gefiltert.
    WNUT-2017 hat keine Sprachfilterung (nur Englisch vorhanden).

    Args:
        dataset_name: "multinerd" oder "wnut_17".
        language:     Sprachfilter fuer MultiNERD (Standard: "en").

    Returns:
        Tuple (DatasetDict, DatasetInfo).
        DatasetDict enthaelt 'train', 'validation', 'test' mit
        Spalten 'tokens' (List[str]) und 'ner_tags' (List[int]).
    """
    info = get_dataset_info(dataset_name)

    console.print(f"[cyan]Lade Datensatz: {info.hf_name}...[/cyan]")
    raw: DatasetDict = load_dataset(info.hf_name)

    if dataset_name == "multinerd":
        # Englische Teilmenge filtern und Sprach-Spalte entfernen
        console.print(f"[cyan]Filtere Sprache: {language}...[/cyan]")
        raw = raw.filter(lambda x: x["lang"] == language)
        # 'lang'-Spalte ist nach dem Filtern nicht mehr noetig
        raw = raw.remove_columns(["lang"])

    # Sicherstellen, dass die erwarteten Spalten vorhanden sind
    expected_cols = {"tokens", "ner_tags"}
    actual_cols = set(raw["train"].column_names)
    if not expected_cols.issubset(actual_cols):
        raise ValueError(
            f"Datensatz {dataset_name} hat unerwartete Spalten: "
            f"{actual_cols}. Erwartet mindestens: {expected_cols}"
        )

    # Statistik ausgeben
    for split_name in ["train", "validation", "test"]:
        if split_name in raw:
            n = len(raw[split_name])
            console.print(f"  {split_name}: {n:,} Saetze")

    return raw, info
