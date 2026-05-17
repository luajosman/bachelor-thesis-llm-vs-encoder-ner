from __future__ import annotations

import pytest

from src.config import DATASET_LANGUAGE, DATASET_NAME
from src.data.dataset_loader import MULTINERD_LABEL_LIST, get_dataset_info


def test_multinerd_english_static_metadata() -> None:
    info = get_dataset_info()
    assert info.name == DATASET_NAME
    assert info.language == DATASET_LANGUAGE
    assert info.num_labels == 31
    assert info.label_list == MULTINERD_LABEL_LIST
    assert info.label_list[0] == "O"
    assert info.entity_types == [
        "PER",
        "ORG",
        "LOC",
        "ANIM",
        "BIO",
        "CEL",
        "DIS",
        "EVE",
        "FOOD",
        "INST",
        "MEDIA",
        "MYTH",
        "PLANT",
        "TIME",
        "VEHI",
    ]


def test_non_multinerd_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="only 'multinerd' is supported"):
        get_dataset_info("other_dataset")
