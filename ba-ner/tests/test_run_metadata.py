from __future__ import annotations

from src.run_metadata import TRACKED_PACKAGES, collect_run_metadata


def test_collect_run_metadata_is_best_effort_and_serializable() -> None:
    metadata = collect_run_metadata({
        "model_revision": "model-rev",
        "dataset_revision": "dataset-rev",
    })

    assert isinstance(metadata, dict)
    assert set(("git", "python", "platform", "hostname", "packages", "cuda")).issubset(metadata)
    assert metadata["revisions"] == {
        "model_revision": "model-rev",
        "dataset_revision": "dataset-rev",
    }
    assert set(TRACKED_PACKAGES).issubset(metadata["packages"])
    for package_info in metadata["packages"].values():
        assert set(package_info) == {"installed", "version"}
        assert isinstance(package_info["installed"], bool)
