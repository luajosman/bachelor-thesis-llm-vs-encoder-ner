"""Pinned, test-only CrossNER loading and the shared PER/ORG/LOC task."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

CROSSNER_REVISION = "2e7ba2a7798c961e3f29fbc51252c5a8d40224bf"
DOMAINS = ("ai", "literature", "music", "politics", "science")
TEST_SIZES = dict(zip(DOMAINS, (431, 416, 465, 651, 543)))
TEST_SHA256 = dict(zip(DOMAINS, (
    "7595ad386e0a98067100fe47f86811b38867a236011fee97cfaa90f778c37bcf",
    "77267f8b3e8afe993865c64cfc605f65671f6a0ac9cba65de970f12180fed20e",
    "c66e345f959ac65593f76c6b1bd8c6d75f2a6f0bc197389b2e6700d101fd094c",
    "05cadc5e129d394f7f92b4a052cf4b6ea4c60878d32a833e500cf046c297455f",
    "3616d59d4d2b9dbf4c1126f8d8ba47ff97886223c95f93649bfd527343e64ced",
)))
COMMON_TYPES = ("PER", "ORG", "LOC")
TYPE_MAPPING = {
    **dict.fromkeys(("person", "politician", "scientist", "writer", "musicalartist", "researcher"), "PER"),
    **dict.fromkeys(("organisation", "politicalparty", "university", "band"), "ORG"),
    **dict.fromkeys(("location", "country"), "LOC"),
}
DOMAIN_TYPES = {
    "ai": {"field", "task", "product", "algorithm", "researcher", "metrics", "programlang", "conference", "university", "country", "person", "organisation", "location", "misc"},
    "literature": {"book", "writer", "award", "poem", "event", "magazine", "literarygenre", "country", "person", "location", "organisation", "misc"},
    "music": {"musicgenre", "song", "band", "album", "musicalartist", "musicalinstrument", "award", "event", "country", "location", "organisation", "person", "misc"},
    "politics": {"country", "politician", "election", "person", "organisation", "location", "misc", "politicalparty", "event"},
    "science": {"scientist", "person", "university", "organisation", "country", "location", "discipline", "enzyme", "protein", "chemicalelement", "chemicalcompound", "astronomicalobject", "academicjournal", "event", "theory", "award", "misc"},
}


def project_multinerd_bio(tags: list[str]) -> list[str]:
    """Keep the three scored types without repairing model BIO predictions."""
    from src.data.dataset_loader import MULTINERD_LABEL_LIST

    allowed = set(MULTINERD_LABEL_LIST)
    if any(tag not in allowed for tag in tags):
        raise ValueError("Prediction contains an unknown MultiNERD BIO label")
    return [tag if tag == "O" or tag[2:] in COMMON_TYPES else "O" for tag in tags]


def parse_crossner(text: str, domain: str) -> list[dict]:
    """Parse strict native IOB2, retaining sentence and entity boundaries."""
    if domain not in DOMAINS:
        raise ValueError(f"Unknown CrossNER domain: {domain}")
    records, tokens, tags = [], [], []

    def finish() -> None:
        if not tokens:
            return
        gold = [tag[:2] + TYPE_MAPPING[tag[2:]] if tag != "O" and tag[2:] in TYPE_MAPPING else "O" for tag in tags]
        records.append({"sample_id": f"{domain}:test:{len(records)}", "tokens": tokens.copy(), "native_bio": tags.copy(), "gold_bio": gold})
        tokens.clear()
        tags.clear()

    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            finish()
            continue
        fields = line.split("\t")
        if len(fields) != 2 or not fields[0].strip():
            raise ValueError(f"{domain}:{line_no}: expected token<TAB>BIO label")
        token, tag = fields
        if tag != "O":
            if tag[:2] not in ("B-", "I-") or tag[2:] not in DOMAIN_TYPES[domain]:
                raise ValueError(f"{domain}:{line_no}: unknown BIO label {tag!r}")
            if tag.startswith("I-") and (not tags or tags[-1] not in ("B-" + tag[2:], tag)):
                raise ValueError(f"{domain}:{line_no}: invalid IOB2 continuation {tag!r}")
        tokens.append(token)
        tags.append(tag)
    finish()
    if not records:
        raise ValueError(f"{domain}: empty test split")
    return records


def load_crossner_test(domain: str, cache_dir: str | Path, *, download: bool = False) -> list[dict]:
    """Load only test.txt; never execute a remote dataset loading script."""
    if domain not in DOMAINS:
        raise ValueError(f"Unknown CrossNER domain: {domain}")
    path = Path(cache_dir) / CROSSNER_REVISION / domain / "test.txt"
    if not path.is_file():
        if not download:
            raise FileNotFoundError(f"Missing CrossNER cache: {path}; run --preflight first")
        url = f"https://raw.githubusercontent.com/zliucr/CrossNER/{CROSSNER_REVISION}/ner_data/{domain}/test.txt"
        with urlopen(url, timeout=120) as response:
            content = response.read()
        _verify(content, domain)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    content = path.read_bytes()
    _verify(content, domain)
    records = parse_crossner(content.decode("utf-8"), domain)
    if len(records) != TEST_SIZES[domain]:
        raise ValueError(f"{domain}: expected {TEST_SIZES[domain]} test examples, got {len(records)}")
    return records


def _verify(content: bytes, domain: str) -> None:
    if hashlib.sha256(content).hexdigest() != TEST_SHA256[domain]:
        raise ValueError(f"{domain}: CrossNER SHA-256 mismatch; refusing altered data")
