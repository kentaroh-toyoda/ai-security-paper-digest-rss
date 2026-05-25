"""Scholar list loader and author matcher.

Loads `config/scholars.yaml` (vendored from the key-scholars repo) and matches
paper author lists against the tracked scholars' name variants. Matching is
case-insensitive and accent-folded (e.g. "Florian Tramer" matches "Florian Tramèr").
"""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from typing import Iterable

import yaml


DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "scholars.yaml",
)


@dataclass(frozen=True)
class Scholar:
    name: str
    affiliation: str
    sub_area: str
    arxiv_authors: tuple[str, ...]


def _normalize(s: str) -> str:
    """Lowercase, NFKD accent-fold, collapse whitespace."""
    folded = unicodedata.normalize("NFKD", s)
    stripped = "".join(c for c in folded if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())


def load_scholars(path: str = DEFAULT_PATH) -> list[Scholar]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    out: list[Scholar] = []
    for entry in data.get("scholars", []):
        out.append(
            Scholar(
                name=entry["name"],
                affiliation=entry["affiliation"],
                sub_area=entry["sub_area"],
                arxiv_authors=tuple(entry["arxiv_authors"]),
            )
        )
    return out


def build_index(scholars: Iterable[Scholar]) -> dict[str, list[Scholar]]:
    """Map normalized name variant -> list of scholars sharing that variant.

    A list is returned (not a single scholar) because common names like
    "Bo Li" or "Yu Su" can map to multiple tracked entries in principle and
    must always be passed through the affiliation verifier.
    """
    index: dict[str, list[Scholar]] = {}
    for s in scholars:
        for variant in s.arxiv_authors:
            key = _normalize(variant)
            index.setdefault(key, []).append(s)
    return index


def match_authors(
    paper_authors: Iterable[str], index: dict[str, list[Scholar]]
) -> list[Scholar]:
    """Return tracked scholars whose name variants appear in paper_authors.

    Deduplicated by scholar name, order preserved.
    """
    seen: set[str] = set()
    matched: list[Scholar] = []
    for author in paper_authors:
        key = _normalize(author)
        if not key:
            continue
        for scholar in index.get(key, []):
            if scholar.name in seen:
                continue
            seen.add(scholar.name)
            matched.append(scholar)
    return matched
