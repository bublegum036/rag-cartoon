"""Canonical, provider-independent identities for immutable catalog snapshots."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from cartoon_rag.domain import Film
from cartoon_rag.ingestion import active_snapshot


def fingerprint(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def catalog_fingerprint(films: Sequence[Film]) -> str:
    cards = sorted((asdict(film) for film in active_snapshot(films)), key=lambda row: row["movie_id"])
    return fingerprint(cards)
