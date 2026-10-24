"""Catalog loading, revision selection and provider-independent indexing."""

import json
from pathlib import Path
from typing import Sequence

from cartoon_rag.domain import Chunk, Film
from cartoon_rag.ports import Chunker, Embedder, IndexWriter


def load_catalog(path: Path) -> tuple[Film, ...]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return tuple(Film(**{**row, "tags": tuple(row.get("tags", ()))}) for row in records)


def active_snapshot(films: Sequence[Film]) -> tuple[Film, ...]:
    latest: dict[str, Film] = {}
    seen: set[tuple[str, int]] = set()
    for film in films:
        key = (film.movie_id, film.version)
        if key in seen:
            raise ValueError(f"Duplicate movie revision: {key}")
        seen.add(key)
        if film.movie_id not in latest or film.version > latest[film.movie_id].version:
            latest[film.movie_id] = film
    # An inactive latest revision removes a movie; do not resurrect older cards.
    result = tuple(film for film in latest.values() if film.active)
    if len({film.source_id for film in result}) != len(result):
        raise ValueError("Active card source IDs must be unique")
    return result


class WholeSynopsis:
    """Baseline for this short catalog; replace with a semantic chunker if needed."""

    def split(self, text: str) -> tuple[str, ...]:
        return (text,)


def index_catalog(
    films: Sequence[Film], chunker: Chunker, embedder: Embedder, writer: IndexWriter,
) -> int:
    chunks: list[Chunk] = []
    for film in active_snapshot(films):
        parts = tuple(chunker.split(film.synopsis))
        if not parts or any(not part.strip() for part in parts):
            raise ValueError(f"Empty chunks for {film.movie_id}")
        for number, part in enumerate(parts, start=1):
            chunk_id = film.source_id if len(parts) == 1 else f"{film.source_id}:part-{number}"
            chunks.append(Chunk(chunk_id, film, part))
    vectors = tuple(embedder.embed([chunk.embedding_text for chunk in chunks])) if chunks else ()
    if len(vectors) != len(chunks):
        raise ValueError("Embedder returned an incorrect number of vectors")
    writer.replace_snapshot(chunks, vectors)
    return len(chunks)
