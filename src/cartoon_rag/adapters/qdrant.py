"""Qdrant REST adapter: immutable collections, atomic alias publication."""

import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import NAMESPACE_URL, uuid4, uuid5

from cartoon_rag.domain import (
    Chunk, EmbeddingBatch, Film, Filters, GroundingError, Hit,
    PublicationResult, QueryPlan, RetrievalUnavailable, SnapshotMismatch,
)
from cartoon_rag.ports import QueryEmbedder
from cartoon_rag.snapshots import catalog_fingerprint, fingerprint


class Transport(Protocol):
    def request(self, method: str, path: str, body: dict | None = None) -> Any: ...


class QdrantHTTP:
    def __init__(self, url: str, *, api_key: str = "", timeout: float = 10.0) -> None:
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def request(self, method: str, path: str, body: dict | None = None) -> Any:
        data = json.dumps(body, allow_nan=False).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        request = Request(self.url + path, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
            return payload["result"]
        except HTTPError as exc:
            raise RetrievalUnavailable(f"Qdrant returned HTTP {exc.code} for {method} {path}") from exc
        except (URLError, OSError, TimeoutError, ValueError, KeyError) as exc:
            raise RetrievalUnavailable(f"Qdrant request failed: {method} {path}") from exc


def qdrant_filter(filters: Filters) -> dict:
    must: list[dict] = [{"key": "active", "match": {"value": True}}]
    bounds = {}
    if filters.duration_minutes_lt is not None:
        bounds["lt"] = filters.duration_minutes_lt
    if filters.duration_minutes_lte is not None:
        bounds["lte"] = filters.duration_minutes_lte
    if bounds:
        must.append({"key": "duration_minutes", "range": bounds})
    if filters.movie_ids:
        must.append({"key": "movie_id", "match": {"any": list(filters.movie_ids)}})
    return {"must": must}


class QdrantIndex:
    def __init__(self, transport: Transport, alias: str, embedder: QueryEmbedder) -> None:
        self.transport = transport
        self.alias = alias
        self.embedder = embedder
        self.dimensions = embedder.identity.dimensions

    def _collection_path(self, name: str) -> str:
        return f"/collections/{quote(name, safe='')}"

    def _vector(self, vector: Sequence[float]) -> list[float]:
        result = [float(value) for value in vector]
        if len(result) != self.dimensions or any(not math.isfinite(value) for value in result):
            raise ValueError("Invalid embedding dimension or non-finite vector")
        if not any(result):
            raise ValueError("Zero vector cannot be used for cosine search")
        return result

    def _resolve(self) -> str:
        aliases = self.transport.request("GET", "/aliases")["aliases"]
        for item in aliases:
            if item["alias_name"] == self.alias:
                return item["collection_name"]
        raise RetrievalUnavailable(f"Index alias {self.alias!r} is missing; run index first")

    def replace_snapshot(self, chunks: Sequence[Chunk], batch: EmbeddingBatch) -> PublicationResult:
        if batch.identity != self.embedder.identity:
            raise ValueError("Batch embedding identity differs from the index query model")
        vectors = batch.vectors
        if len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors must have equal lengths")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Duplicate chunk IDs")
        cards: dict[str, Film] = {}
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            film = chunk.film
            if not film.active or (film.movie_id in cards and cards[film.movie_id] != film):
                raise ValueError("Snapshot must contain one active revision per movie")
            cards[film.movie_id] = film
            payload = {**asdict(film), "chunk_id": chunk.chunk_id, "text": chunk.text}
            points.append({
                "id": str(uuid5(NAMESPACE_URL, f"{film.version}:{chunk.chunk_id}")),
                "vector": self._vector(vector), "payload": payload,
            })
        manifest = {
            "schema_version": 1, "embedding": asdict(batch.identity),
            "catalog_sha256": catalog_fingerprint(tuple(cards.values())),
            "chunks_sha256": fingerprint([point["payload"] for point in points]),
            "movies": len(cards), "chunks": len(chunks),
        }
        name = f"{self.alias}__{uuid4().hex}"
        path = self._collection_path(name)
        self.transport.request("PUT", path, {
            "vectors": {"size": self.dimensions, "distance": "Cosine"}, "metadata": manifest,
        })
        for field, schema in (("active", "bool"), ("duration_minutes", "integer"), ("movie_id", "keyword")):
            self.transport.request("PUT", path + "/index?wait=true", {
                "field_name": field, "field_schema": schema,
            })
        for offset in range(0, len(points), 64):
            self.transport.request("PUT", path + "/points?wait=true", {"points": points[offset:offset + 64]})
        count = self.transport.request("POST", path + "/points/count", {"exact": True})["count"]
        if count != len(points):
            raise RetrievalUnavailable("Snapshot point count differs from the prepared catalog")
        aliases = self.transport.request("GET", "/aliases")["aliases"]
        actions = []
        if any(item["alias_name"] == self.alias for item in aliases):
            actions.append({"delete_alias": {"alias_name": self.alias}})
        actions.append({"create_alias": {"alias_name": self.alias, "collection_name": name}})
        # Do not delete a collection after an alias request timeout: publication
        # may have succeeded. Old and incomplete snapshots are retained for inspection.
        self.transport.request("POST", "/collections/aliases", {"actions": actions})
        return PublicationResult(
            index_name=self.alias, snapshot_id=name, embedding=batch.identity,
            catalog_sha256=manifest["catalog_sha256"], chunks_sha256=manifest["chunks_sha256"],
            movies=len(cards), chunks=len(chunks),
        )

    def _validate_metadata(self, collection: str, expected_catalog_sha256: str | None = None) -> None:
        config = self.transport.request("GET", self._collection_path(collection))["config"]
        metadata = config.get("metadata", {})
        if metadata.get("embedding") != asdict(self.embedder.identity):
            raise ValueError("Index embedding identity differs from the current model; reindex")
        if expected_catalog_sha256 is not None and metadata.get("catalog_sha256") != expected_catalog_sha256:
            raise SnapshotMismatch("JSON catalog differs from the published index; reindex before asking")

    def bind_snapshot(self, expected_catalog_sha256: str) -> "PinnedQdrantRetriever":
        """Bind retrieval to the same immutable catalog used by exact operations."""
        if not re.fullmatch(r"[0-9a-f]{64}", expected_catalog_sha256):
            raise ValueError("Expected catalog fingerprint must be a SHA-256 hex string")
        collection = self._resolve()
        self._validate_metadata(collection, expected_catalog_sha256)
        return PinnedQdrantRetriever(self, collection, expected_catalog_sha256)

    def search(self, plan: QueryPlan, *, limit: int) -> tuple[Hit, ...]:
        return self._search(plan, limit=limit)

    def _search(
        self, plan: QueryPlan, *, limit: int, collection: str | None = None,
        expected_catalog_sha256: str | None = None,
    ) -> tuple[Hit, ...]:
        if limit < 1:
            raise ValueError("Search limit must be positive")
        if not plan.semantic_query.strip():
            raise ValueError("Search query must be nonempty")
        # Resolve once: metadata validation and querying must use the same snapshot
        # even if another process publishes an updated alias in between.
        collection = collection if collection is not None else self._resolve()
        path = self._collection_path(collection)
        self._validate_metadata(collection, expected_catalog_sha256)
        response = self.transport.request("POST", path + "/points/query", {
            "query": self._vector(self.embedder.embed_query(plan.semantic_query)),
            "filter": qdrant_filter(plan.filters), "limit": limit,
            "with_payload": True, "with_vector": False,
        })
        hits = []
        fields = set(Film.__dataclass_fields__)
        for point in response["points"]:
            payload = point["payload"]
            data = {key: payload[key] for key in fields}
            data["tags"] = tuple(data["tags"])
            film = Film(**data)
            if not plan.filters.matches(film):
                raise GroundingError("Qdrant returned a hit violating strict filters")
            hits.append(Hit(Chunk(payload["chunk_id"], film, payload["text"]), float(point["score"])))
        return tuple(hits)


@dataclass(frozen=True)
class PinnedQdrantRetriever:
    """Alias updates cannot switch the collection beneath a composed RAG service."""

    index: QdrantIndex
    collection: str
    catalog_sha256: str

    def search(self, plan: QueryPlan, *, limit: int) -> tuple[Hit, ...]:
        return self.index._search(plan, limit=limit, collection=self.collection,
                                  expected_catalog_sha256=self.catalog_sha256)
