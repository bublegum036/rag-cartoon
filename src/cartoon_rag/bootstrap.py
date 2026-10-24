"""Composition root: the only place that chooses concrete providers."""

from cartoon_rag.adapters.fastembed import E5Embedder
from cartoon_rag.adapters.qdrant import QdrantHTTP, QdrantIndex
from cartoon_rag.settings import Settings


def build_search(settings: Settings) -> tuple[E5Embedder, QdrantIndex]:
    embedder = E5Embedder(
        revision=settings.model_revision, cache=settings.model_cache,
        threads=settings.embedding_threads,
    )
    transport = QdrantHTTP(
        settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=settings.timeout_seconds,
    )
    return embedder, QdrantIndex(transport, settings.qdrant_alias, embedder)
