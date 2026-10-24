"""Composition root: the only place that chooses concrete providers."""

from collections.abc import Sequence

from cartoon_rag.adapters.fastembed import E5Embedder
from cartoon_rag.adapters.gemini import GeminiAnswerGenerator, GeminiHTTP
from cartoon_rag.adapters.json_catalog import JsonExactCatalog
from cartoon_rag.adapters.qdrant import QdrantHTTP, QdrantIndex
from cartoon_rag.adapters.rule_planner import RuleBasedQueryPlanner
from cartoon_rag.domain import Film
from cartoon_rag.application import SearchService
from cartoon_rag.ingestion import active_snapshot
from cartoon_rag.ports import AnswerGenerator, ExactCatalog, QueryPlanner
from cartoon_rag.settings import Settings
from cartoon_rag.snapshots import catalog_fingerprint


def build_planner(films: Sequence[Film]) -> QueryPlanner:
    return RuleBasedQueryPlanner({film.movie_id: film.title for film in active_snapshot(films)})


def build_exact_catalog(films: Sequence[Film]) -> ExactCatalog:
    return JsonExactCatalog(films)


def build_search(settings: Settings) -> tuple[E5Embedder, QdrantIndex]:
    embedder = E5Embedder(
        revision=settings.model_revision, cache=settings.model_cache,
        threads=settings.embedding_threads,
    )
    transport = QdrantHTTP(
        settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=settings.timeout_seconds,
    )
    return embedder, QdrantIndex(transport, settings.qdrant_alias, embedder)


def build_generator(settings: Settings) -> AnswerGenerator:
    return GeminiAnswerGenerator(
        GeminiHTTP(api_key=settings.gemini_api_key, model=settings.gemini_model,
                   timeout=settings.generation_timeout_seconds),
        max_output_tokens=settings.generation_max_output_tokens,
    )


def build_rag(
    films: Sequence[Film], settings: Settings, *, generator: AnswerGenerator | None = None,
) -> SearchService:
    snapshot = active_snapshot(films)
    selected_generator = generator if generator is not None else build_generator(settings)
    _, index = build_search(settings)
    retriever = index.bind_snapshot(catalog_fingerprint(snapshot))
    return SearchService(build_planner(snapshot), retriever, build_exact_catalog(snapshot), selected_generator)
