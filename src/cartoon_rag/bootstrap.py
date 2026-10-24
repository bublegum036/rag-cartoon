"""Composition root: the only place that chooses concrete providers."""

from collections.abc import Sequence
from dataclasses import asdict

from cartoon_rag.adapters.fastembed import E5Embedder
from cartoon_rag.adapters.gemini import GeminiAnswerGenerator, GeminiHTTP, PROMPT_VERSION
from cartoon_rag.adapters.json_catalog import JsonExactCatalog
from cartoon_rag.adapters.qdrant import QdrantHTTP, QdrantIndex
from cartoon_rag.adapters.rule_planner import RuleBasedQueryPlanner
from cartoon_rag.domain import Film, GeneratorIdentity
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


def build_generator(settings: Settings, *, prompt_version: str | None = None) -> AnswerGenerator:
    return GeminiAnswerGenerator(
        GeminiHTTP(api_key=settings.gemini_api_key, model=settings.gemini_model,
                   timeout=settings.generation_timeout_seconds),
        max_output_tokens=settings.generation_max_output_tokens,
        prompt_version=prompt_version,
    )


def build_rag(
    films: Sequence[Film], settings: Settings, *, generator: AnswerGenerator | None = None,
    generator_identity: GeneratorIdentity | None = None,
) -> SearchService:
    return build_rag_runtime(
        films, settings, generator=generator, generator_identity=generator_identity,
    )[0]


def build_rag_runtime(
    films: Sequence[Film], settings: Settings, *, generator: AnswerGenerator | None = None,
    retrieval_k: int = 10, context_movies: int = 3,
    prompt_version: str | None = None,
    generator_identity: GeneratorIdentity | None = None,
) -> tuple[SearchService, dict]:
    """Build a runtime; injected generators require caller-owned, non-secret provenance."""
    if retrieval_k < 1 or context_movies < 1:
        raise ValueError("Retrieval and context limits must be positive")
    if generator is not None:
        if not isinstance(generator_identity, GeneratorIdentity):
            raise ValueError("An injected generator requires a GeneratorIdentity in generator_identity")
        if prompt_version is not None:
            raise ValueError("Configure the injected generator's prompt directly and supply generator_identity")
        selected_generator = generator
        selected_identity = generator_identity
    else:
        if generator_identity is not None:
            raise ValueError("generator_identity requires an injected generator")
        selected_prompt = PROMPT_VERSION if prompt_version is None else prompt_version
        selected_generator = build_generator(settings, prompt_version=selected_prompt)
        selected_identity = GeneratorIdentity(
            settings.gemini_model, selected_prompt, settings.generation_max_output_tokens,
        )
    snapshot = active_snapshot(films)
    _, index = build_search(settings)
    retriever = index.bind_snapshot(catalog_fingerprint(snapshot))
    service = SearchService(build_planner(snapshot), retriever, build_exact_catalog(snapshot),
                            selected_generator, retrieval_k=retrieval_k, context_movies=context_movies)
    # Explicit whitelist: never serialize Settings or adapter state containing keys.
    identity = {
        "index_name": settings.qdrant_alias, "snapshot_id": retriever.collection,
        "catalog_sha256": retriever.catalog_sha256, "embedding": asdict(index.embedder.identity),
        "generator_model": selected_identity.model, "prompt_version": selected_identity.prompt_version,
        "planner_version": RuleBasedQueryPlanner.version,
        "retrieval_k": retrieval_k, "context_movies": context_movies,
        "max_output_tokens": selected_identity.max_output_tokens,
    }
    return service, identity
