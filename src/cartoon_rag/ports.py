"""Only these interfaces couple application services to external providers."""

from typing import Protocol, Sequence

from cartoon_rag.domain import (
    Answer, Chunk, EmbeddingBatch, EmbeddingIdentity, Evidence, ExactResult, Hit,
    PublicationResult, QueryPlan, Trace,
)


class QueryPlanner(Protocol):
    def plan(self, question: str) -> QueryPlan: ...


class QuestionAnswerer(Protocol):
    def ask(self, question: str) -> Trace: ...


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Return vectors with their producer identity, including an empty batch."""
        ...


class QueryEmbedder(Protocol):
    @property
    def identity(self) -> EmbeddingIdentity: ...

    def embed_query(self, text: str) -> Sequence[float]: ...


class Chunker(Protocol):
    def split(self, text: str) -> Sequence[str]: ...


class IndexWriter(Protocol):
    def replace_snapshot(
        self, chunks: Sequence[Chunk], batch: EmbeddingBatch,
    ) -> PublicationResult:
        """Atomically publish a complete active snapshot, retiring stale chunks.

        On failure leave the previous snapshot readable. Adapters must validate
        producer identity and vector dimensions before mutation, and translate
        provider failures to infrastructure errors. Return the published snapshot.
        """
        ...


class Retriever(Protocol):
    def search(self, plan: QueryPlan, *, limit: int) -> Sequence[Hit]:
        """Apply plan filters and active snapshot scope BEFORE ranking/limiting.

        Return descending scores. Raise RetrievalUnavailable on DB failure.
        Hybrid retrieval can implement this same interface.
        """
        ...


class ExactCatalog(Protocol):
    def execute(self, plan: QueryPlan) -> tuple[Sequence[Evidence], ExactResult | None]:
        """Lookup, duration comparison or exact count over the active catalog.

        Counts use the complete filtered catalog, not vector top-k. A comparison
        returns duration minus budget (positive means the film is too long).
        References must resolve to catalog cards. Missing films return ([], None).
        A zero count returns ExactResult(0, "movies", ()), not a missing result.
        Lookup/comparison require one ID. Comparison must not filter out its card
        by duration; invalid or early plans must be rejected by the adapter.
        Use an immutable active snapshot; full RAG composition must ensure its
        catalog revision agrees with the retriever's published index.
        """
        ...


class AnswerGenerator(Protocol):
    def generate(
        self, question: str, plan: QueryPlan, context: Sequence[Evidence],
        exact_result: ExactResult | None,
    ) -> Answer:
        """Use supplied facts only; return insufficient if evidence is inadequate."""
        ...
