"""Only these interfaces couple application services to external providers."""

from typing import Protocol, Sequence

from cartoon_rag.domain import (
    Answer, Chunk, Evidence, ExactResult, Hit, QueryPlan,
)


class QueryPlanner(Protocol):
    def plan(self, question: str) -> QueryPlan: ...


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class Chunker(Protocol):
    def split(self, text: str) -> Sequence[str]: ...


class IndexWriter(Protocol):
    def replace_snapshot(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]],
    ) -> None:
        """Atomically publish a complete active snapshot, retiring stale chunks.

        On failure leave the previous snapshot readable. Adapters must validate
        vector dimensions and translate provider failures to infrastructure errors.
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
        """
        ...


class AnswerGenerator(Protocol):
    def generate(
        self, question: str, plan: QueryPlan, context: Sequence[Evidence],
        exact_result: ExactResult | None,
    ) -> Answer:
        """Use supplied facts only; return insufficient if evidence is inadequate."""
        ...
