"""Orchestration: plan → retrieve/compute → group → generate → validate."""

from cartoon_rag.domain import (
    Answer, Chunk, Evidence, ExactResult, GroundingError, Intent, Status, Trace,
)
from cartoon_rag.ports import AnswerGenerator, ExactCatalog, QueryPlanner, Retriever


def validate_answer(
    answer: Answer, context: tuple[Evidence, ...], *, exact_count: ExactResult | None = None,
) -> None:
    cards = {item.film.movie_id: item for item in context}
    allowed = {item.film.source_id for item in context}
    allowed.update(chunk.chunk_id for item in context for chunk in item.chunks)
    if not answer.text.strip():
        raise GroundingError("Empty answer")
    if not set(answer.source_ids) <= allowed:
        raise GroundingError("Answer cites sources outside its context")
    computed_empty_count = exact_count is not None and exact_count.value == 0
    if answer.status is Status.ANSWERED and not answer.source_ids and not computed_empty_count:
        raise GroundingError("Answered result requires supporting sources")
    if answer.recommendations and answer.status is not Status.ANSWERED:
        raise GroundingError("Recommendations require answered status")
    seen: set[str] = set()
    for rec in answer.recommendations:
        item = cards.get(rec.movie_id)
        if item is None or item.film.title != rec.title or rec.movie_id in seen:
            raise GroundingError("Unknown, renamed or duplicated recommendation")
        seen.add(rec.movie_id)
        own = {item.film.source_id, *(chunk.chunk_id for chunk in item.chunks)}
        if (not rec.reason.strip() or not rec.source_ids
                or not set(rec.source_ids) <= own
                or not set(rec.source_ids) <= set(answer.source_ids)):
            raise GroundingError("Recommendation requires its own supporting sources")


class SearchService:
    def __init__(
        self, planner: QueryPlanner, retriever: Retriever, catalog: ExactCatalog,
        generator: AnswerGenerator, *, retrieval_k: int = 10, context_movies: int = 3,
    ) -> None:
        if retrieval_k < 1 or context_movies < 1:
            raise ValueError("Retrieval and context limits must be positive")
        self.planner = planner
        self.retriever = retriever
        self.catalog = catalog
        self.generator = generator
        self.retrieval_k = retrieval_k
        self.context_movies = context_movies

    def ask(self, question: str) -> Trace:
        plan = self.planner.plan(question)
        if plan.early_status is not None:
            answer = Answer(plan.early_status, plan.early_text)
            return Trace(question, plan, (), (), None, answer)

        hits = ()
        exact = None
        if plan.intent is Intent.SEARCH:
            hits = tuple(self.retriever.search(plan, limit=self.retrieval_k))
            groups: dict[str, list[Chunk]] = {}
            for hit in hits:
                film = hit.chunk.film
                # Fail loudly if an adapter violates prefiltering; postfiltering
                # would silently hide lost top-k candidates.
                if not plan.filters.matches(film):
                    raise GroundingError("Retriever violated strict filters")
                group = groups.setdefault(film.movie_id, [])
                if group and group[0].film != film:
                    raise GroundingError("Mixed movie revisions in retrieval")
                if hit.chunk.chunk_id not in {chunk.chunk_id for chunk in group}:
                    group.append(hit.chunk)
            context = tuple(
                Evidence(chunks[0].film, tuple(chunks))
                for chunks in list(groups.values())[:self.context_movies]
            )
        else:
            evidence, exact = self.catalog.execute(plan)
            context = tuple(evidence)
            if any(not plan.filters.matches(item.film) for item in context):
                raise GroundingError("Catalog violated strict filters")
            if exact and not set(exact.source_ids) <= {item.film.source_id for item in context}:
                raise GroundingError("Exact result cites sources outside its context")

        if not context and exact is None:
            answer = Answer(Status.INSUFFICIENT, "В каталоге нет подтверждения по этому запросу.")
        else:
            answer = self.generator.generate(question, plan, context, exact)
        validate_answer(answer, context, exact_count=exact if plan.intent is Intent.COUNT else None)
        return Trace(question, plan, hits, context, exact, answer)
