"""Exact operations on an immutable active snapshot loaded from catalog JSON."""

from collections.abc import Sequence

from cartoon_rag.domain import Evidence, ExactResult, Film, Intent, QueryPlan
from cartoon_rag.ingestion import active_snapshot


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class JsonExactCatalog:
    """No vector top-k, language model or mutable file reads during execution."""

    def __init__(self, films: Sequence[Film]) -> None:
        self._films = tuple(sorted(active_snapshot(films), key=lambda film: film.movie_id))

    def execute(self, plan: QueryPlan) -> tuple[tuple[Evidence, ...], ExactResult | None]:
        if plan.early_status is not None or plan.intent not in (
            Intent.LOOKUP, Intent.COMPARE_DURATION, Intent.COUNT,
        ):
            raise ValueError("Exact catalog requires a non-early lookup, count or duration comparison")
        for name, bound in (("duration_minutes_lt", plan.filters.duration_minutes_lt),
                            ("duration_minutes_lte", plan.filters.duration_minutes_lte)):
            if bound is not None:
                _positive_integer(bound, name)
        if plan.intent in (Intent.LOOKUP, Intent.COMPARE_DURATION) and len(plan.filters.movie_ids) != 1:
            raise ValueError("Lookup and duration comparison require exactly one movie ID")
        budget = None
        if plan.intent is Intent.COMPARE_DURATION:
            budget = _positive_integer(plan.time_budget_minutes, "time_budget_minutes")
            if (plan.filters.duration_minutes_lt is not None
                    or plan.filters.duration_minutes_lte is not None):
                raise ValueError("Comparison must select its card without duration filters")
        elif plan.time_budget_minutes is not None:
            raise ValueError("Time budgets require a duration comparison")

        cards = tuple(film for film in self._films if plan.filters.matches(film))
        # Exact evidence is a full card; it has no synthetic embedding chunks.
        evidence = tuple(Evidence(film, ()) for film in cards)
        sources = tuple(film.source_id for film in cards)
        if plan.intent is Intent.COUNT:
            return evidence, ExactResult(len(cards), "movies", sources)
        if not cards:
            return (), None
        if budget is not None:
            return evidence, ExactResult(cards[0].duration_minutes - budget,
                                         "minutes", sources)
        return evidence, None
