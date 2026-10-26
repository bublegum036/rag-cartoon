"""Provider-independent selection validation and deterministic answer assembly."""

from collections.abc import Sequence
from dataclasses import dataclass

from cartoon_rag.domain import (
    Answer, Evidence, ExactResult, Film, GroundingError, Intent, QueryPlan, Recommendation, Status,
)


RENDERER_VERSION = "extractive-renderer-v2"


@dataclass(frozen=True)
class SourceFragment:
    fragment_id: str
    movie_id: str
    source_id: str
    text: str


def _cards(plan: QueryPlan, context: Sequence[Evidence]) -> dict[str, Film]:
    cards = {}
    for item in context:
        film = item.film
        if film.movie_id in cards or not plan.filters.matches(film):
            raise GroundingError("Extractive context contains duplicate or disallowed cards")
        cards[film.movie_id] = film
    return cards


def source_fragments(plan: QueryPlan, context: Sequence[Evidence]) -> tuple[SourceFragment, ...]:
    """Offer complete supplied chunks (or a full exact card), never model text."""
    _cards(plan, context)
    fragments = []
    seen = set()
    for item in context:
        rows = [(chunk.chunk_id, chunk.text) for chunk in item.chunks]
        if any(chunk.film != item.film for chunk in item.chunks):
            raise GroundingError("Extractive chunk belongs to a different card")
        if not rows:
            rows = [(item.film.source_id, item.film.synopsis)]
        for source_id, text in rows:
            if (not isinstance(source_id, str) or not source_id.strip() or source_id in seen
                    or not isinstance(text, str) or not text.strip()
                    or " ".join(text.split()) not in " ".join(item.film.synopsis.split())):
                raise GroundingError("Extractive fragment is duplicate, empty or not supported by its card")
            seen.add(source_id)
            fragments.append(SourceFragment(source_id, item.film.movie_id, source_id, text))
    return tuple(fragments)


def render_selection(raw: object, plan: QueryPlan, context: Sequence[Evidence]) -> Answer:
    if plan.intent is not Intent.SEARCH:
        raise GroundingError("Fragment selection is only used for semantic queries")
    if not isinstance(raw, dict) or set(raw) != {"status", "selections"}:
        raise GroundingError("Selection must contain only status and selections")
    if not isinstance(raw["status"], str) or raw["status"] not in {s.value for s in Status}:
        raise GroundingError("Unknown selection status")
    rows, status = raw["selections"], Status(raw["status"])
    if not isinstance(rows, list) or (status is Status.ANSWERED) != bool(rows):
        raise GroundingError("Only answered selection requires nonempty selections")
    cards = _cards(plan, context)
    fragments = {item.fragment_id: item for item in source_fragments(plan, context)}
    if status is not Status.ANSWERED:
        text = ("В переданном контексте нет подтверждения для ответа." if status is Status.INSUFFICIENT
                else "Уточните, какой фильм или сюжет вас интересует.")
        return Answer(status, text)
    seen, recommendations, texts, sources = set(), [], [], []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"movie_id", "fragment_ids"}:
            raise GroundingError("Selection row must contain only movie_id and fragment_ids")
        movie_id, ids = row["movie_id"], row["fragment_ids"]
        if not isinstance(movie_id, str) or movie_id not in cards or movie_id in seen:
            raise GroundingError("Selected movie is unknown or duplicated")
        if (not isinstance(ids, list) or not ids or any(not isinstance(value, str) for value in ids)
                or len(set(ids)) != len(ids)):
            raise GroundingError("Fragment IDs must be a nonempty list of unique strings")
        if any(value not in fragments or fragments[value].movie_id != movie_id for value in ids):
            raise GroundingError("Selected fragment is unknown or belongs to another movie")
        seen.add(movie_id)
        film = cards[movie_id]
        selected = [fragments[value] for value in ids]
        own = tuple(dict.fromkeys([film.source_id, *(item.source_id for item in selected)]))
        reason = "\n".join(f"«{item.text}»" for item in selected)
        texts.append(f"«{film.title}» ({film.year}) — {film.duration_minutes} минут.\n{reason}")
        recommendations.append(Recommendation(movie_id, film.title, reason, own))
        sources.extend(own)
    return Answer(Status.ANSWERED, "\n\n".join(texts), tuple(dict.fromkeys(sources)), tuple(recommendations))


def render_exact(plan: QueryPlan, context: Sequence[Evidence], result: ExactResult | None) -> Answer:
    cards = _cards(plan, context)
    if plan.intent is Intent.COUNT:
        if (result is None or type(result.value) is not int or result.unit != "movies"
                or result.value != len(cards) or len(result.source_ids) != len(cards)
                or set(result.source_ids) != {film.source_id for film in cards.values()}):
            raise GroundingError("Exact count does not match its complete evidence")
        text = f"По условиям запроса найдено {result.value} фильмов."
        conditions = []
        if plan.filters.duration_minutes_lt is not None:
            conditions.append(f"длительность < {plan.filters.duration_minutes_lt} минут")
        if plan.filters.duration_minutes_lte is not None:
            conditions.append(f"длительность <= {plan.filters.duration_minutes_lte} минут")
        if conditions:
            text += " Условия: " + "; ".join(conditions) + "."
        return Answer(Status.ANSWERED, text, result.source_ids)
    if plan.intent not in (Intent.LOOKUP, Intent.COMPARE_DURATION):
        raise GroundingError("Unsupported exact operation")
    if len(cards) != 1 or tuple(cards) != tuple(plan.filters.movie_ids):
        raise GroundingError("Exact fact requires the selected card")
    film = next(iter(cards.values()))
    text = f"«{film.title}» ({film.year}) — {film.duration_minutes} минут."
    if plan.intent is Intent.LOOKUP:
        if result is not None:
            raise GroundingError("Lookup must not contain an arithmetic result")
        # LOOKUP identifies a card, not the requested fact: named-film queries
        # also ask about plot events. Preserve its full synopsis without guessing
        # another question grammar or generating unsupported connective prose.
        text += f"\n«{film.synopsis}»"
    else:
        budget = plan.time_budget_minutes
        if (type(budget) is not int or budget < 1 or result is None or type(result.value) is not int
                or result.unit != "minutes" or result.value != film.duration_minutes - budget
                or result.source_ids != (film.source_id,)):
            raise GroundingError("Duration comparison does not match its evidence")
        if result.value > 0:
            outcome = f"не подходит: превышение на {result.value} минут"
        elif result.value < 0:
            outcome = f"подходит: остаётся {-result.value} минут"
        else:
            outcome = "подходит: укладывается точно (разница 0 минут)"
        text += f" При бюджете {budget} минут фильм {outcome}."
    return Answer(Status.ANSWERED, text, (film.source_id,))
