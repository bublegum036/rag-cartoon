"""Trusted explanations for structured planner reasons; no provider text."""

from enum import Enum
from types import MappingProxyType

from cartoon_rag.domain import Status


EARLY_REPLY_VERSION = "early-replies-ru-v2"


class EarlyReason(str, Enum):
    EMPTY_QUESTION = "empty_question"
    UNKNOWN_TITLE = "unknown_title"
    AMBIGUOUS_TITLE = "ambiguous_title"
    MISSING_MOVIE = "missing_movie"
    UNSUPPORTED_DURATION_FILTER = "unsupported_duration_filter"
    AMBIGUOUS_DURATION = "ambiguous_duration"
    UNSUPPORTED_CONSTRAINT = "unsupported_constraint"
    SEMANTIC_COUNT = "semantic_count"
    MISSING_VOICE_ACTORS = "missing_voice_actors"
    MISSING_RATING = "missing_rating"
    MISSING_AGE_RESTRICTION = "missing_age_restriction"
    MISSING_BIOGRAPHY = "missing_biography"
    UNSUPPORTED_FACT = "unsupported_fact"


_REPLIES = MappingProxyType({
    EarlyReason.EMPTY_QUESTION: (Status.CLARIFY, "Опишите сюжет или укажите название фильма."),
    EarlyReason.UNKNOWN_TITLE: (Status.CLARIFY, "Уточните полное название фильма из каталога."),
    EarlyReason.AMBIGUOUS_TITLE: (Status.CLARIFY, "Уточните, какой один фильм вы имеете в виду."),
    EarlyReason.MISSING_MOVIE: (Status.CLARIFY, "О каком фильме идёт речь? Укажите название."),
    EarlyReason.UNSUPPORTED_DURATION_FILTER: (
        Status.CLARIFY,
        "Сейчас поддерживаются только верхние границы длительности в целых минутах. Уточните условие длительности.",
    ),
    EarlyReason.AMBIGUOUS_DURATION: (
        Status.CLARIFY, "Уточните условие длительности: граница включается или требуется строго меньше?",
    ),
    EarlyReason.UNSUPPORTED_CONSTRAINT: (
        Status.CLARIFY, "Это сочетание условий пока не поддерживается. Уточните условия запроса.",
    ),
    EarlyReason.SEMANTIC_COUNT: (
        Status.CLARIFY,
        "Точный подсчёт по сюжету пока не поддерживается. Уточните запрос: можно посчитать все фильмы по году и длительности.",
    ),
    EarlyReason.MISSING_VOICE_ACTORS: (Status.INSUFFICIENT, "В каталоге нет сведений об актёрах озвучивания."),
    EarlyReason.MISSING_RATING: (Status.INSUFFICIENT, "В каталоге нет сведений о рейтингах."),
    EarlyReason.MISSING_AGE_RESTRICTION: (Status.INSUFFICIENT, "В каталоге нет сведений о возрастных ограничениях."),
    EarlyReason.MISSING_BIOGRAPHY: (Status.INSUFFICIENT, "В каталоге нет биографических сведений."),
    EarlyReason.UNSUPPORTED_FACT: (
        Status.INSUFFICIENT, "Система не может подтвердить запрошенные сведения по карточкам каталога.",
    ),
})


def early_reply(reason: EarlyReason) -> tuple[Status, str]:
    return _REPLIES[reason]
