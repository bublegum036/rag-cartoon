"""Conservative Russian query grammar; no access to film facts or evaluation data."""

import re
from collections.abc import Mapping

from cartoon_rag.domain import Filters, Intent, QueryPlan, Status


_QUOTED = re.compile(r'«([^»]*)»|"([^"]*)"|„([^“]*)“|“([^”]*)”')
_MINUTES = r"(?:минут(?:а|ы)?|мин\.?)"
_NUMBER = r"(?P<value>[0-9]+)(?![0-9]|[.,][0-9])"
_DURATION = re.compile(
    r"(?<!\w)(?P<operator>не\s+(?:больше|более|дольше|длиннее)|максимум|"
    r"короче|меньше|менее)\s+(?:чем\s+)?" + _NUMBER
    + r"(?:\s*(?P<unit>" + _MINUTES + r"))?(?!\w)"
)
_BUDGET = re.compile(
    r"\b(?:у\s+меня\s+(?:(?:есть|всего|только)\s+)*|за\s+)"
    + _NUMBER + r"\s*" + _MINUTES + r"(?!\w)"
)
_COMPARE = re.compile(r"\b(?:подойдет|подходит|успею|успеем|уложусь|уложимся|хватит)\b")
_COUNT = re.compile(r"\bсколько\b[\s\S]*\b(?:мультфильмов|фильмов)\b|\bпосчитай(?:те)?\b")
_UNPARSED_DURATION = re.compile(
    r"\b(?:короче|меньше|менее|дольше|длиннее|больше|более|максимум|"
    r"коротк\w*|длительностью|минут\w*|мин|час|часа|часов)\b"
)
_FACT_QUESTION = re.compile(r"\b(?:длится|идет|продолжается|длительность|продолжительность|года|вышел|вышла)\b")
_DURATION_FACT_QUESTION = re.compile(
    r"\b(?:сколько\s+(?:минут|мин\.?|часов)\s+(?:длится|идет|продолжается)|"
    r"(?:какова|какая)\s+(?:длительность|продолжительность)\s+в\s+(?:минутах|часах))\b"
)
_AGE_RESTRICTION = re.compile(
    r"\bвозрастн\w*\s+(?:огранич\w*|рейтинг\w*|ценз\w*|допуск\w*|категори\w*|маркировк\w*)|"
    r"\b(?:огранич\w*|допуск\w*)\s+по\s+возрасту\b|"
    r"\bс\s+какого\s+возраста\b|\b(?:на|для)\s+како\w*\s+возраст\w*|"
    r"\b\d+\s*\+|\bдля\s+детей\s+(?:от\s+)?\d+"
)
_MISSING_FACTS = (
    (re.compile(r"\bозвуч\w*|\bактер\w*\s+озвуч\w*"), "актёрах озвучивания"),
    (re.compile(r"\bрейтинг\w*"), "рейтингах"),
    (_AGE_RESTRICTION, "возрастных ограничениях"),
)


def _normalize(text: str) -> str:
    return text.lower().replace("ё", "е")


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", _normalize(text)))


def _mask(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    return "".join(chars)


def _early(question: str, status: Status, message: str) -> QueryPlan:
    return QueryPlan(question, early_status=status, early_text=message)


class RuleBasedQueryPlanner:
    """A replaceable QueryPlanner adapter for a deliberately limited grammar.

    The constructor receives only movie IDs and titles. No synopsis, duration,
    reference answer or source IDs are available to infer an answer from.
    Quoted titles must match a catalog title after case/ё/token normalization;
    unquoted titles must contain the complete title. No fuzzy/morphological match.
    """

    version = "ru-rules-v1.1"

    def __init__(self, titles: Mapping[str, str]) -> None:
        self._titles = []
        for movie_id, title in titles.items():
            if not isinstance(movie_id, str) or not movie_id.strip() or not isinstance(title, str):
                raise ValueError("Movie references require nonempty string IDs and titles")
            tokens = _tokens(title)
            if not tokens:
                raise ValueError("Movie references require nonempty titles")
            pattern = re.compile(r"(?<!\w)" + r"[\W_]+".join(map(re.escape, tokens)) + r"(?!\w)")
            self._titles.append((movie_id, tokens, pattern))

    def plan(self, question: str) -> QueryPlan:
        original = question.strip()
        if not original:
            return _early(original, Status.CLARIFY, "Опишите сюжет или укажите название фильма.")
        text = _normalize(original)
        quoted = list(_QUOTED.finditer(text))
        spans = [match.span() for match in quoted]
        movie_ids = []
        unknown_title = False
        for quote in quoted:
            name = _tokens(next(group for group in quote.groups() if group is not None))
            matches = [movie_id for movie_id, tokens, _ in self._titles if tokens == name]
            unknown_title |= not matches
            movie_ids.extend(matches)
        # Quoted text is handled as an exact reference, never as a substring.
        unquoted = _mask(text, spans)
        for movie_id, _, pattern in self._titles:
            matches = list(pattern.finditer(unquoted))
            if matches:
                movie_ids.append(movie_id)
                spans.extend(match.span() for match in matches)
        movie_ids = list(dict.fromkeys(movie_ids))
        masked = _mask(text, spans)

        for pattern, fact in _MISSING_FACTS:
            if pattern.search(masked):
                return _early(original, Status.INSUFFICIENT,
                              f"В каталоге нет сведений о {fact}.")
        if unknown_title:
            return _early(original, Status.CLARIFY,
                          "Уточните название: используйте полное название из каталога.")
        if len(movie_ids) > 1:
            return _early(original, Status.CLARIFY, "Укажите один фильм для этого запроса.")
        if any(char in _mask(text, [match.span() for match in quoted]) for char in '«»"„“”'):
            return _early(original, Status.CLARIFY, "Проверьте кавычки вокруг названия фильма.")

        duration_matches = list(_DURATION.finditer(masked))
        budget_matches = list(_BUDGET.finditer(masked))
        # A bare number is accepted at a phrase boundary, but "90 seconds"
        # or "90 characters" must not accidentally become 90 movie minutes.
        if any(match["unit"] is None
               and not re.match(r"\s*(?:$|[.,;:?!]|и\b|или\b)", masked[match.end():])
               for match in duration_matches):
            return _early(original, Status.CLARIFY,
                          "Укажите длительность в целых минутах.")
        constraint_spans = [match.span() for match in duration_matches + budget_matches]
        # Asking for a duration in minutes does not constrain its value. Mask
        # only the recognized fact phrase, leaving every numeric condition to
        # the normal validation and routing below.
        fact_spans = [match.span() for match in _DURATION_FACT_QUESTION.finditer(masked)]
        remainder = _mask(masked, constraint_spans + fact_spans)
        if re.search(r"\d", remainder) or _UNPARSED_DURATION.search(remainder):
            return _early(original, Status.CLARIFY,
                          "Уточните ограничение: поддерживаются целые минуты, «короче N» "
                          "и «не больше N». Другие числовые условия пока не поддерживаются.")
        values = [int(match["value"]) for match in duration_matches + budget_matches]
        if any(value < 1 for value in values):
            return _early(original, Status.CLARIFY, "Укажите положительное число минут.")
        if any(re.search(r"\bне\s*$", masked[:match.start()])
               for match in duration_matches + budget_matches):
            return _early(original, Status.CLARIFY,
                          "Отрицание этого условия длительности пока не поддерживается. Уточните условие.")
        if any(match[0].startswith("за") for match in budget_matches) and not _COMPARE.search(masked):
            return _early(original, Status.CLARIFY,
                          "Уточните, означает ли «за N минут» бюджет просмотра: «у меня N минут».")
        if len(constraint_spans) > 1 and re.search(r"\bили\b", masked):
            return _early(original, Status.CLARIFY, "Укажите одно условие длительности без «или».")
        if len(budget_matches) > 1 or (budget_matches and duration_matches):
            return _early(original, Status.CLARIFY, "Укажите один бюджет времени или условие длительности.")

        lt = [int(match["value"]) for match in duration_matches
              if match["operator"] in ("короче", "меньше", "менее")]
        lte = [int(match["value"]) for match in duration_matches
               if match["operator"] not in ("короче", "меньше", "менее")]
        budget = int(budget_matches[0]["value"]) if budget_matches else None

        if _COUNT.search(masked):
            # ExactCatalog has no semantic topic predicate. Never count vector top-k.
            leftover = re.sub(r"\b(?:в|из)\s+каталог(?:е|а)\b", " ", remainder)
            leftover = re.sub(r"\b(?:сколько|всего|все|посчитай(?:те)?|количество|мультфильмов|"
                              r"мультфильмы|фильмов|фильмы|есть|доступно|и)\b", " ", leftover)
            if re.search(r"\w", leftover) or movie_ids or budget is not None:
                return _early(original, Status.CLARIFY,
                              "Могу точно посчитать все фильмы каталога по длительности. "
                              "Уточните запрос без сюжетного или другого дополнительного условия.")
            return QueryPlan(original, intent=Intent.COUNT, filters=Filters(
                duration_minutes_lt=min(lt) if lt else None,
                duration_minutes_lte=min(lte) if lte else None,
            ))

        if movie_ids and budget is not None:
            return QueryPlan(original, intent=Intent.COMPARE_DURATION,
                             filters=Filters(movie_ids=tuple(movie_ids)), time_budget_minutes=budget)
        if _COMPARE.search(masked):
            return _early(original, Status.CLARIFY,
                          "Для сравнения укажите один фильм и бюджет: «у меня N минут».")
        if movie_ids:
            if duration_matches:
                return _early(original, Status.CLARIFY,
                              "Для проверки названного фильма укажите бюджет: «у меня N минут».")
            return QueryPlan(original, intent=Intent.LOOKUP, filters=Filters(movie_ids=tuple(movie_ids)))
        if (_FACT_QUESTION.search(masked)
                and not re.search(r"\b(?:про|где|котор\w*)\b", masked)):
            return _early(original, Status.CLARIFY, "О каком фильме идёт речь? Укажите название.")

        if budget is not None:
            lte.append(budget)
        semantic = _mask(original, constraint_spans).strip(" \t\n.,;:?!—-") or "мультфильм"
        return QueryPlan(semantic, filters=Filters(
            duration_minutes_lt=min(lt) if lt else None,
            duration_minutes_lte=min(lte) if lte else None,
        ))
