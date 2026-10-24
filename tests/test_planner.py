import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from cartoon_rag.adapters.rule_planner import RuleBasedQueryPlanner
from cartoon_rag.application import SearchService
from cartoon_rag.bootstrap import build_planner
from cartoon_rag.cli import main
from cartoon_rag.domain import Answer, Chunk, Evidence, Hit, Intent, Status
from cartoon_rag.evaluation import load_cases
from cartoon_rag.ingestion import load_catalog


ROOT = Path(__file__).resolve().parents[1]


class PlannerTest(unittest.TestCase):
    def setUp(self):
        self.films = load_catalog(ROOT / "catalog.json")
        self.planner = build_planner(self.films)

    def test_duration_paraphrases_and_boundaries(self):
        for phrase, bound, inclusive in (
            ("короче 90 минут", 90, False), ("меньше чем 81 минуты", 81, False),
            ("менее 75 мин.", 75, False), ("не больше 90 минут", 90, True),
            ("не более 78", 78, True), ("не дольше 67 минут", 67, True),
            ("не длиннее 52 минут", 52, True), ("максимум 89 минут", 89, True),
            ("короче 90.", 90, False),
        ):
            with self.subTest(phrase=phrase):
                plan = self.planner.plan("Хочу про море, " + phrase)
                self.assertIsNone(plan.early_status)
                self.assertIs(plan.intent, Intent.SEARCH)
                self.assertEqual(plan.filters.duration_minutes_lt, None if inclusive else bound)
                self.assertEqual(plan.filters.duration_minutes_lte, bound if inclusive else None)
                self.assertIn("море", plan.semantic_query)
                self.assertNotIn(str(bound), plan.semantic_query)
        strict = self.planner.plan("про море, короче 78 минут")
        inclusive = self.planner.plan("про море, не больше 78 минут")
        self.assertFalse(strict.filters.matches(self.films[0]))
        self.assertTrue(inclusive.filters.matches(self.films[0]))

    def test_conjoined_bounds_are_intersected(self):
        plan = self.planner.plan("Про море короче 100 минут и короче 90 минут и не больше 85 минут")
        self.assertIsNone(plan.early_status)
        self.assertEqual(plan.filters.duration_minutes_lt, 90)
        self.assertEqual(plan.filters.duration_minutes_lte, 85)

    def test_budget_is_comparison_evidence_selection_not_retrieval_filter(self):
        for text in (
            "Подойдёт ли «Коралловая почта», если у меня только 90 минут?",
            "Успею посмотреть Коралловая почта за 80 минут?",
            "На «Коралловая почта» у меня есть 100 минут.",
        ):
            with self.subTest(text=text):
                plan = self.planner.plan(text)
                self.assertIsNone(plan.early_status)
                self.assertIs(plan.intent, Intent.COMPARE_DURATION)
                self.assertEqual(plan.filters.movie_ids, ("film-02",))
                self.assertIn(plan.time_budget_minutes, (90, 80, 100))
                self.assertIsNone(plan.filters.duration_minutes_lt)
                self.assertIsNone(plan.filters.duration_minutes_lte)
                self.assertTrue(plan.filters.matches(self.films[1]))

    def test_budget_for_open_recommendation_is_inclusive(self):
        plan = self.planner.plan("У меня только 90 минут. Посоветуй про море")
        self.assertIs(plan.intent, Intent.SEARCH)
        self.assertIsNone(plan.early_status)
        self.assertEqual(plan.filters.duration_minutes_lte, 90)
        self.assertIsNone(plan.time_budget_minutes)
        self.assertIn("море", plan.semantic_query)

    def test_named_lookup_and_title_normalization(self):
        for text in ('Сколько длится "маяк ДЛЯ кита"?', "Какого года МАЯК ДЛЯ КИТА?",
                     "Расскажи о «Маяк   для кита»"):
            with self.subTest(text=text):
                plan = self.planner.plan(text)
                self.assertIs(plan.intent, Intent.LOOKUP)
                self.assertIsNone(plan.early_status)
                self.assertEqual(plan.filters.movie_ids, ("film-01",))
        planner = RuleBasedQueryPlanner({"custom": "Полёт-7"})
        plan = planner.plan("Сколько длится «полет 7»?")
        self.assertIs(plan.intent, Intent.LOOKUP)
        self.assertEqual(plan.filters.movie_ids, ("custom",))
        self.assertIsNone(plan.early_status)

    def test_duration_fact_units_are_not_duration_constraints(self):
        for question, movie_id in (
            ("Сколько минут длится «Маяк для кита»?", "film-01"),
            ("Сколько минут длится „Маяк для кита“?", "film-01"),
            ('Сколько минут идёт "Робот и сад"?', "film-05"),
            ("Какова продолжительность в минутах «Робот и сад»?", "film-05"),
        ):
            with self.subTest(question=question):
                plan = self.planner.plan(question)
                self.assertIsNone(plan.early_status)
                self.assertIs(plan.intent, Intent.LOOKUP)
                self.assertEqual(plan.filters.movie_ids, (movie_id,))
                self.assertIsNone(plan.filters.duration_minutes_lt)
                self.assertIsNone(plan.filters.duration_minutes_lte)
        self.assertIs(self.planner.plan("Сколько минут он длится?").early_status, Status.CLARIFY)
        for condition in ("короче 90 минут", "не меньше 90 минут", "короче 90 секунд"):
            with self.subTest(condition=condition):
                plan = self.planner.plan(f"Сколько минут длится «Маяк для кита», если {condition}?")
                self.assertIs(plan.early_status, Status.CLARIFY)

    def test_age_as_story_topic_does_not_request_age_restrictions(self):
        for question in ("Посоветуй фильм про дружбу героев разного возраста",
                         "Ищу мультфильм про возраст и взросление",
                         "Нужна история о разнице в возрасте героев"):
            with self.subTest(question=question):
                plan = self.planner.plan(question)
                self.assertIsNone(plan.early_status)
                self.assertIs(plan.intent, Intent.SEARCH)
                self.assertEqual(plan.semantic_query, question)
        for question in ("Возрастное ограничение «Робот и сад»?",
                         "Какая возрастная маркировка у фильма?",
                         "Есть ли ограничение по возрасту?",
                         "С какого возраста можно смотреть «Робот и сад»?",
                         "На какой возраст рассчитан «Робот и сад»?",
                         "Для какого возраста этот мультфильм?",
                         "Можно смотреть детям 6+?", "Подходит ли для детей от 6 лет?"):
            with self.subTest(question=question):
                self.assertIs(self.planner.plan(question).early_status, Status.INSUFFICIENT)

    def test_unknown_partial_or_declined_quoted_title_requires_clarification(self):
        for title in ("Несуществующий фильм", "Маяк", "Маяке для кита", "Маяк для кита 2"):
            with self.subTest(title=title):
                self.assertIs(self.planner.plan(f"Сколько длится «{title}»?").early_status, Status.CLARIFY)

    def test_unquoted_title_is_not_a_partial_word(self):
        plan = self.planner.plan("Маяк для китаянки")
        self.assertIs(plan.intent, Intent.SEARCH)
        self.assertEqual(plan.filters.movie_ids, ())

    def test_ambiguous_or_multiple_titles_require_clarification(self):
        planner = RuleBasedQueryPlanner({"first": "Маяк", "second": "Маяк"})
        self.assertIs(planner.plan("Сколько длится «Маяк»?").early_status, Status.CLARIFY)
        self.assertIs(self.planner.plan("Сравни «Маяк для кита» и «Коралловая почта»").early_status,
                      Status.CLARIFY)

    def test_title_words_are_not_treated_as_fact_requests_or_constraints(self):
        planner = RuleBasedQueryPlanner({"custom": "Рейтинг 90 минут"})
        for question in ("Что за фильм «Рейтинг 90 минут»?", "Расскажи про Рейтинг 90 минут"):
            plan = planner.plan(question)
            self.assertIsNone(plan.early_status)
            self.assertIs(plan.intent, Intent.LOOKUP)
            self.assertEqual(plan.filters.movie_ids, ("custom",))

    def test_missing_facts_are_not_guessed(self):
        for text in ("Кто озвучивает девочку в «Маяке для кита»?",
                     "Какой рейтинг у «Робот и сад»?", "Возрастное ограничение «Робот и сад»?",
                     "Нужен мультфильм с рейтингом выше 8", "Подходит ли для детей от 6 лет?"):
            with self.subTest(text=text):
                plan = self.planner.plan(text)
                self.assertIs(plan.early_status, Status.INSUFFICIENT)
                self.assertTrue(plan.early_text)
        self.assertIsNone(self.planner.plan("Ищу мультфильм про актёра").early_status)

    def test_empty_or_unresolved_reference_requires_clarification(self):
        for text in ("", "  ", "Сколько он длится?", "Какого года этот фильм?",
                     "Подойдёт ли он за 90 минут?", "Сколько длится мультфильм?"):
            with self.subTest(text=text):
                self.assertIs(self.planner.plan(text).early_status, Status.CLARIFY)
        plan = self.planner.plan("Сколько длится фильм про девочку и кита?")
        self.assertIs(plan.intent, Intent.SEARCH)
        self.assertIsNone(plan.early_status)

    def test_count_uses_only_exact_catalog_predicates(self):
        for text in ("Сколько фильмов в каталоге?", "Сколько в каталоге мультфильмов?",
                     "Посчитай все фильмы из каталога"):
            with self.subTest(text=text):
                plan = self.planner.plan(text)
                self.assertIs(plan.intent, Intent.COUNT)
                self.assertIsNone(plan.early_status)
        plan = self.planner.plan("Сколько мультфильмов короче 90 минут?")
        self.assertIs(plan.intent, Intent.COUNT)
        self.assertEqual(plan.filters.duration_minutes_lt, 90)
        for text in ("Сколько фильмов про море короче 90 минут?",
                     "Сколько фильмов 2020 года?", "Посчитай фильмы с высоким рейтингом"):
            with self.subTest(text=text):
                self.assertIsNotNone(self.planner.plan(text).early_status)

    def test_unrepresentable_constraints_are_never_silently_dropped(self):
        for text in ("про море короче девяноста минут", "про море до 90 минут",
                     "про море не меньше 90 минут", "про море не короче 90 минут",
                     "про море длиннее 90 минут", "про море короче 1 часа",
                     "про море короче 90 секунд", "про море короче 2 дней",
                     "мультфильм, где меньше 90 героев",
                     "про море короче 80,5 минут", "про море короче 80.5 минут",
                     "про море короче -5 минут", "про море короче 0 минут",
                     "про море короче 90 или короче 100 минут", "про море после 2020 года",
                     "Покажи 2 фильма про море", "короткий мультфильм про море",
                     "У меня 90 минут и 100 минут", "У меня 90 минут, короче 80 минут",
                     "«Маяк для кита» короче 90 минут?", 'Сколько длится «Маяк для кита?'):
            with self.subTest(text=text):
                self.assertIs(self.planner.plan(text).early_status, Status.CLARIFY)

    def test_story_time_and_negated_budget_are_not_treated_as_viewing_limits(self):
        for text in ("Мультфильм про путешествие за 80 минут",
                     "«Робот и сад» про путешествие за 80 минут",
                     "Успею посмотреть «Робот и сад» не за 80 минут?"):
            with self.subTest(text=text):
                self.assertIs(self.planner.plan(text).early_status, Status.CLARIFY)

    def test_filter_only_search_has_nonempty_embedding_text(self):
        plan = self.planner.plan("Короче 90 минут")
        self.assertIsNone(plan.early_status)
        self.assertTrue(plan.semantic_query.strip())
        self.assertEqual(plan.filters.duration_minutes_lt, 90)

    def test_plan_is_not_answer_or_catalog_existence_check(self):
        plan = self.planner.plan("Нужен фильм про море короче 40 минут")
        self.assertIsNone(plan.early_status)
        self.assertEqual(plan.filters.duration_minutes_lt, 40)
        self.assertNotIn("78", plan.semantic_query)

    def test_composition_uses_only_latest_active_title_references(self):
        inactive = replace(self.films[0], version=2, active=False)
        planner = build_planner((*self.films, inactive))
        self.assertIs(planner.plan("Сколько длится «Маяк для кита»?").early_status, Status.CLARIFY)

    def test_supplied_questions_plan_routes_without_using_answers(self):
        cases = load_cases(ROOT / "questions.jsonl")
        for case in cases:
            with self.subTest(case=case.id):
                plan = self.planner.plan(case.question)
                if case.id == "unknown":
                    self.assertIs(plan.early_status, Status.INSUFFICIENT)
                elif case.id == "ambiguous":
                    self.assertIs(plan.early_status, Status.CLARIFY)
                elif case.id == "marine-long":
                    self.assertIs(plan.intent, Intent.COMPARE_DURATION)
                    self.assertEqual(plan.filters.movie_ids, ("film-02",))
                    self.assertEqual(plan.time_budget_minutes, 90)
                    self.assertIsNone(plan.early_status)
                else:
                    self.assertIs(plan.intent, Intent.SEARCH)
                    self.assertIsNone(plan.early_status)
                    if case.filters.duration_minutes_lt is not None:
                        self.assertEqual(plan.filters.duration_minutes_lt,
                                         case.filters.duration_minutes_lt)


class PlannerIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.films = load_catalog(ROOT / "catalog.json")
        self.planner = build_planner(self.films)

    def test_early_routes_do_not_access_retrieval_catalog_or_generator(self):
        class MustNotCall:
            def search(self, *args, **kwargs):
                raise AssertionError("Must not retrieve")

            def execute(self, *args):
                raise AssertionError("Must not execute")

            def generate(self, *args):
                raise AssertionError("Must not generate")

        dependency = MustNotCall()
        service = SearchService(self.planner, dependency, dependency, dependency)
        for question, expected in (("Сколько он длится?", Status.CLARIFY),
                                   ("Кто озвучивает девочку?", Status.INSUFFICIENT)):
            trace = service.ask(question)
            self.assertIs(trace.answer.status, expected)
            self.assertEqual(trace.answer.source_ids, ())
            self.assertEqual(trace.context, ())

    def test_duration_fact_with_units_reaches_catalog(self):
        film = self.films[0]
        catalog, retriever, generator = Mock(), Mock(), Mock()
        catalog.execute.return_value = ((Evidence(film, ()),), None)
        generator.generate.return_value = Answer(Status.ANSWERED, "Длительность: 78 минут.",
                                                (film.source_id,))
        trace = SearchService(self.planner, retriever, catalog, generator).ask(
            "Сколько минут длится „Маяк для кита“?")
        catalog.execute.assert_called_once_with(trace.plan)
        retriever.search.assert_not_called()
        generator.generate.assert_called_once()
        self.assertIs(trace.plan.intent, Intent.LOOKUP)
        self.assertEqual(trace.context[0].film.movie_id, "film-01")
        self.assertIs(trace.answer.status, Status.ANSWERED)

    def test_age_story_topic_reaches_search_and_generator(self):
        question = "Посоветуй фильм про дружбу героев разного возраста"
        film = self.films[0]
        hit = Hit(Chunk(film.source_id, film, film.synopsis), 0.9)
        catalog, retriever, generator = Mock(), Mock(), Mock()
        retriever.search.return_value = (hit,)
        # This test checks routing. Retrieval is not evidence of age diversity.
        generator.generate.return_value = Answer(Status.INSUFFICIENT, "Возраст героев не указан.")
        trace = SearchService(self.planner, retriever, catalog, generator).ask(question)
        retriever.search.assert_called_once_with(trace.plan, limit=10)
        catalog.execute.assert_not_called()
        generator.generate.assert_called_once_with(question, trace.plan, trace.context, None)
        self.assertIsNone(trace.plan.early_status)
        self.assertIs(trace.plan.intent, Intent.SEARCH)
        self.assertEqual(trace.retrieved, (hit,))

    def test_comparison_keeps_long_film_for_exact_operation(self):
        film = self.films[1]
        captured = []

        class Catalog:
            def execute(self, plan):
                captured.append(plan)
                return (Evidence(film, ()),), None

        class Generator:
            def generate(self, question, plan, context, exact_result):
                return Answer(Status.ANSWERED, "Карточка для сравнения", (film.source_id,))

        trace = SearchService(self.planner, None, Catalog(), Generator()).ask(
            "Подойдёт ли «Коралловая почта», если у меня 90 минут?")
        self.assertEqual(captured[0].time_budget_minutes, 90)
        self.assertEqual(trace.context[0].film.duration_minutes, 96)
        self.assertEqual(trace.retrieved, ())

    def test_natural_language_filter_is_forwarded_to_retrieval(self):
        captured = []

        class Retriever:
            def search(self, plan, *, limit):
                captured.append((plan, limit))
                return ()

        trace = SearchService(self.planner, Retriever(), None, None).ask(
            "Хочу мультфильм про море короче 90 минут")
        self.assertEqual(captured[0][0].filters.duration_minutes_lt, 90)
        self.assertEqual(captured[0][1], 10)
        self.assertEqual(trace.plan, captured[0][0])
        self.assertIs(trace.answer.status, Status.INSUFFICIENT)

    def test_cli_plan_is_offline_and_does_not_load_env_or_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid_env = Path(directory) / "invalid.env"
            invalid_env.write_text("THIS IS INVALID", encoding="utf-8")
            output = io.StringIO()
            with patch("sys.argv", ["cartoon-rag", "--env-file", str(invalid_env), "plan",
                                     "Хочу про море короче 90 минут"]), \
                    patch("cartoon_rag.bootstrap.build_search", side_effect=AssertionError("No provider")), \
                    redirect_stdout(output):
                main()
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["intent"], "search")
            self.assertEqual(plan["filters"]["duration_minutes_lt"], 90)
            self.assertIsNone(plan["early_status"])


if __name__ == "__main__":
    unittest.main()
