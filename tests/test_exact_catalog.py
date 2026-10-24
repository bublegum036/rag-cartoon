import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

from cartoon_rag.adapters.json_catalog import JsonExactCatalog
from cartoon_rag.application import SearchService, execute_exact
from cartoon_rag.bootstrap import build_exact_catalog, build_planner
from cartoon_rag.cli import main
from cartoon_rag.domain import (
    Answer, Evidence, ExactResult, Filters, GroundingError, Intent, QueryPlan,
    RetrievalUnavailable, Status,
)
from cartoon_rag.ingestion import load_catalog


ROOT = Path(__file__).resolve().parents[1]


class ExactCatalogTest(unittest.TestCase):
    def setUp(self):
        self.films = load_catalog(ROOT / "catalog.json")
        self.catalog = build_exact_catalog(self.films)

    def test_lookup_returns_full_card_without_synthetic_chunks(self):
        evidence, result = self.catalog.execute(QueryPlan(
            "Сколько минут длится фильм?", intent=Intent.LOOKUP,
            filters=Filters(movie_ids=("film-01",)),
        ))
        self.assertEqual(evidence, (Evidence(self.films[0], ()),))
        self.assertIsNone(result)
        self.assertEqual(evidence[0].film.source_id, "film-01#plot")

    def test_count_uses_complete_filtered_catalog(self):
        for filters, expected in ((Filters(), 12), (Filters(duration_minutes_lt=90), 9),
                                  (Filters(duration_minutes_lt=40), 0)):
            with self.subTest(filters=filters):
                evidence, result = self.catalog.execute(QueryPlan("", intent=Intent.COUNT, filters=filters))
                self.assertEqual(result.value, expected)
                self.assertEqual(result.unit, "movies")
                self.assertEqual(len(evidence), expected)
                self.assertEqual(result.source_ids, tuple(item.film.source_id for item in evidence))

    def test_strict_boundary_and_intersection_with_id_filter(self):
        for filters, expected in (
            (Filters(duration_minutes_lt=78), 4), (Filters(duration_minutes_lte=78), 5),
            (Filters(duration_minutes_lt=90, duration_minutes_lte=78), 5),
            (Filters(duration_minutes_lt=90, movie_ids=("film-01", "film-02", "film-01")), 1),
        ):
            with self.subTest(filters=filters):
                evidence, result = self.catalog.execute(QueryPlan("", intent=Intent.COUNT, filters=filters))
                self.assertEqual(result.value, expected)
                self.assertTrue(all(filters.matches(item.film) for item in evidence))
                self.assertEqual(len({item.film.movie_id for item in evidence}), len(evidence))

    def test_zero_count_is_distinct_from_missing_lookup_or_comparison(self):
        empty = JsonExactCatalog(())
        evidence, result = empty.execute(QueryPlan("", intent=Intent.COUNT))
        self.assertEqual(evidence, ())
        self.assertEqual(result, ExactResult(0, "movies", ()))
        for intent in (Intent.LOOKUP, Intent.COMPARE_DURATION):
            with self.subTest(intent=intent):
                plan = QueryPlan("", intent=intent, filters=Filters(movie_ids=("missing",)),
                                 time_budget_minutes=90 if intent is Intent.COMPARE_DURATION else None)
                self.assertEqual(self.catalog.execute(plan), ((), None))

    def test_duration_comparison_is_signed_and_has_original_source(self):
        for budget, expected in ((90, 6), (96, 0), (100, -4)):
            with self.subTest(budget=budget):
                evidence, result = self.catalog.execute(QueryPlan(
                    "", intent=Intent.COMPARE_DURATION, filters=Filters(movie_ids=("film-02",)),
                    time_budget_minutes=budget,
                ))
                self.assertEqual(result, ExactResult(expected, "minutes", ("film-02#plot",)))
                self.assertEqual(evidence[0].film.duration_minutes, 96)

    def test_latest_revision_and_inactive_tombstone(self):
        original = self.films[0]
        updated = replace(original, version=2, duration_minutes=92)
        catalog = JsonExactCatalog((updated, original))
        plan = QueryPlan("", intent=Intent.LOOKUP, filters=Filters(movie_ids=("film-01",)))
        self.assertEqual(catalog.execute(plan)[0][0].film, updated)
        count = QueryPlan("", intent=Intent.COUNT, filters=Filters(duration_minutes_lt=90))
        self.assertEqual(catalog.execute(count)[1].value, 0)
        compare = replace(plan, intent=Intent.COMPARE_DURATION, time_budget_minutes=90)
        self.assertEqual(catalog.execute(compare)[1].value, 2)
        retired = JsonExactCatalog((original, updated, replace(updated, version=3, active=False)))
        self.assertEqual(retired.execute(plan), ((), None))
        self.assertEqual(retired.execute(QueryPlan("", intent=Intent.COUNT))[1].value, 0)

    def test_snapshot_does_not_follow_mutation_of_input_sequence(self):
        rows = [self.films[0]]
        catalog = JsonExactCatalog(rows)
        rows.clear()
        self.assertEqual(catalog.execute(QueryPlan("", intent=Intent.COUNT))[1].value, 1)

    def test_duplicate_revisions_and_source_ids_are_rejected(self):
        for rows in ((self.films[0], self.films[0]),
                     (self.films[0], replace(self.films[1], source_id=self.films[0].source_id))):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    JsonExactCatalog(rows)

    def test_invalid_and_non_exact_plans_are_rejected(self):
        plans = [QueryPlan("море"), QueryPlan("", intent=Intent.LOOKUP),
                 QueryPlan("", intent=Intent.LOOKUP, filters=Filters(movie_ids=("film-01", "film-02"))),
                 QueryPlan("", intent=Intent.COUNT, time_budget_minutes=90),
                 QueryPlan("", intent=Intent.COUNT, early_status=Status.CLARIFY, early_text="Уточните")]
        plans.extend(QueryPlan("", intent=Intent.COUNT, filters=Filters(duration_minutes_lt=value))
                     for value in (0, -1, True, 1.5))
        plans.extend(QueryPlan("", intent=Intent.COMPARE_DURATION,
                               filters=Filters(movie_ids=("film-01",)), time_budget_minutes=value)
                     for value in (0, -1, True, 1.5))
        for plan in plans:
            with self.subTest(plan=plan):
                with self.assertRaises(ValueError):
                    self.catalog.execute(plan)

    def test_comparison_rejects_duration_filter_instead_of_hiding_long_film(self):
        plan = QueryPlan("", intent=Intent.COMPARE_DURATION,
                         filters=Filters(movie_ids=("film-02",), duration_minutes_lt=90),
                         time_budget_minutes=90)
        with self.assertRaisesRegex(ValueError, "without duration filters"):
            self.catalog.execute(plan)


class ExactIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.films = load_catalog(ROOT / "catalog.json")
        self.planner = build_planner(self.films)
        self.catalog = build_exact_catalog(self.films)

    def test_count_is_not_truncated_by_retrieval_or_context_limits(self):
        retriever, generator = Mock(), Mock()
        generator.generate.return_value = Answer(Status.ANSWERED, "12 фильмов.",
                                                tuple(film.source_id for film in self.films))
        service = SearchService(self.planner, retriever, self.catalog, generator,
                                retrieval_k=1, context_movies=1)
        trace = service.ask("Сколько фильмов в каталоге?")
        retriever.search.assert_not_called()
        self.assertEqual(trace.exact_result.value, 12)
        self.assertEqual(len(trace.context), 12)
        self.assertEqual(trace.retrieved, ())

    def test_zero_count_reaches_generator_with_no_fabricated_sources(self):
        generator = Mock()
        generator.generate.return_value = Answer(Status.ANSWERED, "0 фильмов.")
        trace = SearchService(self.planner, None, self.catalog, generator).ask(
            "Сколько фильмов короче 40 минут?")
        generator.generate.assert_called_once()
        self.assertEqual(trace.context, ())
        self.assertEqual(trace.exact_result, ExactResult(0, "movies", ()))
        self.assertIs(trace.answer.status, Status.ANSWERED)

    def test_planner_comparison_computes_six_minutes_before_generation(self):
        generator = Mock()
        generator.generate.return_value = Answer(Status.ANSWERED, "На 6 минут дольше.", ("film-02#plot",))
        question = "Подойдёт ли «Коралловая почта», если у меня 90 минут?"
        trace = SearchService(self.planner, None, self.catalog, generator).ask(question)
        self.assertEqual(trace.exact_result.value, 6)
        self.assertEqual(trace.context[0].film.duration_minutes, 96)
        generator.generate.assert_called_once_with(question, trace.plan, trace.context, trace.exact_result)

    def test_missing_lookup_does_not_call_generator(self):
        planner, generator = Mock(), Mock()
        planner.plan.return_value = QueryPlan("", intent=Intent.LOOKUP,
                                             filters=Filters(movie_ids=("missing",)))
        trace = SearchService(planner, None, self.catalog, generator).ask("Длительность?")
        generator.generate.assert_not_called()
        self.assertIs(trace.answer.status, Status.INSUFFICIENT)

    def test_exact_use_case_preserves_early_response_without_catalog_call(self):
        catalog = Mock()
        plan = self.planner.plan("Сколько он длится?")
        result = execute_exact(catalog, plan)
        catalog.execute.assert_not_called()
        self.assertEqual(result.context, ())
        self.assertEqual(result.plan.early_status, Status.CLARIFY)

    def test_exact_use_case_rejects_invalid_provider_evidence(self):
        film = self.films[0]
        plan = QueryPlan("", intent=Intent.COUNT, filters=Filters(duration_minutes_lt=90))
        for evidence, result in (
            ((Evidence(film, ()),), ExactResult(1, "movies", ("invented",))),
            ((Evidence(film, ()), Evidence(film, ())), ExactResult(2, "movies", (film.source_id,))),
            ((Evidence(self.films[1], ()),), None),
        ):
            catalog = Mock()
            catalog.execute.return_value = evidence, result
            with self.subTest(evidence=evidence, result=result):
                with self.assertRaises(GroundingError):
                    execute_exact(catalog, plan)

    def test_catalog_failure_is_not_insufficient(self):
        catalog, generator = Mock(), Mock()
        catalog.execute.side_effect = RetrievalUnavailable("timeout")
        with self.assertRaises(RetrievalUnavailable):
            SearchService(self.planner, None, catalog, generator).ask("Сколько фильмов в каталоге?")
        generator.generate.assert_not_called()


class ExactCLITest(unittest.TestCase):
    def invoke(self, args):
        out, err = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["cartoon-rag", *args]), redirect_stdout(out), redirect_stderr(err):
            main()
        return json.loads(out.getvalue())

    def test_exact_is_offline_and_loads_catalog_once(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid_env = Path(directory) / "invalid.env"
            invalid_env.write_text("INVALID ENV", encoding="utf-8")
            with patch("cartoon_rag.cli.load_catalog", wraps=load_catalog) as loader, \
                    patch("cartoon_rag.bootstrap.build_search", side_effect=AssertionError("No provider")):
                output = self.invoke(["--env-file", str(invalid_env), "exact",
                                      "Сколько минут длится „Маяк для кита“?"])
                loader.assert_called_once_with(Path("catalog.json"))
            self.assertEqual(output["plan"]["intent"], "lookup")
            self.assertEqual(output["context"][0]["film"]["duration_minutes"], 78)
            self.assertIsNone(output["exact_result"])

    def test_exact_uses_latest_revision_from_custom_catalog(self):
        film = load_catalog(ROOT / "catalog.json")[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps([asdict(film), asdict(replace(film, version=2, duration_minutes=92))]))
            output = self.invoke(["exact", "Подойдёт ли «Маяк для кита», если у меня 90 минут?",
                                  "--catalog", str(path)])
        self.assertEqual(output["exact_result"]["value"], 2)
        self.assertEqual(output["context"][0]["film"]["version"], 2)

    def test_early_and_semantic_questions_are_not_executed_as_counts(self):
        output = self.invoke(["exact", "Сколько он длится?"])
        self.assertEqual(output["plan"]["early_status"], "clarify")
        self.assertEqual(output["context"], [])
        self.assertIsNone(output["exact_result"])
        err = io.StringIO()
        with patch("sys.argv", ["cartoon-rag", "exact", "Хочу про море"]), \
                redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            main()
        self.assertEqual(caught.exception.code, 1)
        self.assertEqual(json.loads(err.getvalue())["error"]["type"], "ValueError")

    def test_missing_file_is_a_technical_error_not_absence_of_films(self):
        with tempfile.TemporaryDirectory() as directory:
            err = io.StringIO()
            with patch("sys.argv", ["cartoon-rag", "exact", "Сколько фильмов в каталоге?",
                                     "--catalog", str(Path(directory) / "missing.json")]), \
                    redirect_stderr(err), self.assertRaises(SystemExit) as caught:
                main()
        self.assertEqual(caught.exception.code, 1)
        error = json.loads(err.getvalue())
        self.assertEqual(error["error"]["type"], "FileNotFoundError")
        self.assertNotIn("status", error)


if __name__ == "__main__":
    unittest.main()
