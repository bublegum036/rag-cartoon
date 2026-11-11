"""Year constraints: planning, exact operations, prefiltering and archive compatibility."""

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

from cartoon_rag.adapters.gemini_planner import GeminiQueryPlanner, InvalidPlan, decode_plan
from cartoon_rag.adapters.json_catalog import JsonExactCatalog
from cartoon_rag.adapters.qdrant import QdrantIndex, qdrant_filter
from cartoon_rag.application import SearchService
from cartoon_rag.bootstrap import build_planner
from cartoon_rag.cli import main
from cartoon_rag.domain import Answer, Chunk, Evidence, Filters, GroundingError, Hit, Intent, PreparedQuery, QueryPlan, Status, Trace
from cartoon_rag.evaluation import Case, ReviewedSemanticJudge, load_cases
from cartoon_rag.experiments import load_run, run_evaluation, score_run
from cartoon_rag.ingestion import load_catalog
from cartoon_rag.serialization import archive_data
from cartoon_rag.snapshots import fingerprint


ROOT = Path(__file__).resolve().parents[1]


class YearFilterTest(unittest.TestCase):
    def setUp(self):
        self.films = load_catalog(ROOT / 'catalog.json')
        self.planner = build_planner(self.films)

    def test_rule_boundaries_exact_year_range_and_duration_intersection(self):
        cases = (
            ('Посоветуй фильм про море, выпущенный не раньше 1990 года.', 1990, None),
            ('про море не позже 1990 года', None, 1990),
            ('про море не позднее 1990 года', None, 1990),
            ('про море ранее 1990 года', None, 1989),
            ('про море позднее 1990 года', 1991, None),
            ('про море после 1990 года', 1991, None),
            ('про море раньше 1990', None, 1989),
            ('про море до 1990 года', None, 1989),
            ('про море в 1990 году', 1990, 1990),
            ('мультфильм 1990 года', 1990, 1990),
            ('про море с 1980 по 1990 год', 1980, 1990),
            ('про море не раньше 1980 года и не позже 1990 года', 1980, 1990),
            ('про море не раньше 1980 года и после 1990 года', 1991, None),
        )
        for question, lower, upper in cases:
            with self.subTest(question=question):
                plan = self.planner.plan(question)
                self.assertIsNone(plan.early_status)
                self.assertIs(plan.intent, Intent.SEARCH)
                self.assertEqual((plan.filters.year_gte, plan.filters.year_lte), (lower, upper))
                self.assertNotRegex(plan.semantic_query, r'\d')
        plan = self.planner.plan('про море не раньше 1990 года и короче 90 минут')
        self.assertEqual(plan.filters, Filters(duration_minutes_lt=90, year_gte=1990))

    def test_lower_year_operator_is_consumed_whole_not_as_an_exact_year(self):
        films = load_catalog(ROOT / 'data/catalog-expanded.json')
        planner, catalog = build_planner(films), JsonExactCatalog(films)
        expected = tuple(sorted((film for film in films if film.year >= 1990), key=lambda film: film.movie_id))
        self.assertEqual(len(expected), 20)
        for operator in ('с', 'начиная с', 'не ранее', 'не раньше'):
            with self.subTest(operator=operator):
                plan = planner.plan(f'про море {operator} 1990 года')
                self.assertIsNone(plan.early_status)
                self.assertIs(plan.intent, Intent.SEARCH)
                self.assertEqual(plan.filters, Filters(year_gte=1990))
                self.assertEqual(plan.semantic_query, 'про море')
                # Check actual membership, not just the representation of the plan.
                self.assertEqual(tuple(sorted((f for f in films if plan.filters.matches(f)),
                                              key=lambda f: f.movie_id)), expected)
                count_plan = planner.plan(f'Сколько фильмов {operator} 1990 года?')
                self.assertIs(count_plan.intent, Intent.COUNT)
                evidence, result = catalog.execute(count_plan)
                self.assertEqual(result.value, 20)
                self.assertEqual(tuple(item.film for item in evidence), expected)
                self.assertEqual(result.source_ids, tuple(f.source_id for f in expected))

    def test_invalid_ranges_units_negations_or_alternatives_require_clarification(self):
        for question in ('про море с 2000 по 1990 год', 'про море после 1990 года и до 1991 года',
                         'про море не после 1990 года', 'про море не раньше 0000 года',
                         'про море после 0000 года', 'про море в 0000 году',
                         'про море не раньше 1990 года или не позже 2000 года',
                         'про море после 1990 секунд', 'про море после 1990.5 года',
                         'про море после 19900 года', 'про море между 1980 и 1990 годами',
                         'про море не раньше двухтысячного года', 'про море после прошлого года', 'про море от 1990 года',
                         'про море новее 1990 года', 'про море старше 1990 года',
                         'про море не моложе 1990 года', 'про море с 1990 года по'):
            with self.subTest(question=question):
                self.assertIs(self.planner.plan(question).early_status, Status.CLARIFY)
        for value in (True, '1990', 1990.0, 0, 10000):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Filters(year_gte=value)
        with self.assertRaises(ValueError):
            Filters(year_gte=2000, year_lte=1990)

    def test_exact_count_uses_full_year_and_duration_filtered_catalog(self):
        films = load_catalog(ROOT / 'data/catalog-expanded.json')
        planner, exact = build_planner(films), JsonExactCatalog(films)
        plan = planner.plan('Сколько фильмов с 1980 по 1990 год и короче 90 минут?')
        self.assertIs(plan.intent, Intent.COUNT)
        evidence, result = exact.execute(plan)
        expected = sorted((f for f in films if 1980 <= f.year <= 1990 and f.duration_minutes < 90),
                          key=lambda f: f.movie_id)
        self.assertGreater(len(expected), 1)
        self.assertEqual(result.value, len(expected))
        self.assertEqual(result.source_ids, tuple(f.source_id for f in expected))
        self.assertEqual([e.film for e in evidence], expected)
        self.assertEqual(exact.execute(planner.plan('Сколько фильмов в 1900 году?'))[1].value, 0)

    def test_named_lookup_keeps_year_constraint(self):
        plan = self.planner.plan('Что происходит в «Маяк для кита», выпущенном не раньше 2021 года?')
        self.assertIs(plan.intent, Intent.LOOKUP)
        self.assertEqual(len(JsonExactCatalog(self.films).execute(plan)[0]), 1)
        excluded = replace(plan, filters=replace(plan.filters, year_gte=2022))
        self.assertEqual(JsonExactCatalog(self.films).execute(excluded), ((), None))

    def test_qdrant_body_combines_year_and_duration_before_top_k(self):
        from test_search_adapters import RecordingTransport, StubEmbedder
        transport = RecordingTransport()
        index = QdrantIndex(transport, 'test', StubEmbedder())
        filters = Filters(duration_minutes_lt=90, year_gte=1980, year_lte=1990)
        index.search(QueryPlan('море', filters=filters), limit=1)
        body = transport.calls[-1][2]
        self.assertEqual(body['limit'], 1)
        self.assertEqual(body['filter'], {'must': [
            {'key': 'active', 'match': {'value': True}},
            {'key': 'duration_minutes', 'range': {'lt': 90}},
            {'key': 'year', 'range': {'gte': 1980, 'lte': 1990}},
        ]})

    def test_application_rejects_retriever_violating_year_bound(self):
        film = replace(self.films[0], year=1989)
        retriever = Mock()
        retriever.search.return_value = (Hit(Chunk(film.source_id, film, film.synopsis), 1.0),)
        generator = Mock()
        service = SearchService(self.planner, retriever, JsonExactCatalog((film,)), generator)
        with self.assertRaises(GroundingError):
            service.ask('про море не раньше 1990 года')
        generator.generate.assert_not_called()

    def test_gemini_year_schema_decodes_and_validates_without_provider_calls(self):
        value = {'intent': 'search', 'semantic_query': 'море', 'time_budget_minutes': None,
                 'early_status': None, 'early_reason': None,
                 'filters': {'duration_minutes_lt': None, 'duration_minutes_lte': None, 'movie_ids': [],
                             'year_gte': 1990, 'year_lte': None}}
        transport = Mock()
        transport.generate.return_value = {'candidates': [{'finishReason': 'STOP', 'content': {
            'parts': [{'text': json.dumps(value)}]}}]}
        planner = GeminiQueryPlanner(transport, {f.movie_id: f.title for f in self.films})
        plan = planner.plan('Посоветуй фильм про море, выпущенный не раньше 1990 года.')
        self.assertIsNone(plan.early_status)
        self.assertEqual(plan.filters.year_gte, 1990)
        for bounds in ({'year_gte': True}, {'year_gte': 1990.5}, {'year_lte': '1990'},
                       {'year_gte': 2000, 'year_lte': 1990}):
            with self.subTest(bounds=bounds), self.assertRaises(InvalidPlan):
                decode_plan({**value, 'filters': {**value['filters'], **bounds}}, planner.titles)

    def test_cli_plan_exact_and_explicit_year_search_flags(self):
        for argv in (['plan', 'про море не раньше 1990 года'],
                     ['exact', 'Сколько фильмов не раньше 2021 года?']):
            out = io.StringIO()
            with patch('sys.argv', ['cartoon-rag', *argv]), redirect_stdout(out):
                main()
            row = json.loads(out.getvalue())
            plan = row['plan'] if 'plan' in row else row
            self.assertIsNotNone(plan['filters']['year_gte'])
        with patch('sys.argv', ['cartoon-rag', 'search', 'море', '--year-gte', '1980', '--year-lte', '1990']), \
                patch('cartoon_rag.bootstrap.build_search') as factory, redirect_stdout(io.StringIO()):
            from cartoon_rag.settings import Settings
            retriever = Mock()
            retriever.search.return_value = ()
            factory.return_value = (Mock(), retriever)
            with patch.object(Settings, 'load', return_value=Settings()):
                main()
            self.assertEqual(retriever.search.call_args.args[0].filters, Filters(year_gte=1980, year_lte=1990))

    def test_schema3_year_metric_and_review_binding_survive_roundtrip(self):
        film = self.films[0]
        filters = Filters(year_gte=2021, year_lte=2021)
        case = Case('year', 'про море в 2021 году', Status.ANSWERED, (film.source_id,), filters, 'Маяк для кита')
        prepared = PreparedQuery(case.question, QueryPlan('море', filters=filters), (), (Evidence(film, ()),), None)
        trace = Trace(prepared.question, prepared.plan, prepared.retrieved, prepared.context, None,
                      Answer(Status.ANSWERED, film.title, (film.source_id,)))
        service = Mock()
        service.prepare.return_value, service.answer.return_value = prepared, trace
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            run_evaluation((case,), lambda: (service, {}), output, label='year')
            reviews = output / 'reviews.json'
            reviews.write_text(json.dumps({'reviewer': 'unit-test', 'decisions': [{
                'case_id': case.id, 'case_sha256': fingerprint(asdict(case)),
                'trace_sha256': fingerprint(asdict(trace)), 'explanation': 'Test-only decision',
                **{key: True for key in ReviewedSemanticJudge.checks}}]}))
            result = score_run(output, reviews)
            self.assertEqual(result['strict_filter_cases'], 1)
            self.assertEqual(result['answer_acceptance_rate'], 1.0)
            self.assertEqual(json.loads((output / 'manifest.json').read_text())['artifact_schema_version'], 3)
            data = json.loads((output / 'year/prepared.json').read_text())
            self.assertEqual(data['plan']['filters']['year_gte'], 2021)

    def test_schema1_cannot_introduce_year_constraints_even_without_reviews(self):
        source = ROOT / 'artifacts/ab-A-2026-10-25'
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'run'
            shutil.copytree(source, target)
            trace_path = target / 'sea-short/trace.json'
            trace = json.loads(trace_path.read_text())
            trace['plan']['filters']['year_gte'] = 1990
            trace_path.write_text(json.dumps(trace))
            ledger = json.loads((target / 'results.json').read_text())
            ledger[0]['trace_sha256'] = fingerprint(trace)
            (target / 'results.json').write_text(json.dumps(ledger))
            (target / 'reviews.json').write_text(json.dumps({'reviewer': 'test', 'decisions': []}))
            with self.assertRaisesRegex(ValueError, 'require artifact schema 3'):
                load_run(target, target / 'reviews.json')

    def test_legacy_archives_are_verified_using_original_filters_and_hashes(self):
        names = ('ab-A-2026-10-25', 'ab-B-2026-10-25', 'expanded-run-v2-2026-10-26',
                 'llm-planner-2026-11-12/A-rules', 'llm-planner-2026-11-12/B-gemini',
                 'llm-planner-2026-11-12/paraphrases-rules', 'llm-planner-2026-11-12/paraphrases-gemini',
                 'llm-planner-2026-11-12/early-recheck-v1.1')
        with tempfile.TemporaryDirectory() as directory:
            for i, name in enumerate(names):
                with self.subTest(name=name):
                    source = ROOT / 'artifacts' / name
                    target = Path(directory) / str(i)
                    shutil.copytree(source, target)
                    expected = json.loads((source / 'metrics.json').read_text())
                    self.assertEqual(score_run(target, target / 'reviews.json'), expected)
        for schema in (1, 2):
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                archive_data({'filters': {'year_gte': 1990}}, schema)


if __name__ == '__main__':
    unittest.main()
