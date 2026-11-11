"""Regressions: stage isolation, honest coverage and corrupt catalog rejection."""

import io
import json
from contextlib import redirect_stdout
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

from cartoon_rag.application import AnswerService
from cartoon_rag.domain import Answer, Chunk, Evidence, GenerationUnavailable, Hit, PreparedQuery, QueryPlan, Status, Trace
from cartoon_rag.evaluation import Case, ReviewedSemanticJudge
from cartoon_rag.experiments import run_evaluation
from cartoon_rag.ingestion import load_catalog
from cartoon_rag.quality import plan_fields, stage_report
from cartoon_rag.snapshots import catalog_fingerprint, fingerprint


ROOT = Path(__file__).resolve().parents[1]


class CatalogValidationTest(unittest.TestCase):
    def test_rejects_invalid_data_at_load_boundary(self):
        row = json.loads((ROOT / 'catalog.json').read_text())[0]
        malformed = [None, {}, [None], [{**row, 'extra': 1}],
                     [{key: value for key, value in row.items() if key != 'year'}]]
        for field, value in [('year', True), ('year', '1990'), ('year', 0),
                             ('duration_minutes', False), ('duration_minutes', 1.5),
                             ('version', True), ('version', -1), ('active', 1),
                             ('synopsis', '  '), ('title', None), ('movie_id', []),
                             ('source_id', 3), ('tags', 'sea'), ('tags', [1]), ('tags', [' '])]:
            malformed.append([{**row, field: value}])
        malformed += [[row, row], [row, {**row, 'movie_id': 'another'}]]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'bad.json'
            for data in malformed:
                with self.subTest(data=data):
                    path.write_text(json.dumps(data))
                    with self.assertRaises(ValueError):
                        load_catalog(path)
            path.write_text('[{"movie_id":"a","movie_id":"b"}]')
            with self.assertRaisesRegex(ValueError, 'Duplicate catalog JSON field'):
                load_catalog(path)

    def test_versions_and_empty_snapshot_remain_supported(self):
        row = json.loads((ROOT / 'catalog.json').read_text())[0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'valid.json'
            path.write_text(json.dumps([row, {**row, 'version': 2, 'active': False}]))
            self.assertEqual(len(load_catalog(path)), 2)
            path.write_text('[]')
            self.assertEqual(load_catalog(path), ())


class QualityStageTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.film = load_catalog(ROOT / 'catalog.json')[0]
        self.case = Case('whale', 'Ищу мультфильм про кита', Status.ANSWERED,
                         (self.film.source_id,), QueryPlan('').filters, 'Сюжет и решение проблемы.')
        chunk = Chunk(self.film.source_id, self.film, self.film.synopsis)
        self.prepared = PreparedQuery(self.case.question, QueryPlan(self.case.question),
                                      (Hit(chunk, 1.0),), (Evidence(self.film, (chunk,)),), None)
        self.gold = {'schema_version': 1, 'version': 'test-v1',
                     'catalog_sha256': catalog_fingerprint(load_catalog(ROOT / 'catalog.json')),
                     'case_set_sha256': fingerprint([asdict(self.case)]),
                     'expectations': {'whale': {'plan': plan_fields(self.prepared.plan), 'exact_result': None}}}

    def run_report(self, *, failure=False, context_missing=False, negative_review=False):
        prepared = replace(self.prepared, context=()) if context_missing else self.prepared
        trace = Trace(prepared.question, prepared.plan, prepared.retrieved, prepared.context,
                      prepared.exact_result, Answer(Status.INSUFFICIENT, 'Недостаточно данных')
                      if context_missing else Answer(Status.ANSWERED, 'Подтверждённый ответ', (self.film.source_id,)))
        service = Mock()
        service.prepare.return_value = prepared
        if failure:
            service.answer.side_effect = GenerationUnavailable('503', http_status=503)
        else:
            service.answer.return_value = trace
        run_dir = self.root / 'run'
        run_evaluation((self.case,), lambda: (service, {}), run_dir, label='test')
        decisions = [] if failure else [{
            'case_id': self.case.id, 'case_sha256': fingerprint(asdict(self.case)),
            'trace_sha256': fingerprint(asdict(trace)), 'explanation': 'Test decision',
            **{name: not negative_review for name in ReviewedSemanticJudge.checks}}]
        reviews = self.root / 'reviews.json'
        reviews.write_text(json.dumps({'reviewer': 'test', 'decisions': decisions}))
        gold = self.root / 'gold.json'
        gold.write_text(json.dumps(self.gold))
        before = {p: p.read_bytes() for p in run_dir.rglob('*') if p.is_file()}
        report = stage_report(run_dir, gold, ROOT / 'catalog.json', reviews)
        self.assertEqual(before, {p: p.read_bytes() for p in run_dir.rglob('*') if p.is_file()})
        self.assertEqual(report, stage_report(run_dir, gold, ROOT / 'catalog.json', reviews))
        return report

    def test_failure_does_not_erase_retrieval_or_pass_generation(self):
        report = self.run_report(failure=True)
        self.assertEqual(report['metrics']['retrieval_recall_at_k']['observed_mean'], 1.0)
        self.assertEqual(report['metrics']['answer_status']['missing'], 1)
        self.assertIsNone(report['metrics']['answer_status']['observed_mean'])
        self.assertEqual(report['metrics']['completion']['observed_mean'], 0.0)
        self.assertEqual(report['metrics']['exact_result']['not_applicable'], 1)

    def test_context_loss_is_distinct_from_retrieval_failure(self):
        report = self.run_report(context_missing=True)
        self.assertEqual(report['metrics']['retrieval_recall_at_k']['observed_mean'], 1.0)
        self.assertEqual(report['metrics']['context_recall']['observed_mean'], 0.0)
        self.assertEqual(report['metrics']['answer_status']['observed_mean'], 0.0)

    def test_correct_status_and_references_do_not_prove_semantics(self):
        report = self.run_report(negative_review=True)
        self.assertEqual(report['metrics']['answer_status']['observed_mean'], 1.0)
        self.assertEqual(report['metrics']['answer_contract']['observed_mean'], 1.0)
        self.assertEqual(report['metrics']['answer_semantics']['observed_mean'], 0.0)

    def test_context_chunk_with_stale_card_fails_catalog_check_after_roundtrip(self):
        stale = replace(self.film, version=2, duration_minutes=92)
        chunk = replace(self.prepared.context[0].chunks[0], film=stale, text='Посторонний текст')
        self.prepared = replace(self.prepared, context=(Evidence(self.film, (chunk,)),))
        report = self.run_report()
        self.assertEqual(report['metrics']['catalog_evidence']['fail'], 1)
        self.assertEqual(report['metrics']['retrieval_recall_at_k']['observed_mean'], 1.0)
        self.assertEqual(report['metrics']['context_recall']['observed_mean'], 1.0)
        self.assertEqual(report['metrics']['answer_contract']['observed_mean'], 1.0)

    def test_context_chunk_from_another_active_movie_fails_parent_check(self):
        other = load_catalog(ROOT / 'catalog.json')[1]
        chunk = Chunk(other.source_id, other, other.synopsis)
        self.prepared = replace(self.prepared, context=(Evidence(self.film, (chunk,)),))
        report = self.run_report()
        self.assertEqual(report['metrics']['catalog_evidence']['fail'], 1)

    def test_invalid_context_chunk_is_detected_without_completed_answer(self):
        stale = replace(self.film, version=2, duration_minutes=92)
        chunk = replace(self.prepared.context[0].chunks[0], film=stale)
        self.prepared = replace(self.prepared, context=(Evidence(self.film, (chunk,)),))
        report = self.run_report(failure=True)
        self.assertEqual(report['metrics']['catalog_evidence']['fail'], 1)
        self.assertEqual(report['metrics']['answer_contract']['missing'], 1)

    def test_catalog_evidence_accepts_valid_parent_without_chunks(self):
        self.prepared = replace(self.prepared, context=(Evidence(self.film, ()),))
        report = self.run_report()
        self.assertEqual(report['metrics']['catalog_evidence']['pass'], 1)

    def test_gold_cannot_be_used_with_another_catalog(self):
        self.gold['catalog_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'another catalog'):
            self.run_report()

    def test_cli_stage_report_is_offline(self):
        self.run_report()
        from cartoon_rag.cli import main
        from cartoon_rag.settings import Settings

        argv = ['cartoon-rag', 'evaluate', 'stages', '--run-dir', str(self.root / 'run'),
                '--expectations', str(self.root / 'gold.json'), '--catalog', str(ROOT / 'catalog.json'),
                '--reviews', str(self.root / 'reviews.json'), '--output', str(self.root / 'report.json')]
        with patch('sys.argv', argv), redirect_stdout(io.StringIO()), \
                patch.object(Settings, 'load', side_effect=AssertionError('Offline command')):
            main()
        self.assertEqual(json.loads((self.root / 'report.json').read_text())['report_version'], 'stage-quality-v1.1')

    def test_answer_service_needs_only_generator_and_prepared_evidence(self):
        generator = Mock()
        generator.generate.return_value = Answer(Status.ANSWERED, 'Ответ', (self.film.source_id,))
        trace = AnswerService(generator).answer(self.prepared)
        self.assertEqual(trace.retrieved, self.prepared.retrieved)
        generator.generate.assert_called_once_with(self.case.question, self.prepared.plan,
                                                   self.prepared.context, None)


if __name__ == '__main__':
    unittest.main()
