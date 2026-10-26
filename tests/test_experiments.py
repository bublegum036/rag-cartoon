import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

from cartoon_rag.cli import main
from cartoon_rag.application import SearchService
from cartoon_rag.adapters.json_catalog import JsonExactCatalog
from cartoon_rag.domain import (
    Answer, Chunk, Evidence, Filters, GenerationUnavailable, Hit, Intent, PreparedQuery,
    QueryPlan, RetrievalUnavailable, Status, Trace,
)
from cartoon_rag.evaluation import (
    Case, ReviewedSemanticJudge, strict_filters_compliant, trace_from_dict,
)
from cartoon_rag.experiments import run_evaluation, score_run
from cartoon_rag import experiments
from cartoon_rag.ingestion import load_catalog
from cartoon_rag.settings import Settings
from cartoon_rag.snapshots import fingerprint


ROOT = Path(__file__).resolve().parents[1]


class ExperimentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "run"
        self.film = load_catalog(ROOT / "catalog.json")[0]
        self.cases = (
            Case("whale", "Ищу кита", Status.ANSWERED, (self.film.source_id,),
                 Filters(duration_minutes_lt=90), "Девочка и кит чинят маяк, менее 90 минут."),
            Case("clarify", "Сколько он длится?", Status.CLARIFY, (), Filters(), "Уточнить фильм."),
        )
        chunk = Chunk(self.film.source_id + ":part-1", self.film, self.film.synopsis)
        self.traces = {
            "whale": Trace(self.cases[0].question, QueryPlan("кит", filters=self.cases[0].filters),
                           (Hit(chunk, 0.9),), (Evidence(self.film, (chunk,)),), None,
                           Answer(Status.ANSWERED, "Маяк для кита, 78 минут.", (self.film.source_id,))),
            "clarify": Trace(self.cases[1].question, QueryPlan("", early_status=Status.CLARIFY,
                                                               early_text="Укажите фильм"),
                             (), (), None, Answer(Status.CLARIFY, "Укажите фильм")),
        }
        self.service = Mock()
        def prepare(question):
            trace = next(self.traces[case.id] for case in self.cases if case.question == question)
            return PreparedQuery(trace.question, trace.plan, trace.retrieved, trace.context, trace.exact_result)

        self.service.prepare.side_effect = prepare
        self.service.answer.side_effect = lambda prepared: next(
            trace for trace in self.traces.values() if trace.question == prepared.question)
        self.factory = Mock(return_value=(self.service, {"snapshot_id": "test-snapshot", "api_key": "private-key"}))

    def run_cases(self, cases=None):
        return run_evaluation(cases or self.cases, self.factory, self.output, label="test-only", k=3)

    def reviews(self, cases=None):
        return {"reviewer": "test reviewer", "decisions": [
            {"case_id": case.id, "case_sha256": fingerprint(asdict(case)),
             "trace_sha256": fingerprint(asdict(self.traces[case.id])),
             **{name: True for name in ReviewedSemanticJudge.checks},
             "explanation": "Test-only structured decision after checking status, rubric, facts and references."}
            for case in (cases or self.cases)
        ]}

    def write_reviews(self, document=None):
        path = self.root / "reviews.json"
        path.write_text(json.dumps(document or self.reviews()), encoding="utf-8")
        return path

    def test_separate_artifacts_pending_acceptance_and_parent_card_recall(self):
        metrics = self.run_cases()
        self.assertIsNone(metrics["answer_acceptance_rate"])
        self.assertEqual(metrics["pending_reviews"], 2)
        self.assertEqual(metrics["recall_at_1"], 1.0)
        self.assertEqual(metrics["reference_cases"], 1)
        self.assertNotIn("private-key", (self.output / "manifest.json").read_text())
        self.assertEqual([call.args for call in self.service.prepare.call_args_list],
                         [(case.question,) for case in self.cases])
        self.assertEqual(self.service.answer.call_count, 2)
        self.assertEqual(metrics["prepared_cases"], 2)
        for case in self.cases:
            for name in ("trace", "prepared", "plan", "exact-result", "retrieved", "retrieved-ids", "context", "answer", "case"):
                self.assertTrue((self.output / case.id / (name + ".json")).exists())
            saved = json.loads((self.output / case.id / "trace.json").read_text())
            self.assertEqual(trace_from_dict(saved), self.traces[case.id])

    def test_complete_reviews_produce_acceptance_and_auditable_judgments(self):
        self.run_cases()
        metrics = score_run(self.output, self.write_reviews())
        self.assertEqual(metrics["answer_acceptance_rate"], 1.0)
        self.assertEqual(metrics["reviewed_cases"], 2)
        self.assertEqual(metrics["pending_reviews"], 0)
        judgment = json.loads((self.output / "whale/judgment.json").read_text())
        self.assertTrue(judgment["accepted"])
        self.assertIn("explanation", judgment["semantic"])

    def test_partial_reviews_do_not_claim_full_acceptance_rate(self):
        self.run_cases()
        metrics = score_run(self.output, self.write_reviews(self.reviews(self.cases[:1])))
        self.assertEqual(metrics["pending_reviews"], 1)
        self.assertIsNone(metrics["answer_acceptance_rate"])

    def test_rescoring_with_partial_reviews_does_not_leave_stale_pass(self):
        self.run_cases()
        score_run(self.output, self.write_reviews())
        metrics = score_run(self.output, self.write_reviews(self.reviews(self.cases[:1])))
        judgment = json.loads((self.output / "clarify/judgment.json").read_text())
        self.assertEqual(judgment["state"], "pending_review")
        self.assertIsNone(judgment["accepted"])
        self.assertEqual(metrics["pending_reviews"], 1)

    def test_review_is_bound_to_both_case_and_trace(self):
        judge = ReviewedSemanticJudge(self.reviews())
        with self.assertRaisesRegex(ValueError, "case/rubric"):
            judge.accepts(replace(self.cases[0], rubric="Other rubric"), self.traces["whale"])
        with self.assertRaisesRegex(ValueError, "another generated trace"):
            judge.accepts(self.cases[0], replace(self.traces["whale"], answer=Answer(Status.INSUFFICIENT, "Нет")))
        invalid = self.reviews()
        invalid["decisions"][0]["facts_supported"] = "false"
        with self.assertRaisesRegex(ValueError, "booleans"):
            ReviewedSemanticJudge(invalid)

    def test_separate_answer_cannot_be_changed_behind_trace(self):
        self.run_cases()
        (self.output / "whale/answer.json").write_text('{}')
        with self.assertRaisesRegex(ValueError, "Separate run artifacts"):
            score_run(self.output, self.write_reviews())

    def test_wrong_status_cannot_be_overridden_by_positive_semantic_review(self):
        case = replace(self.cases[0], expected_status=Status.INSUFFICIENT)
        self.run_cases((case,))
        metrics = score_run(self.output, self.write_reviews(self.reviews((case,))))
        self.assertEqual(metrics["answer_acceptance_rate"], 0.0)

    def test_filter_compliance_checks_cited_prose_even_without_recommendations(self):
        long = load_catalog(ROOT / "catalog.json")[1]
        trace = Trace(self.cases[0].question,
                      QueryPlan("", intent=Intent.LOOKUP, filters=Filters(movie_ids=(long.movie_id,))),
                      (), (Evidence(long, ()),), None,
                      Answer(Status.ANSWERED, "96 минут", (long.source_id,)))
        self.assertFalse(strict_filters_compliant(self.cases[0], trace))
        self.traces["whale"] = trace
        self.run_cases(self.cases[:1])
        metrics = score_run(self.output, self.write_reviews(self.reviews(self.cases[:1])))
        self.assertEqual(metrics["strict_filter_compliance"], 0.0)
        self.assertEqual(metrics["answer_acceptance_rate"], 0.0)

    def test_quota_and_access_errors_stop_requests_without_dropping_cases(self):
        for status in (401, 403, 429):
            with self.subTest(status=status):
                service = Mock()
                service.prepare.side_effect = self.service.prepare.side_effect
                service.answer.side_effect = GenerationUnavailable("provider error", http_status=status)
                output = self.root / str(status)
                metrics = run_evaluation(self.cases, lambda: (service, {}), output, label="test")
                service.prepare.assert_called_once_with(self.cases[0].question)
                service.answer.assert_called_once()
                self.assertEqual(metrics["cases"], 2)
                self.assertEqual(metrics["technical_errors"], 1)
                self.assertEqual(metrics["not_executed"], 1)
                self.assertEqual(metrics["recall_at_k"], 1.0)
                self.assertEqual(metrics["prepared_cases"], 1)
                self.assertEqual(metrics["answer_acceptance_rate"], 0.0)
                self.assertTrue((output / "clarify/error.json").exists())

    def test_unexpected_error_keeps_denominator_and_redacts_arbitrary_details(self):
        self.service.prepare.side_effect = [TypeError("private-key"),
            PreparedQuery(self.traces["clarify"].question, self.traces["clarify"].plan, (), (), None)]
        self.run_cases()
        error = json.loads((self.output / "whale/error.json").read_text())
        self.assertEqual(error["type"], "TypeError")
        self.assertNotIn("private-key", error["message"])
        metrics = score_run(self.output, self.write_reviews(self.reviews(self.cases[1:])))
        self.assertEqual(metrics["answer_acceptance_rate"], 0.5)
        self.assertEqual(metrics["technical_errors"], 1)

    def test_existing_run_is_never_overwritten_and_unsafe_ids_are_rejected(self):
        self.run_cases()
        with self.assertRaises(FileExistsError):
            self.run_cases()
        self.assertEqual(self.factory.call_count, 1)
        with self.assertRaises(ValueError):
            run_evaluation((replace(self.cases[0], id="../unsafe"),), self.factory,
                           self.root / "other", label="test")
        self.assertFalse((self.root / "other").exists())

    def test_setup_failure_is_archived_for_every_case(self):
        self.factory.side_effect = ValueError("Invalid configuration")
        metrics = self.run_cases()
        self.assertEqual(metrics["not_executed"], 2)
        self.assertEqual(metrics["answer_acceptance_rate"], 0.0)
        self.assertEqual(json.loads((self.output / "manifest.json").read_text())["state"], "failed")

    def test_cli_score_is_offline(self):
        self.run_cases()
        reviews = self.write_reviews()
        out = io.StringIO()
        with patch("sys.argv", ["cartoon-rag", "evaluate", "score", "--run-dir", str(self.output),
                                 "--reviews", str(reviews)]), \
                patch.object(Settings, "load", side_effect=AssertionError("No API settings")), redirect_stdout(out):
            main()
        self.assertEqual(json.loads(out.getvalue())["answer_acceptance_rate"], 1.0)

    def test_cli_run_returns_nonzero_summary_on_quota_stop(self):
        questions = self.root / "questions.jsonl"
        questions.write_text("\n".join(json.dumps(asdict(case)) for case in self.cases) + "\n")
        self.service.answer.side_effect = GenerationUnavailable("Quota exceeded", http_status=429)
        out, err = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["cartoon-rag", "evaluate", "run", "--questions", str(questions),
                                 "--output", str(self.output), "--label", "test", "--delay-seconds", "0",
                                 "--prompt-version", "grounded-ru-v3"]), \
                patch.object(Settings, "load", return_value=Settings()), \
                patch("cartoon_rag.bootstrap.build_rag_runtime", return_value=(self.service, {})) as factory, \
                redirect_stdout(out), redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            main()
        self.assertEqual(caught.exception.code, 1)
        summary = json.loads(out.getvalue())
        self.assertEqual(summary["technical_errors"], 1)
        self.assertEqual(summary["not_executed"], 1)
        self.service.prepare.assert_called_once_with(self.cases[0].question)
        self.service.answer.assert_called_once()
        self.assertEqual(factory.call_args.kwargs["prompt_version"], "grounded-ru-v3")

    def test_generation_failure_keeps_evidence_written_before_model_call(self):
        for intent in (Intent.SEARCH, Intent.COUNT):
            with self.subTest(intent=intent):
                output = self.root / intent.value
                planner, retriever, generator = Mock(), Mock(), Mock()
                planner.plan.return_value = QueryPlan("кит", intent=intent, filters=self.cases[0].filters)
                retriever.search.return_value = self.traces["whale"].retrieved
                service = SearchService(planner, retriever, JsonExactCatalog((self.film,)), generator)

                def fail_after_checking_files(*args):
                    folder = output / "whale"
                    prepared = json.loads((folder / "prepared.json").read_text())
                    self.assertEqual(prepared["context"][0]["film"]["source_id"], self.film.source_id)
                    self.assertTrue((folder / "retrieved-ids.json").exists())
                    if intent is Intent.SEARCH:
                        self.assertEqual(prepared["retrieved"][0]["chunk"]["film"]["source_id"], self.film.source_id)
                    else:
                        self.assertEqual(json.loads((folder / "exact-result.json").read_text())["value"], 1)
                    raise GenerationUnavailable("HTTP 503", http_status=503)

                generator.generate.side_effect = fail_after_checking_files
                metrics = run_evaluation(self.cases[:1], lambda: (service, {}), output, label="failure")
                generator.generate.assert_called_once()
                planner.plan.assert_called_once_with(self.cases[0].question)
                self.assertEqual(retriever.search.call_count, int(intent is Intent.SEARCH))
                self.assertEqual(metrics["technical_errors"], 1)
                self.assertEqual(metrics["completed"], 0)
                self.assertEqual(metrics["prepared_cases"], 1)
                self.assertEqual(metrics["recall_at_1"], 1.0)
                self.assertEqual(metrics["strict_filter_compliance"], 1.0)
                self.assertEqual(metrics["answer_acceptance_rate"], 0.0)
                self.assertFalse((output / "whale/answer.json").exists())
                self.assertFalse((output / "whale/trace.json").exists())
                reviews = self.root / "empty-reviews.json"
                reviews.write_text(json.dumps({"reviewer": "test", "decisions": []}))
                self.assertEqual(score_run(output, reviews), metrics)
                judgment = json.loads((output / "whale/judgment.json").read_text())
                self.assertEqual(judgment["state"], "no_answer")
                self.assertFalse(judgment["accepted"])

    def test_retrieval_error_has_no_preparation_or_false_success(self):
        planner, retriever, generator = Mock(), Mock(), Mock()
        planner.plan.return_value = self.traces["whale"].plan
        retriever.search.side_effect = RetrievalUnavailable("timeout")
        service = SearchService(planner, retriever, JsonExactCatalog((self.film,)), generator)
        metrics = run_evaluation(self.cases[:1], lambda: (service, {}), self.output, label="failure")
        generator.generate.assert_not_called()
        self.assertEqual(metrics["prepared_cases"], 0)
        self.assertEqual(metrics["recall_at_k"], 0.0)
        self.assertEqual(metrics["strict_filter_compliance"], 0.0)
        self.assertFalse((self.output / "whale/prepared.json").exists())

    def test_persistence_failure_prevents_generation(self):
        write = experiments._write

        def disk_full(path, value):
            if path.name == "prepared.json":
                raise OSError("disk full")
            write(path, value)

        with patch("cartoon_rag.experiments._write", side_effect=disk_full):
            metrics = self.run_cases(self.cases[:1])
        self.service.answer.assert_not_called()
        self.assertEqual(metrics["prepared_cases"], 0)
        self.assertEqual(metrics["technical_errors"], 1)

    def test_corrupted_preparation_or_separate_evidence_cannot_be_scored(self):
        self.run_cases()
        reviews = self.write_reviews()
        for name in ("prepared", "retrieved", "context", "retrieved-ids", "plan", "exact-result"):
            with self.subTest(file=name):
                path = self.output / "whale" / (name + ".json")
                original = path.read_text()
                path.write_text('{}')
                with self.assertRaisesRegex(ValueError, "checksum|Separate prepared"):
                    score_run(self.output, reviews)
                path.write_text(original)

    def test_completed_trace_cannot_replace_prepared_evidence(self):
        original = self.service.answer.side_effect
        self.service.answer.side_effect = lambda prepared: replace(original(prepared), retrieved=())
        metrics = self.run_cases(self.cases[:1])
        self.assertEqual(metrics["completed"], 0)
        self.assertEqual(metrics["technical_errors"], 1)
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertFalse((self.output / "whale/trace.json").exists())

    def test_failed_generation_still_exposes_independent_filter_violation(self):
        long = load_catalog(ROOT / "catalog.json")[1]
        chunk = Chunk(long.source_id, long, long.synopsis)
        prepared = PreparedQuery(self.cases[0].question, QueryPlan("кит"),
                                 (Hit(chunk, 1.0),), (Evidence(long, (chunk,)),), None)
        self.service.prepare.return_value = prepared
        self.service.prepare.side_effect = None
        self.service.answer.side_effect = GenerationUnavailable("HTTP 503", http_status=503)
        metrics = self.run_cases(self.cases[:1])
        self.assertEqual(metrics["strict_filter_compliance"], 0.0)
        self.assertEqual(metrics["recall_at_k"], 0.0)
        self.assertEqual(metrics["answer_acceptance_rate"], 0.0)

    def test_scoring_checks_cross_phase_equality_even_with_valid_prepared_hash(self):
        self.run_cases(self.cases[:1])
        prepared = json.loads((self.output / "whale/prepared.json").read_text())
        prepared["retrieved"] = []
        experiments._save_prepared(self.output / "whale", prepared)
        records = json.loads((self.output / "results.json").read_text())
        records[0]["prepared_sha256"] = fingerprint(prepared)
        (self.output / "results.json").write_text(json.dumps(records))
        with self.assertRaisesRegex(ValueError, "Completed trace differs"):
            score_run(self.output, self.write_reviews(self.reviews(self.cases[:1])))

    def test_legacy_run_metrics_remain_identical_when_scored_in_copies(self):
        for name in ("ab-A-2026-10-25", "ab-B-2026-10-25", "expanded-run-v2-2026-10-26"):
            with self.subTest(run=name):
                source = ROOT / "artifacts" / name
                copy = self.root / name
                shutil.copytree(source, copy)
                expected = json.loads((source / "metrics.json").read_text())
                self.assertEqual(score_run(copy, copy / "reviews.json"), expected)


if __name__ == "__main__":
    unittest.main()
