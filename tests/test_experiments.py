import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

from cartoon_rag.cli import main
from cartoon_rag.domain import (
    Answer, Chunk, Evidence, Filters, GenerationUnavailable, Hit, Intent, QueryPlan, Status, Trace,
)
from cartoon_rag.evaluation import (
    Case, ReviewedSemanticJudge, strict_filters_compliant, trace_from_dict,
)
from cartoon_rag.experiments import run_evaluation, score_run
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
        self.service.ask.side_effect = lambda question: next(
            self.traces[case.id] for case in self.cases if case.question == question)
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
        self.assertEqual([call.args for call in self.service.ask.call_args_list],
                         [(case.question,) for case in self.cases])
        for case in self.cases:
            for name in ("trace", "retrieved", "retrieved-ids", "context", "answer", "case"):
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
                service.ask.side_effect = GenerationUnavailable("provider error", http_status=status)
                output = self.root / str(status)
                metrics = run_evaluation(self.cases, lambda: (service, {}), output, label="test")
                service.ask.assert_called_once()
                self.assertEqual(metrics["cases"], 2)
                self.assertEqual(metrics["technical_errors"], 1)
                self.assertEqual(metrics["not_executed"], 1)
                self.assertEqual(metrics["recall_at_k"], 0.0)
                self.assertEqual(metrics["answer_acceptance_rate"], 0.0)
                self.assertTrue((output / "clarify/error.json").exists())

    def test_unexpected_error_keeps_denominator_and_redacts_arbitrary_details(self):
        self.service.ask.side_effect = [TypeError("private-key"), self.traces["clarify"]]
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
        self.service.ask.side_effect = GenerationUnavailable("Quota exceeded", http_status=429)
        out, err = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["cartoon-rag", "evaluate", "run", "--questions", str(questions),
                                 "--output", str(self.output), "--label", "test", "--delay-seconds", "0",
                                 "--prompt-version", "grounded-ru-v1"]), \
                patch.object(Settings, "load", return_value=Settings()), \
                patch("cartoon_rag.bootstrap.build_rag_runtime", return_value=(self.service, {})) as factory, \
                redirect_stdout(out), redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            main()
        self.assertEqual(caught.exception.code, 1)
        summary = json.loads(out.getvalue())
        self.assertEqual(summary["technical_errors"], 1)
        self.assertEqual(summary["not_executed"], 1)
        self.service.ask.assert_called_once_with(self.cases[0].question)
        self.assertEqual(factory.call_args.kwargs["prompt_version"], "grounded-ru-v1")


if __name__ == "__main__":
    unittest.main()
