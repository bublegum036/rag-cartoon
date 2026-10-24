import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

from cartoon_rag.adapters.qdrant import QdrantIndex
from cartoon_rag.bootstrap import build_rag
from cartoon_rag.cli import main
from cartoon_rag.domain import (
    Answer, EmbeddingIdentity, GenerationUnavailable, QueryPlan, SnapshotMismatch, Status,
)
from cartoon_rag.ingestion import load_catalog
from cartoon_rag.settings import Settings
from cartoon_rag.snapshots import catalog_fingerprint


ROOT = Path(__file__).resolve().parents[1]


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        self.films = load_catalog(ROOT / "catalog.json")
        self.digest = catalog_fingerprint(self.films)
        self.embedder = Mock()
        self.embedder.identity = EmbeddingIdentity("test", "test", "v1", 2)
        self.embedder.embed_query.return_value = (1.0, 0.0)
        self.transport = Mock()
        self.metadata = {"embedding": asdict(self.embedder.identity), "catalog_sha256": self.digest}
        self.alias_target = "old-snapshot"

        def request(method, path, body=None):
            if path == "/aliases":
                return {"aliases": [{"alias_name": "test", "collection_name": self.alias_target}]}
            if path.endswith("/points/query"):
                return {"points": []}
            return {"config": {"metadata": self.metadata}}

        self.transport.request.side_effect = request
        self.index = QdrantIndex(self.transport, "test", self.embedder)

    def test_catalog_hash_matches_existing_published_format_and_ignores_record_order(self):
        published = json.loads((ROOT / "artifacts/verification-2026-10-25/index.json").read_text())
        self.assertEqual(self.digest, published["catalog_sha256"])
        self.assertEqual(catalog_fingerprint(tuple(reversed(self.films))), self.digest)
        updated = replace(self.films[0], version=2, duration_minutes=92)
        self.assertNotEqual(catalog_fingerprint((*self.films, updated)), self.digest)
        self.assertEqual(catalog_fingerprint((*self.films, updated)),
                         catalog_fingerprint((updated, *self.films[1:])))

    def test_wrong_or_missing_hash_fails_before_embedding_and_query(self):
        for metadata in ({"embedding": asdict(self.embedder.identity)},
                         {**self.metadata, "catalog_sha256": "0" * 64}):
            self.metadata = metadata
            with self.assertRaises(SnapshotMismatch):
                self.index.bind_snapshot(self.digest)
        self.embedder.embed_query.assert_not_called()
        self.assertFalse(any(call.args[1].endswith("/points/query")
                             for call in self.transport.request.call_args_list))

    def test_alias_switch_cannot_change_collection_after_binding(self):
        pinned = self.index.bind_snapshot(self.digest)
        self.alias_target = "new-snapshot"
        self.transport.reset_mock()
        pinned.search(QueryPlan("море"), limit=3)
        paths = [call.args[1] for call in self.transport.request.call_args_list]
        self.assertNotIn("/aliases", paths)
        self.assertEqual(paths, ["/collections/old-snapshot", "/collections/old-snapshot/points/query"])

    def test_bound_metadata_is_rechecked_before_model_inference(self):
        pinned = self.index.bind_snapshot(self.digest)
        self.metadata["catalog_sha256"] = "0" * 64
        with self.assertRaises(SnapshotMismatch):
            pinned.search(QueryPlan("море"), limit=3)
        self.embedder.embed_query.assert_not_called()

    def test_composition_checks_catalog_before_exact_generation_too(self):
        generator = Mock()
        generator.generate.return_value = Answer(Status.ANSWERED, "12 фильмов.",
                                                tuple(f.source_id for f in self.films))
        with patch("cartoon_rag.bootstrap.build_search", return_value=(self.embedder, self.index)):
            service = build_rag(self.films, Settings(), generator=generator)
            trace = service.ask("Сколько фильмов в каталоге?")
            self.assertEqual(trace.exact_result.value, 12)
            self.metadata["catalog_sha256"] = "0" * 64
            with self.assertRaises(SnapshotMismatch):
                build_rag(self.films, Settings(), generator=generator)
        self.assertEqual(generator.generate.call_count, 1)


class AskCLITest(unittest.TestCase):
    def test_early_answers_work_without_env_or_providers_and_save_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "bad.env").write_text("INVALID")
            for question, status in (("Сколько он длится?", "clarify"),
                                     ("Кто озвучивает девочку?", "insufficient")):
                out = io.StringIO()
                with patch("sys.argv", ["cartoon-rag", "--env-file", str(path / "bad.env"),
                                         "ask", question, "--trace", str(path / "trace.json")]), \
                        patch("cartoon_rag.bootstrap.build_rag", side_effect=AssertionError("No model")), \
                        patch.object(Settings, "load", side_effect=AssertionError("No env")), \
                        redirect_stdout(out):
                    main()
                self.assertEqual(json.loads(out.getvalue())["status"], status)
                trace = json.loads((path / "trace.json").read_text())
                self.assertEqual(trace["answer"]["status"], status)
                self.assertEqual(trace["retrieved"], [])

    def test_model_error_has_no_answer_status_or_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.json"
            out, err = io.StringIO(), io.StringIO()
            service = Mock()
            service.execute_plan.side_effect = GenerationUnavailable("HTTP 429")
            with patch("sys.argv", ["cartoon-rag", "ask", "про море", "--trace", str(trace_path)]), \
                    patch.object(Settings, "load", return_value=Settings()), \
                    patch("cartoon_rag.bootstrap.build_rag", return_value=service), \
                    redirect_stdout(out), redirect_stderr(err), self.assertRaises(SystemExit) as caught:
                main()
            self.assertEqual(caught.exception.code, 1)
            self.assertEqual(out.getvalue(), "")
            self.assertEqual(json.loads(err.getvalue())["error"]["type"], "GenerationUnavailable")
            self.assertFalse(trace_path.exists())


if __name__ == "__main__":
    unittest.main()
