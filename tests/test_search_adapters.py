import io
import json
import os
import tempfile
import unittest
from dataclasses import asdict, replace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from cartoon_rag.adapters.fastembed import E5Embedder
from cartoon_rag.adapters.qdrant import QdrantHTTP, QdrantIndex, qdrant_filter
from cartoon_rag.domain import (
    Chunk, EmbeddingBatch, EmbeddingIdentity, Filters, GroundingError, Hit,
    PublicationResult, QueryPlan, RetrievalUnavailable,
)
from cartoon_rag.cli import main
from cartoon_rag.ingestion import SentenceChunker, index_catalog, load_catalog
from cartoon_rag.settings import MODEL_REVISION, Settings, read_env


ROOT = Path(__file__).resolve().parents[1]


class StubEmbedder:
    identity = EmbeddingIdentity("test", "test-only", "v1", 2)

    def embed(self, texts):
        return EmbeddingBatch(self.identity, tuple((1.0, 0.0) for _ in texts))

    def embed_query(self, text):
        return (1.0, 0.0)


class RecordingTransport:
    """Observe requests without opening sockets or implementing a search engine."""

    def __init__(self, *, existing=True, fail_at=None):
        self.calls = []
        self.existing = existing
        self.fail_at = fail_at
        self.points = []
        self.query_response = {"points": []}
        self.metadata = {"embedding": asdict(StubEmbedder.identity)}

    def request(self, method, path, body=None):
        self.calls.append((method, path, body))
        if self.fail_at and self.fail_at in path:
            raise RetrievalUnavailable("injected failure")
        if path == "/aliases":
            return {"aliases": ([{"alias_name": "test", "collection_name": "previous"}]
                                if self.existing else [])}
        if path.endswith("/points/count"):
            return {"count": len(self.points)}
        if path.endswith("/points?wait=true"):
            self.points.extend(body["points"])
        if path.endswith("/points/query"):
            return self.query_response
        if method == "GET" and path.startswith("/collections/"):
            return {"config": {"metadata": self.metadata}}
        return True


class SearchAdapterTest(unittest.TestCase):
    def setUp(self):
        self.film = load_catalog(ROOT / "catalog.json")[0]
        self.chunk = Chunk(self.film.source_id, self.film, self.film.synopsis)
        self.transport = RecordingTransport()
        self.store = QdrantIndex(self.transport, "test", StubEmbedder())

    def test_prefilter_sent_in_same_request_as_limit_and_vector(self):
        filters = Filters(duration_minutes_lt=90, duration_minutes_lte=85, movie_ids=("film-01",))
        self.assertEqual(self.store.search(QueryPlan("кит", filters=filters), limit=1), ())
        method, path, body = self.transport.calls[-1]
        self.assertEqual((method, path), ("POST", "/collections/previous/points/query"))
        self.assertEqual(body["limit"], 1)
        self.assertEqual(body["query"], [1.0, 0.0])
        self.assertEqual(body["filter"], {"must": [
            {"key": "active", "match": {"value": True}},
            {"key": "duration_minutes", "range": {"lt": 90, "lte": 85}},
            {"key": "movie_id", "match": {"any": ["film-01"]}},
        ]})

    def test_active_filter_always_present(self):
        self.assertEqual(qdrant_filter(Filters()), {"must": [
            {"key": "active", "match": {"value": True}},
        ]})

    def test_payload_roundtrip(self):
        self.transport.query_response = {"points": [{
            "score": 0.9, "payload": {**asdict(self.film),
                                     "chunk_id": self.chunk.chunk_id, "text": self.chunk.text},
        }]}
        hits = self.store.search(QueryPlan("кит"), limit=1)
        self.assertEqual(hits, (Hit(self.chunk, 0.9),))

    def test_query_model_mismatch_fails_before_vector_query(self):
        self.transport.metadata = {"embedding": {"model": "different"}}
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.store.search(QueryPlan("кит"), limit=1)
        self.assertFalse(any(path.endswith("/points/query") for _, path, _ in self.transport.calls))

    def test_missing_index_is_an_error(self):
        self.transport.existing = False
        with self.assertRaisesRegex(RetrievalUnavailable, "run index first"):
            self.store.search(QueryPlan("кит"), limit=1)

    def test_filter_violation_in_server_response_is_rejected(self):
        self.transport.query_response = {"points": [{
            "score": 0.9, "payload": {**asdict(self.film),
                                     "chunk_id": self.chunk.chunk_id, "text": self.chunk.text},
        }]}
        with self.assertRaises(GroundingError):
            self.store.search(QueryPlan("кит", filters=Filters(duration_minutes_lt=40)), limit=1)

    def test_publication_follows_completed_upload_and_count(self):
        publication = self.store.replace_snapshot([self.chunk], EmbeddingBatch(StubEmbedder.identity, ((1.0, 0.0),)))
        method, path, body = self.transport.calls[-1]
        self.assertEqual((method, path), ("POST", "/collections/aliases"))
        self.assertEqual(body["actions"][0], {"delete_alias": {"alias_name": "test"}})
        target = body["actions"][1]["create_alias"]["collection_name"]
        self.assertNotEqual(target, "previous")
        create = self.transport.calls[0]
        self.assertEqual(create[1], f"/collections/{target}")
        self.assertEqual(create[2]["metadata"]["embedding"], asdict(StubEmbedder.identity))
        self.assertEqual(publication.movies, 1)
        self.assertEqual(publication.embedding, StubEmbedder.identity)
        self.assertEqual(publication.snapshot_id, target)
        self.assertTrue(any(path.endswith("/points/count") for _, path, _ in self.transport.calls))

    def test_indexing_rejects_foreign_model_identity_before_any_db_request(self):
        original = StubEmbedder.identity
        for foreign in (replace(original, model="another-model"),
                        replace(original, revision="v2"),
                        replace(original, encoding="another-encoding")):
            for films in ((self.film,), ()):
                with self.subTest(identity=foreign, empty=not films):
                    producer = StubEmbedder()
                    producer.identity = foreign  # Same dimensions, different vector space.
                    with self.assertRaisesRegex(ValueError, "Batch embedding identity differs"):
                        index_catalog(films, SentenceChunker(), producer, self.store)
                    self.assertEqual(self.transport.calls, [])

    def test_failed_upload_does_not_publish_or_delete_previous_snapshot(self):
        self.transport.fail_at = "/points?wait=true"
        with self.assertRaises(RetrievalUnavailable):
            self.store.replace_snapshot([self.chunk], EmbeddingBatch(StubEmbedder.identity, ((1.0, 0.0),)))
        self.assertFalse(any(path == "/collections/aliases" for _, path, _ in self.transport.calls))
        self.assertFalse(any(method == "DELETE" for method, _, _ in self.transport.calls))

    def test_ambiguous_alias_timeout_does_not_delete_potentially_active_snapshot(self):
        self.transport.fail_at = "/collections/aliases"
        with self.assertRaises(RetrievalUnavailable):
            self.store.replace_snapshot([self.chunk], EmbeddingBatch(StubEmbedder.identity, ((1.0, 0.0),)))
        self.assertFalse(any(method == "DELETE" for method, _, _ in self.transport.calls))

    def test_invalid_vectors_fail_before_mutating_storage(self):
        for vector in ([1.0], [float("nan"), 0.0], [float("inf"), 0.0], [0.0, 0.0]):
            with self.subTest(vector=vector):
                with self.assertRaises(ValueError):
                    self.store.replace_snapshot([self.chunk], EmbeddingBatch(StubEmbedder.identity, (tuple(vector),)))
                self.assertEqual(self.transport.calls, [])

    def test_empty_snapshot_can_replace_existing_catalog(self):
        publication = self.store.replace_snapshot([], StubEmbedder().embed([]))
        self.assertEqual(publication.chunks, 0)
        self.assertEqual(self.transport.calls[-1][1], "/collections/aliases")

    def test_http_failure_keeps_status_and_hides_key(self):
        client = QdrantHTTP("http://localhost:6333", api_key="secret")
        error = HTTPError("http://localhost", 503, "unavailable", {}, None)
        with patch("cartoon_rag.adapters.qdrant.urlopen", side_effect=error):
            with self.assertRaisesRegex(RetrievalUnavailable, "HTTP 503") as caught:
                client.request("GET", "/aliases")
        self.assertNotIn("secret", str(caught.exception))

    def test_http_serializes_json_and_extracts_result(self):
        client = QdrantHTTP("http://localhost:6333", api_key="key", timeout=3)
        with patch("cartoon_rag.adapters.qdrant.urlopen") as mocked:
            mocked.return_value.__enter__.return_value = io.BytesIO(b'{"result": {"count": 2}}')
            self.assertEqual(client.request("POST", "/count", {"exact": True}), {"count": 2})
        request = mocked.call_args.args[0]
        self.assertEqual(json.loads(request.data), {"exact": True})
        self.assertEqual(request.get_header("Api-key"), "key")
        self.assertEqual(mocked.call_args.kwargs["timeout"], 3)


class ChunkAndSettingsTest(unittest.TestCase):
    def test_short_catalog_cards_remain_whole(self):
        for film in load_catalog(ROOT / "catalog.json"):
            self.assertEqual(SentenceChunker().split(film.synopsis), (film.synopsis,))

    def test_sentence_split_retains_parent_and_embedding_title(self):
        film = replace(load_catalog(ROOT / "catalog.json")[0], synopsis="Один. Два. Три.")

        class Writer:
            def replace_snapshot(self, chunks, batch):
                self.chunks = chunks
                return PublicationResult("fake", "snapshot", batch.identity, "catalog", "parts", 1, len(chunks))

        writer = Writer()
        index_catalog((film,), SentenceChunker(6), StubEmbedder(), writer)
        self.assertEqual([chunk.text for chunk in writer.chunks], ["Один.", "Два.", "Три."])
        self.assertTrue(all(chunk.film.source_id == film.source_id for chunk in writer.chunks))
        self.assertEqual(writer.chunks[0].chunk_id, film.source_id + ":part-1")
        self.assertIn(film.title, writer.chunks[0].embedding_text)

    def test_environment_overrides_file_without_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text('QDRANT_ALIAS="file"\nMODEL_CACHE=$(literal)\n', encoding="utf-8")
            with patch.dict(os.environ, {"QDRANT_ALIAS": "process"}, clear=True):
                settings = Settings.load(path)
            self.assertEqual(settings.qdrant_alias, "process")
            self.assertEqual(settings.model_cache, Path("$(literal)"))

    def test_invalid_settings_rejected(self):
        for kwargs in ({"qdrant_url": "file:///tmp/qdrant"}, {"qdrant_alias": "a/b"},
                       {"timeout_seconds": float("nan")}, {"timeout_seconds": 0},
                       {"model_revision": "main"}, {"embedding_threads": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Settings(**kwargs)
        self.assertEqual(Settings().model_revision, MODEL_REVISION)

    def test_invalid_env_line_has_line_number_without_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("QDRANT_API_KEY='secret\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1") as caught:
                read_env(path)
            self.assertNotIn("secret", str(caught.exception))

    def test_e5_prefixes_and_token_overflow(self):
        # Exercise preprocessing without downloading weights or substituting a
        # fake semantic model for a real search acceptance test.
        embedder = object.__new__(E5Embedder)
        seen = []

        class Model:
            def embed(self, inputs, **kwargs):
                seen.extend(inputs)
                return [(1.0, 0.0)] * len(inputs)

        class Tokenizer:
            def encode(self, text):
                class Encoding:
                    ids = list(range(len(text)))
                return Encoding()

        embedder.identity = StubEmbedder.identity
        embedder.model = Model()
        embedder.tokenizer = Tokenizer()
        embedder.embed(["Название\nОписание"])
        embedder.embed_query("кит")
        self.assertEqual(seen, ["passage: Название\nОписание", "query: кит"])
        with self.assertRaisesRegex(ValueError, "512 tokens"):
            embedder.embed(["x" * 513])


class CLITest(unittest.TestCase):
    def test_index_serializes_result_from_provider_independent_writer(self):
        publication = PublicationResult("fake-index", "fake-snapshot", StubEmbedder.identity,
                                        "catalog-hash", "parts-hash", 12, 12)

        class Writer:
            def replace_snapshot(self, chunks, batch):
                return publication

        output = io.StringIO()
        with patch("cartoon_rag.bootstrap.build_search", return_value=(StubEmbedder(), Writer())), \
                patch.object(Settings, "load", return_value=Settings()), \
                patch("sys.argv", ["cartoon-rag", "index", "--catalog", str(ROOT / "catalog.json")]), \
                redirect_stdout(output):
            main()
        self.assertEqual(json.loads(output.getvalue()), asdict(publication))

    def test_search_outputs_chunks_and_unique_movies_with_explicit_filters(self):
        film = load_catalog(ROOT / "catalog.json")[0]
        hits = (Hit(Chunk(film.source_id + ":part-1", film, "Первая часть."), 0.9),
                Hit(Chunk(film.source_id + ":part-2", film, "Вторая часть."), 0.8))

        class Store:
            def search(self, plan, *, limit):
                self.plan = plan
                return hits[:limit]

        store = Store()
        output = io.StringIO()
        with patch("cartoon_rag.bootstrap.build_search", return_value=(None, store)), \
                patch.object(Settings, "load", return_value=Settings()), \
                patch("sys.argv", ["cartoon-rag", "search", "кит", "--duration-lt", "90"]), \
                redirect_stdout(output):
            main()
        result = json.loads(output.getvalue())
        self.assertEqual(len(result["retrieved"]), 2)
        self.assertEqual(len(result["movies"]), 1)
        self.assertEqual(len(result["movies"][0]["chunks"]), 2)
        self.assertEqual(store.plan.filters.duration_minutes_lt, 90)
        self.assertNotIn("answer", result)

    def test_database_failure_exits_nonzero_without_insufficient_result(self):
        errors = io.StringIO()
        output = io.StringIO()
        with patch("cartoon_rag.bootstrap.build_search", side_effect=RetrievalUnavailable("timeout")), \
                patch.object(Settings, "load", return_value=Settings()), \
                patch("sys.argv", ["cartoon-rag", "search", "кит"]), \
                redirect_stderr(errors), redirect_stdout(output):
            with self.assertRaises(SystemExit) as caught:
                main()
        self.assertEqual(caught.exception.code, 1)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(json.loads(errors.getvalue())["error"]["type"], "RetrievalUnavailable")


if __name__ == "__main__":
    unittest.main()
