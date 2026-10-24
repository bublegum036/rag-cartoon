"""Opt-in checks against a real Qdrant server; never use the application's alias."""

import os
import unittest
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from cartoon_rag.adapters.qdrant import QdrantHTTP, QdrantIndex
from cartoon_rag.domain import EmbeddingBatch, EmbeddingIdentity, Filters, QueryPlan, RetrievalUnavailable
from cartoon_rag.ingestion import SentenceChunker, WholeSynopsis, index_catalog, load_catalog


ROOT = Path(__file__).resolve().parents[1]


class FixedEmbedder:
    """Deliberate vectors isolate database filtering from semantic model quality."""

    identity = EmbeddingIdentity("test", "integration-test-only", "v1", 2)

    def embed(self, texts):
        # The long film ranks above the short film WITHOUT the database filter.
        return EmbeddingBatch(self.identity, tuple(
            (1.0, 0.0) if "Коралловая почта" in text else (0.8, 0.2) for text in texts
        ))

    def embed_query(self, text):
        return (1.0, 0.0)


@unittest.skipUnless(os.environ.get("QDRANT_INTEGRATION_URL"), "Set QDRANT_INTEGRATION_URL for real DB checks")
class QdrantLiveTest(unittest.TestCase):
    def setUp(self):
        self.transport = QdrantHTTP(
            os.environ["QDRANT_INTEGRATION_URL"],
            api_key=os.environ.get("QDRANT_API_KEY", ""), timeout=15,
        )
        self.alias = "test_" + uuid4().hex
        self.store = QdrantIndex(self.transport, self.alias, FixedEmbedder())
        self.films = load_catalog(ROOT / "catalog.json")

    def tearDown(self):
        # Delete only this test's random namespace, including failed staging loads.
        aliases = self.transport.request("GET", "/aliases")["aliases"]
        if any(item["alias_name"] == self.alias for item in aliases):
            self.transport.request("POST", "/collections/aliases", {
                "actions": [{"delete_alias": {"alias_name": self.alias}}],
            })
        collections = self.transport.request("GET", "/collections")["collections"]
        for item in collections:
            if item["name"].startswith(self.alias + "__"):
                self.transport.request("DELETE", "/collections/" + item["name"])

    def test_prefilter_before_top_k_repeat_and_version_update(self):
        short, long = self.films[:2]
        plan = QueryPlan("море", filters=Filters(duration_minutes_lt=90))
        index_catalog((short, long), WholeSynopsis(), FixedEmbedder(), self.store)
        self.assertEqual(self.store.search(QueryPlan("море"), limit=1)[0].chunk.film.movie_id, "film-02")
        self.assertEqual(self.store.search(plan, limit=1)[0].chunk.film.movie_id, "film-01")
        # Strict boundary and <= produce different results at exactly 78.
        self.assertEqual(self.store.search(QueryPlan("море", filters=Filters(duration_minutes_lt=78)), limit=5), ())
        self.assertEqual(len(self.store.search(QueryPlan("море", filters=Filters(duration_minutes_lte=78)), limit=5)), 1)
        publication = index_catalog((short, long), WholeSynopsis(), FixedEmbedder(), self.store)
        self.assertEqual(len(self.store.search(plan, limit=10)), 1)
        old_collection = publication.snapshot_id
        updated_publication = index_catalog(
            (short, replace(short, version=2, duration_minutes=92), long),
            WholeSynopsis(), FixedEmbedder(), self.store,
        )
        self.assertEqual(self.store.search(plan, limit=10), ())
        updated = self.store.search(QueryPlan("кит", filters=Filters(movie_ids=("film-01",))), limit=10)
        self.assertEqual([(hit.chunk.film.version, hit.chunk.film.duration_minutes) for hit in updated], [(2, 92)])
        self.assertNotEqual(updated_publication.snapshot_id, old_collection)

    def test_failed_upload_preserves_readable_snapshot_and_empty_snapshot_retires_it(self):
        film = self.films[0]
        index_catalog((film,), WholeSynopsis(), FixedEmbedder(), self.store)
        transport = self.transport

        class BrokenUpload:
            def request(self, method, path, body=None):
                if path.endswith("/points?wait=true"):
                    raise RetrievalUnavailable("simulated upload failure")
                return transport.request(method, path, body)

        broken = QdrantIndex(BrokenUpload(), self.alias, FixedEmbedder())
        with self.assertRaises(RetrievalUnavailable):
            index_catalog((replace(film, version=2, duration_minutes=92),),
                          WholeSynopsis(), FixedEmbedder(), broken)
        self.assertEqual(self.store.search(QueryPlan("кит"), limit=1)[0].chunk.film.version, 1)
        self.store.replace_snapshot([], FixedEmbedder().embed([]))
        self.assertEqual(self.store.search(QueryPlan("кит"), limit=5), ())

    @unittest.skipUnless(os.environ.get("RUN_SEMANTIC_SMOKE") == "1", "Set RUN_SEMANTIC_SMOKE=1 to download E5 and check semantics")
    def test_real_embeddings_find_whale_and_robot(self):
        from cartoon_rag.adapters.fastembed import E5Embedder
        from cartoon_rag.settings import Settings

        settings = Settings.load()
        embedder = E5Embedder(revision=settings.model_revision, cache=settings.model_cache,
                              threads=settings.embedding_threads)
        store = QdrantIndex(self.transport, self.alias, embedder)
        index_catalog(self.films, SentenceChunker(settings.chunk_chars), embedder, store)
        for question, expected in (("Девочка и кит чинят маяк", "film-01"),
                                   ("Робот выращивает сад вместе с детьми", "film-05")):
            with self.subTest(question=question):
                hits = store.search(QueryPlan(question), limit=3)
                self.assertIn(expected, {hit.chunk.film.movie_id for hit in hits})
        hits = store.search(QueryPlan("море и дружба", filters=Filters(duration_minutes_lt=90)), limit=5)
        self.assertIn("film-01", {hit.chunk.film.movie_id for hit in hits})
        self.assertTrue(all(hit.chunk.film.duration_minutes < 90 for hit in hits))


if __name__ == "__main__":
    unittest.main()
