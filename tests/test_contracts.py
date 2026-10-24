import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from cartoon_rag.application import RetrievalService, SearchService
from cartoon_rag.domain import (
    Answer, Chunk, EmbeddingBatch, EmbeddingIdentity, Filters, GroundingError, Hit,
    PublicationResult, QueryPlan, RetrievalUnavailable, Status,
)
from cartoon_rag.evaluation import accepted, load_cases, recall_at_k, save_trace
from cartoon_rag.ingestion import WholeSynopsis, active_snapshot, index_catalog, load_catalog


ROOT = Path(__file__).resolve().parents[1]


class Planner:
    def __init__(self, plan):
        self.result = plan

    def plan(self, question):
        return self.result


class Retrieval:
    def __init__(self, hits=(), error=None):
        self.hits = hits
        self.error = error

    def search(self, plan, *, limit):
        if self.error:
            raise self.error
        return self.hits[:limit]


class Catalog:
    def execute(self, plan):
        raise AssertionError("Semantic search must not call exact catalog")


class Generator:
    def __init__(self, source_ids=("film-01#plot",)):
        self.source_ids = source_ids

    def generate(self, question, plan, context, exact_result):
        return Answer(Status.ANSWERED, "Маяк для кита", self.source_ids)


class ContractsTest(unittest.TestCase):
    def setUp(self):
        self.films = load_catalog(ROOT / "catalog.json")
        self.film = self.films[0]
        self.chunk = Chunk(self.film.source_id, self.film, self.film.synopsis)

    def service(self, retrieval, *, plan=None, generator=None):
        return SearchService(
            Planner(plan or QueryPlan("кит")), retrieval, Catalog(), generator or Generator(),
        )

    def test_supplied_dataset(self):
        self.assertEqual(len(active_snapshot(self.films)), 12)
        cases = load_cases(ROOT / "questions.jsonl")
        self.assertEqual(len(cases), 10)
        known = {film.source_id for film in self.films}
        self.assertTrue(all(set(case.relevant_source_ids) <= known for case in cases))

    def test_strict_duration_boundary(self):
        self.assertFalse(Filters(duration_minutes_lt=78).matches(self.film))
        self.assertTrue(Filters(duration_minutes_lte=78).matches(self.film))

    def test_catalog_rejects_non_boolean_active(self):
        row = json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            for value in ("false", "true", "", 0, 1, None, [], {}):
                with self.subTest(active=value):
                    path.write_text(json.dumps([{**row, "active": value}]), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "active.*boolean"):
                        load_catalog(path)

    def test_catalog_accepts_boolean_active(self):
        row = json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            for value in (True, False):
                with self.subTest(active=value):
                    path.write_text(json.dumps([{**row, "active": value}]), encoding="utf-8")
                    films = load_catalog(path)
                    self.assertIs(films[0].active, value)
                    self.assertEqual(len(active_snapshot(films)), int(value))

    def test_evaluation_checks_all_retrieved_hits_against_reference_filters(self):
        case = load_cases(ROOT / "questions.jsonl")[0]

        class Judge:
            def accepts(self, case, trace):
                return True

        # The planner lost the filter; a correct final answer alone must not
        # hide a violation in a hit excluded from the selected context.
        good = self.service(Retrieval((Hit(self.chunk, 1.0),))).ask(case.question)
        self.assertTrue(accepted(case, good, Judge()))
        for duration in (90, 96):
            with self.subTest(duration=duration):
                film = replace(self.films[1], duration_minutes=duration)
                bad_hit = Hit(Chunk(film.source_id, film, film.synopsis), 0.5)
                trace = replace(good, retrieved=good.retrieved + (bad_hit,))
                self.assertFalse(accepted(case, trace, Judge()))

    def test_new_revision_and_deactivation(self):
        updated = replace(self.film, version=2, duration_minutes=92)
        snapshot = active_snapshot((self.film, updated))
        self.assertEqual(snapshot, (updated,))
        self.assertFalse(Filters(duration_minutes_lt=90).matches(snapshot[0]))
        self.assertEqual(active_snapshot((self.film, replace(updated, active=False))), ())

    def test_index_embeds_title_and_publishes_only_latest_revision(self):
        captured = {}
        identity = EmbeddingIdentity("test", "test-model", "v1", 1)
        publication = PublicationResult("fake", "snapshot", identity, "catalog", "parts", 1, 1)

        class Embed:
            def embed(self, texts):
                captured["texts"] = texts
                return EmbeddingBatch(identity, tuple((1.0,) for _ in texts))

        class Writer:
            def replace_snapshot(self, chunks, batch):
                captured["chunks"] = chunks
                captured["batch"] = batch
                return publication

        updated = replace(self.film, version=2, duration_minutes=92)
        self.assertIs(index_catalog((self.film, updated), WholeSynopsis(), Embed(), Writer()), publication)
        self.assertEqual(captured["batch"].identity, identity)
        self.assertIn(self.film.title, captured["texts"][0])
        self.assertIn(self.film.synopsis, captured["texts"][0])
        self.assertEqual(captured["chunks"][0].film.version, 2)

    def test_grouping_and_chunk_parent_recall(self):
        parts = (replace(self.chunk, chunk_id="film-01#plot:part-1"),
                 replace(self.chunk, chunk_id="film-01#plot:part-2"))
        trace = self.service(Retrieval(tuple(Hit(part, 1.0) for part in parts))).ask("кит")
        self.assertEqual(len(trace.context), 1)
        self.assertEqual(len(trace.context[0].chunks), 2)
        case = load_cases(ROOT / "questions.jsonl")[0]
        self.assertEqual(recall_at_k(case, trace, 1), 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            save_trace(path, trace)
            self.assertIn('"retrieved"', path.read_text())

    def test_invalid_retrieval_filter_is_not_silently_hidden(self):
        service = self.service(
            Retrieval((Hit(self.chunk, 1.0),)),
            plan=QueryPlan("кит", filters=Filters(duration_minutes_lt=40)),
        )
        with self.assertRaises(GroundingError):
            service.ask("короче 40")

    def test_shared_retrieval_deduplicates_chunks_and_matches_rag_context(self):
        first = replace(self.chunk, chunk_id="film-01#plot:part-1")
        second = replace(self.chunk, chunk_id="film-01#plot:part-2")
        other = self.films[1]
        retrieval = Retrieval((Hit(first, 0.9), Hit(second, 0.8), Hit(second, 0.8),
                               Hit(Chunk(other.source_id, other, other.synopsis), 0.7)))
        result = RetrievalService(retrieval).search(QueryPlan("кит"), limit=10)
        self.assertEqual(len(result.retrieved), 4)
        self.assertEqual([movie.film.movie_id for movie in result.movies], ["film-01", "film-02"])
        self.assertEqual(result.movies[0].chunks, (first, second))
        self.assertEqual(result.movies[0].score, 0.9)
        self.assertEqual(self.service(retrieval).ask("кит").context, result.context)

    def test_shared_retrieval_rejects_mixed_revisions(self):
        newer = replace(self.chunk, film=replace(self.film, version=2, duration_minutes=92))
        retrieval = Retrieval((Hit(self.chunk, 0.9), Hit(newer, 0.8)))
        with self.assertRaisesRegex(GroundingError, "Mixed movie revisions"):
            RetrievalService(retrieval).search(QueryPlan("кит"), limit=10)

    def test_timeout_is_not_insufficient(self):
        with self.assertRaises(RetrievalUnavailable):
            self.service(Retrieval(error=RetrievalUnavailable("timeout"))).ask("кит")

    def test_missing_results_are_insufficient(self):
        self.assertIs(self.service(Retrieval()).ask("кит").answer.status, Status.INSUFFICIENT)

    def test_unknown_generated_sources_are_rejected(self):
        with self.assertRaises(GroundingError):
            self.service(Retrieval((Hit(self.chunk, 1.0),)),
                         generator=Generator(("invented",))).ask("кит")

    def test_clarification_does_not_query_database(self):
        trace = self.service(
            Retrieval(error=AssertionError("Must not retrieve")),
            plan=QueryPlan("", early_status=Status.CLARIFY, early_text="Какой фильм?"),
        ).ask("Сколько он длится?")
        self.assertIs(trace.answer.status, Status.CLARIFY)


if __name__ == "__main__":
    unittest.main()
