import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

from cartoon_rag.adapters.gemini_extractive import (
    EXTRACTIVE_INSTRUCTION, SELECTION_SCHEMA, GeminiExtractiveGenerator,
)
from cartoon_rag.application import SearchService
from cartoon_rag.bootstrap import build_exact_catalog, build_planner
from cartoon_rag.domain import (
    Chunk, Evidence, ExactResult, Filters, GenerationUnavailable, GroundingError, Intent, QueryPlan, Status,
)
from cartoon_rag.extractive import render_selection, source_fragments
from cartoon_rag.ingestion import load_catalog


ROOT = Path(__file__).resolve().parents[1]


def envelope(value, finish="STOP"):
    return {"candidates": [{"finishReason": finish, "content": {
        "parts": [{"text": json.dumps(value, ensure_ascii=False)}],
    }}]}


class ExtractiveTest(unittest.TestCase):
    def setUp(self):
        self.films = load_catalog(ROOT / "data/catalog-expanded.json")
        self.film = self.films[0]
        self.chunk = Chunk(self.film.source_id, self.film, self.film.synopsis)
        self.context = (Evidence(self.film, (self.chunk,)),)
        self.plan = QueryPlan("море и дружба", filters=Filters(duration_minutes_lt=90))
        self.selection = {"status": "answered", "selections": [
            {"movie_id": self.film.movie_id, "fragment_ids": [self.chunk.chunk_id]},
        ]}
        self.transport = Mock()
        self.transport.generate.return_value = envelope(self.selection)
        self.generator = GeminiExtractiveGenerator(self.transport)

    def test_output_uses_exact_context_quote_and_numeric_card_facts(self):
        answer = self.generator.generate("море короче 90 минут", self.plan, self.context, None)
        self.assertIs(answer.status, Status.ANSWERED)
        self.assertIn("78 минут", answer.text)
        self.assertIn("2021", answer.text)
        self.assertEqual(answer.recommendations[0].reason, f"«{self.film.synopsis}»")
        self.assertNotIn("В благодарность", answer.text)
        self.assertEqual(answer.source_ids, (self.film.source_id,))
        body = self.transport.generate.call_args.args[0]
        self.assertEqual(body["systemInstruction"]["parts"][0]["text"], EXTRACTIVE_INSTRUCTION)
        self.assertEqual(body["generationConfig"]["responseFormat"]["text"]["schema"], SELECTION_SCHEMA)
        payload = json.loads(body["contents"][0]["parts"][0]["text"])
        self.assertEqual(payload["candidates"][0]["fragments"][0]["text"], self.film.synopsis)
        self.transport.generate.assert_called_once()

    def test_complete_mechanical_whale_fragment_preserves_final_use_of_measurements(self):
        film = next(f for f in self.films if f.movie_id == "film-37")
        selection = {"status": "answered", "selections": [
            {"movie_id": film.movie_id, "fragment_ids": [film.source_id]},
        ]}
        answer = render_selection(selection, QueryPlan("механический кит"), (Evidence(film, ()),))
        self.assertIn("рычага и лебёдки", answer.text)
        self.assertIn("безопасный проход для лодок", answer.text)
        self.assertEqual(answer.recommendations[0].reason, f"«{film.synopsis}»")

    def test_fragment_ownership_and_unknown_duplicate_or_forged_selections(self):
        other = self.films[1]
        context = (*self.context, Evidence(other, ()))
        variants = [
            {**self.selection, "text": "В благодарность кит помогает Нике"},
            {"status": "answered", "selections": [{"movie_id": self.film.movie_id,
                "fragment_ids": [self.chunk.chunk_id], "quote": "В благодарность кит помогает Нике"}]},
            {"status": "answered", "selections": [{"movie_id": self.film.movie_id,
                "fragment_ids": ["invented-quote"]}]},
            {"status": "answered", "selections": [{"movie_id": self.film.movie_id,
                "fragment_ids": [other.source_id]}]},
            {"status": "answered", "selections": [{"movie_id": "unknown", "fragment_ids": [self.chunk.chunk_id]}]},
            {"status": "answered", "selections": self.selection["selections"] * 2},
            {"status": "answered", "selections": [{"movie_id": self.film.movie_id,
                "fragment_ids": [self.chunk.chunk_id] * 2}]},
            {"status": "answered", "selections": [{"movie_id": self.film.movie_id, "fragment_ids": []}]},
            {"status": "answered", "selections": [{"movie_id": self.film.movie_id, "fragment_ids": [{}]}]},
            {"status": "answered", "selections": []},
            {"status": "insufficient", "selections": self.selection["selections"]},
            {"status": True, "selections": []},
        ]
        for value in variants:
            with self.subTest(value=value), self.assertRaises(GroundingError):
                render_selection(value, QueryPlan("кит"), context)

    def test_supplied_invented_chunk_is_rejected_before_model_call(self):
        bad = replace(self.chunk, text="В благодарность кит помогает Нике")
        with self.assertRaises(GroundingError):
            self.generator.generate("кит", self.plan, (Evidence(self.film, (bad,)),), None)
        self.transport.generate.assert_not_called()

    def test_selected_part_is_not_replaced_by_unretrieved_card_text(self):
        text = "Кит помогает ей добраться до затопленной мастерской, где лежит запасная линза."
        chunk = Chunk(self.film.source_id + ":part-2", self.film, text)
        selection = {"status": "answered", "selections": [
            {"movie_id": self.film.movie_id, "fragment_ids": [chunk.chunk_id]},
        ]}
        answer = render_selection(selection, self.plan, (Evidence(self.film, (chunk,)),))
        self.assertEqual(answer.recommendations[0].reason, f"«{text}»")
        self.assertEqual(answer.source_ids, (self.film.source_id, chunk.chunk_id))

    def test_whole_fragment_keeps_negation_and_refuses_filter_or_mixed_card_context(self):
        film = replace(self.film, synopsis="Это не история про космос. Герои ремонтируют поезд.")
        chunk = Chunk(film.source_id, film, film.synopsis)
        answer = render_selection(self.selection, self.plan, (Evidence(film, (chunk,)),))
        self.assertIn("Это не история про космос.", answer.recommendations[0].reason)
        invalid = [
            (Evidence(replace(self.film, duration_minutes=90), ()),),
            (Evidence(self.film, (replace(self.chunk, film=self.films[1]),)),),
            (*self.context, *self.context),
        ]
        for context in invalid:
            with self.subTest(context=context), self.assertRaises(GroundingError):
                source_fragments(self.plan, context)

    def test_exact_counts_and_zero_count_bypass_gemini(self):
        for limit, value in ((None, 40), (90, 30), (19, 0)):
            with self.subTest(limit=limit):
                filters = Filters(duration_minutes_lt=limit)
                context = tuple(Evidence(film, ()) for film in self.films if filters.matches(film))
                sources = tuple(item.film.source_id for item in context)
                answer = self.generator.generate("сколько фильмов", QueryPlan("", Intent.COUNT, filters),
                                                 context, ExactResult(value, "movies", sources))
                self.assertIs(answer.status, Status.ANSWERED)
                self.assertIn(str(value), answer.text)
                self.assertEqual(answer.source_ids, sources)
        self.transport.generate.assert_not_called()

    def test_exact_comparison_sign_and_lookup_bypass_gemini(self):
        filters = Filters(movie_ids=(self.film.movie_id,))
        for budget, expected in ((72, "не подходит: превышение на 6"),
                                 (78, "укладывается точно"), (90, "остаётся 12")):
            with self.subTest(budget=budget):
                plan = QueryPlan("", Intent.COMPARE_DURATION, filters, budget)
                answer = self.generator.generate("подойдёт?", plan, self.context,
                    ExactResult(self.film.duration_minutes - budget, "minutes", (self.film.source_id,)))
                self.assertIn(expected, answer.text)
                self.assertIn("78 минут", answer.text)
        answer = self.generator.generate("длительность?", QueryPlan("", Intent.LOOKUP, filters), self.context, None)
        self.assertIn("78 минут", answer.text)
        self.assertEqual(answer.source_ids, (self.film.source_id,))
        self.transport.generate.assert_not_called()

    def test_named_plot_questions_return_the_full_source_via_lookup_without_providers(self):
        retriever = Mock()
        retriever.search.side_effect = AssertionError("Lookup must not run semantic retrieval")
        self.transport.generate.side_effect = AssertionError("Lookup must not call Gemini")
        service = SearchService(build_planner(self.films), retriever,
                                build_exact_catalog(self.films), self.generator)
        examples = (
            ('Как дракон помог пекарю в „Пекарня дракона“, когда сломалась печь?', 'film-10'),
            ('Расскажи сюжет «Маяк для кита».', 'film-01'),
            ('О чём мультфильм „Маяк для кита“?', 'film-01'),
        )
        for question, movie_id in examples:
            with self.subTest(question=question):
                film = next(f for f in self.films if f.movie_id == movie_id)
                trace = service.ask(question)
                self.assertIs(trace.plan.intent, Intent.LOOKUP)
                self.assertIs(trace.answer.status, Status.ANSWERED)
                self.assertIn(f"«{film.synopsis}»", trace.answer.text)
                self.assertIn(str(film.duration_minutes), trace.answer.text)
                self.assertEqual(trace.answer.source_ids, (film.source_id,))
                self.assertEqual(trace.context[0].film, film)
                if movie_id == 'film-10':
                    self.assertIn('собственным дыханием', trace.answer.text)
        retriever.search.assert_not_called()
        self.transport.generate.assert_not_called()

    def test_inconsistent_computed_results_cannot_be_rendered(self):
        plan = QueryPlan("", Intent.COUNT)
        for value in (ExactResult(40, "movies", (self.film.source_id,)),
                      ExactResult(True, "movies", (self.film.source_id,)),
                      ExactResult(1, "movies", ("other",)), ExactResult(1, "minutes", (self.film.source_id,))):
            with self.subTest(result=value), self.assertRaises(GroundingError):
                self.generator.generate("count", plan, self.context, value)
        with self.assertRaises(GroundingError):
            self.generator.generate("compare", QueryPlan("", Intent.COMPARE_DURATION,
                Filters(movie_ids=(self.film.movie_id,)), 90), self.context,
                ExactResult(12, "minutes", (self.film.source_id,)))
        self.transport.generate.assert_not_called()

    def test_technical_and_selection_failures_do_not_become_insufficient(self):
        self.transport.generate.side_effect = GenerationUnavailable("HTTP 503", http_status=503)
        with self.assertRaises(GenerationUnavailable):
            self.generator.generate("кит", self.plan, self.context, None)
        self.transport.generate.side_effect = None
        self.transport.generate.return_value = envelope(self.selection, finish="MAX_TOKENS")
        with self.assertRaises(GenerationUnavailable):
            self.generator.generate("кит", self.plan, self.context, None)
        self.transport.generate.return_value = envelope({**self.selection, "text": "invented"})
        with self.assertRaises(GroundingError):
            self.generator.generate("кит", self.plan, self.context, None)

    def test_nonanswered_selections_have_fixed_text_and_empty_evidence(self):
        for status in ("insufficient", "clarify"):
            answer = render_selection({"status": status, "selections": []}, self.plan, self.context)
            self.assertEqual(answer.status.value, status)
            self.assertEqual(answer.source_ids, ())
            self.assertEqual(answer.recommendations, ())
        answer = self.generator.generate("кит", self.plan, (), None)
        self.assertIs(answer.status, Status.INSUFFICIENT)
        self.transport.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
