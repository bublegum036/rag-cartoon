"""Offline regression: archived question, mocked reason, real decoder/renderer."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

from cartoon_rag.adapters.gemini_planner import (
    GeminiQueryPlanner, PLAN_SCHEMA, PLANNER_DECODER_VERSION, PLANNER_INSTRUCTION,
    PLANNER_PROMPT_VERSION, PLANNER_VERSION,
)
from cartoon_rag.application import SearchService
from cartoon_rag.early_replies import EARLY_REPLY_VERSION
from cartoon_rag.ingestion import load_catalog


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'artifacts/llm-planner-2026-11-12/early-recheck-v1.1'


def main():
    case = json.loads((ARCHIVE / 'year-filter/case.json').read_text())
    old_answer = json.loads((ARCHIVE / 'year-filter/answer.json').read_text())
    proposed = {
        'semantic_query': 'фильм про море', 'intent': 'search',
        'filters': {'duration_minutes_lt': None, 'duration_minutes_lte': None, 'movie_ids': []},
        'time_budget_minutes': None, 'early_status': 'clarify',
        'early_reason': 'unsupported_year_filter',
    }
    transport = Mock()
    transport.generate.return_value = {'candidates': [{'finishReason': 'STOP', 'content': {
        'parts': [{'text': json.dumps(proposed, ensure_ascii=False)}]}}]}
    films = load_catalog(ROOT / 'catalog.json')
    assert all(type(film.year) is int for film in films)
    planner = GeminiQueryPlanner(transport, {film.movie_id: film.title for film in films})
    retriever, catalog, generator = Mock(), Mock(), Mock()
    trace = SearchService(planner, retriever, catalog, generator).ask(case['question'])
    assert trace.answer.status.value == 'clarify'
    assert trace.answer.text == ('Данные о годе есть, но фильтрация по году пока не реализована. '
                                 'Уточните запрос без этого условия.')
    retriever.search.assert_not_called()
    catalog.execute.assert_not_called()
    generator.generate.assert_not_called()
    transport.generate.assert_called_once()
    sources = ('adapters/gemini_planner.py', 'early_replies.py', 'bootstrap.py', 'experiments.py')
    record = {
        'verification': 'offline-mocked-reason-v1', 'classification_source': 'mock transport, not Gemini',
        'real_provider_calls': 0, 'question': case['question'], 'old_archived_answer': old_answer,
        'mocked_proposed_plan': proposed, 'new_trace': asdict(trace),
        'versions': {'planner': PLANNER_VERSION, 'prompt': PLANNER_PROMPT_VERSION,
                     'decoder': PLANNER_DECODER_VERSION, 'renderer': EARLY_REPLY_VERSION},
        'source_sha256': {name: hashlib.sha256((ROOT / 'src/cartoon_rag' / name).read_bytes()).hexdigest()
                          for name in sources},
        'verification_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUTPUT / 'regression.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    (OUTPUT / 'planner-schema.json').write_text(json.dumps(PLAN_SCHEMA, ensure_ascii=False, indent=2) + '\n')
    (OUTPUT / 'planner-prompt.txt').write_text(PLANNER_INSTRUCTION)
    print(trace.answer.text)


if __name__ == '__main__':
    main()
