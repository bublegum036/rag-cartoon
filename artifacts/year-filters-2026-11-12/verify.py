"""Offline year plans, exact operations and read-only legacy metric replay."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from cartoon_rag.adapters.json_catalog import JsonExactCatalog
from cartoon_rag.adapters.qdrant import qdrant_filter
from cartoon_rag.bootstrap import build_planner
from cartoon_rag.experiments import load_run, summarize
from cartoon_rag.ingestion import load_catalog
from cartoon_rag.snapshots import catalog_fingerprint


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent


def main():
    films = load_catalog(ROOT / 'data/catalog-expanded.json')
    planner, exact = build_planner(films), JsonExactCatalog(films)
    examples = []
    questions = (
        'Посоветуй фильм про море, выпущенный не раньше 1990 года.',
        'про море с 1980 по 1990 год и короче 90 минут',
        'Сколько фильмов с 1980 по 1990 год и короче 90 минут?',
        'Сколько фильмов в 1900 году?',
        'Что происходит в «Маяк для кита», выпущенном в 2021 году?',
    )
    for question in questions:
        plan = planner.plan(question)
        assert plan.early_status is None
        row = {'question': question, 'plan': asdict(plan), 'qdrant_filter': qdrant_filter(plan.filters)}
        if plan.intent.value != 'search':
            context, result = exact.execute(plan)
            row.update(context_movie_ids=[item.film.movie_id for item in context],
                       exact_result=asdict(result) if result else None)
        examples.append(row)
    assert examples[0]['plan']['filters']['year_gte'] == 1990
    assert examples[2]['exact_result']['value'] == 7
    assert examples[3]['exact_result']['value'] == 0
    legacy = []
    names = ('ab-A-2026-10-25', 'ab-B-2026-10-25', 'expanded-run-v2-2026-10-26',
             'llm-planner-2026-11-12/A-rules', 'llm-planner-2026-11-12/B-gemini',
             'llm-planner-2026-11-12/paraphrases-rules', 'llm-planner-2026-11-12/paraphrases-gemini',
             'llm-planner-2026-11-12/early-recheck-v1.1')
    for name in names:
        path = ROOT / 'artifacts' / name
        before = {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in path.rglob('*') if p.is_file()}
        manifest, records, cases, traces, preparations, judge = load_run(path, path / 'reviews.json')
        metrics = summarize(cases, records, traces, k=manifest['recall_k'], judge=judge,
                            preparations=preparations if manifest.get('artifact_schema_version', 1) >= 2 else None)
        assert metrics == json.loads((path / 'metrics.json').read_text())
        after = {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in path.rglob('*') if p.is_file()}
        assert before == after
        legacy.append({'run': name, 'metrics_unchanged': True, 'files_unchanged': True})
    sources = ('domain.py', 'serialization.py', 'adapters/rule_planner.py', 'adapters/gemini_planner.py',
               'adapters/qdrant.py', 'evaluation.py', 'experiments.py', 'quality.py', 'early_replies.py')
    output = {'verification_version': 'year-filters-v1', 'catalog_sha256': catalog_fingerprint(films),
              'planner_version': planner.version, 'provider_calls': 0, 'examples': examples,
              'legacy': legacy, 'source_sha256': {
                  name: hashlib.sha256((ROOT / 'src/cartoon_rag' / name).read_bytes()).hexdigest() for name in sources}}
    (OUTPUT / 'verification.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print('Year plans/exact examples passed; eight legacy runs unchanged; zero provider calls.')


if __name__ == '__main__':
    main()
