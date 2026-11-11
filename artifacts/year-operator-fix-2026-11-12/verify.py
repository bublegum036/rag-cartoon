"""Replay the operator defect and its fix against the unchanged expanded catalog."""

import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

from cartoon_rag.adapters.json_catalog import JsonExactCatalog
from cartoon_rag.adapters.qdrant import qdrant_filter
from cartoon_rag.bootstrap import build_planner
from cartoon_rag.ingestion import load_catalog
from cartoon_rag.snapshots import catalog_fingerprint


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent


def main():
    source = OUTPUT / 'rule_planner-before-fix.py'
    module_spec = importlib.util.spec_from_file_location('historical_rule_planner', source)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    films = load_catalog(ROOT / 'data/catalog-expanded.json')
    old = module.RuleBasedQueryPlanner({film.movie_id: film.title for film in films})
    new, exact = build_planner(films), JsonExactCatalog(films)
    expected = sorted(film.movie_id for film in films if film.year >= 1990)
    assert len(expected) == 20
    examples = []
    for operator in ('с', 'не ранее'):
        question = f'про море {operator} 1990 года'
        before, after = old.plan(question), new.plan(question)
        before_ids = sorted(f.movie_id for f in films if before.filters.matches(f))
        after_ids = sorted(f.movie_id for f in films if after.filters.matches(f))
        assert before.filters.year_gte == before.filters.year_lte == 1990
        assert len(before_ids) == 1
        assert after.early_status is None and after.filters.year_gte == 1990 and after.filters.year_lte is None
        assert after_ids == expected
        count_plan = new.plan(f'Сколько фильмов {operator} 1990 года?')
        context, result = exact.execute(count_plan)
        assert result.value == 20
        assert sorted(item.film.movie_id for item in context) == expected
        examples.append({'question': question, 'before_plan': asdict(before), 'after_plan': asdict(after),
                         'before_year_candidates': before_ids, 'after_year_candidates': after_ids,
                         'qdrant_filter': qdrant_filter(after.filters), 'exact_count': asdict(result)})
    record = {'verification_version': 'year-operator-fix-v1', 'provider_calls': 0,
              'planner_before': old.version, 'planner_after': new.version,
              'catalog_sha256': catalog_fingerprint(films), 'examples': examples,
              'note': 'Candidate membership checks the year bound, not semantic relevance to the sea topic.',
              'before_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'after_source_sha256': hashlib.sha256((ROOT / 'src/cartoon_rag/adapters/rule_planner.py').read_bytes()).hexdigest()}
    (OUTPUT / 'regression.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print('Both operators: before =1990 (1 candidate); after >=1990 (20 candidates), exact count=20.')


if __name__ == '__main__':
    main()
