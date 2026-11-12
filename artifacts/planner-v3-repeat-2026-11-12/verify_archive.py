"""Offline replay of saved planner proposals; never calls Gemini or Qdrant."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from cartoon_rag.adapters.gemini_planner import decode_plan
from cartoon_rag.adapters.json_catalog import JsonExactCatalog
from cartoon_rag.application import early_trace
from cartoon_rag.ingestion import active_snapshot, load_catalog
from cartoon_rag.snapshots import catalog_fingerprint, fingerprint


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent


def load(name):
    return json.loads((OUTPUT / name).read_text())


def main():
    manifest = load('planner-manifest.json')
    assert manifest['state'] in ('completed', 'completed_with_errors')
    document, records, calls = load('cases.json'), load('planner-results.json'), load('provider-calls.json')
    assert fingerprint(document) == manifest['cases_sha256']
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT / 'src/cartoon_rag' / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((OUTPUT / 'run_checks.py').read_bytes()).hexdigest() == manifest['runner_sha256']
    films = active_snapshot(load_catalog(ROOT / manifest['catalog']))
    assert catalog_fingerprint(films) == manifest['catalog_sha256']
    titles, exact = {film.movie_id: film.title for film in films}, JsonExactCatalog(films)
    cases = {case['id']: case for case in document['cases']}
    assert [record['id'] for record in records] == [case['id'] for case in document['cases']]
    assert len({call['case_id'] for call in calls}) == len(calls)
    for call in calls:
        request = load(call['case_id'] + '/request.json')
        assert fingerprint(request) == call['request_sha256']
        payload = json.loads(request['contents'][0]['parts'][0]['text'])
        assert set(payload) == {'question', 'titles'}
        assert payload['question'] == cases[call['case_id']]['question']
        assert payload['titles'] == [{'movie_id': key, 'title': title} for key, title in sorted(titles.items())]
    replayed = []
    for record in records:
        if record['state'] != 'completed':
            continue  # Preserve error/missing coverage; no inferred plan.
        case, prefix = cases[record['id']], record['id'] + '/'
        request = load(prefix + 'request.json')
        payload = json.loads(request['contents'][0]['parts'][0]['text'])
        assert set(payload) == {'question', 'titles'}
        assert payload['question'] == case['question']
        assert payload['titles'] == [{'movie_id': key, 'title': title} for key, title in sorted(titles.items())]
        proposal = load(prefix + 'proposed-plan.json')
        plan = decode_plan(proposal, titles)
        assert json.loads(json.dumps(asdict(plan))) == load(prefix + 'plan.json')
        projection = json.loads(json.dumps({'intent': plan.intent, 'filters': asdict(plan.filters),
                                            'time_budget_minutes': plan.time_budget_minutes,
                                            'early_status': plan.early_status}))
        checks = {'plan_matches': projection == case['expected_plan'],
                  'reason_matches': proposal.get('early_reason') == case['expected_reason'],
                  'provider_free_text_absent': 'early_text' not in proposal}
        reply = early_trace(case['question'], plan)
        if reply:
            assert json.loads(json.dumps(asdict(reply.answer))) == load(prefix + 'early-answer.json')
        else:
            candidates = [{'movie_id': film.movie_id, 'year': film.year, 'duration_minutes': film.duration_minutes}
                          for film in films if plan.filters.matches(film)]
            assert candidates == load(prefix + 'year-candidates.json')
        if case['expected_count'] is not None:
            from cartoon_rag.domain import Intent
            value = exact.execute(plan)[1] if plan.intent is Intent.COUNT and plan.early_status is None else None
            serialized = json.loads(json.dumps(asdict(value))) if value else None
            assert serialized == load(prefix + 'exact-result.json')
            checks['exact_count_matches'] = value is not None and value.value == case['expected_count']
        assert checks == record['checks']
        assert all(checks.values()) == record['automatic_pass']
        replayed.append(record['id'])
    review = load('planner-review.json')
    for decision in review['decisions']:
        if decision['state'] != 'reviewed':
            assert decision['passed'] is None
            continue
        prefix = decision['case_id'] + '/'
        assert fingerprint(cases[decision['case_id']]) == decision['case_sha256']
        assert fingerprint(load(prefix + 'proposed-plan.json')) == decision['proposal_sha256']
        assert fingerprint(load(prefix + 'plan.json')) == decision['plan_sha256']
    reference = load('reference.json')
    previous = ROOT / reference['prior_run']
    for name, digest in reference['prior_file_sha256'].items():
        assert hashlib.sha256((previous / name).read_bytes()).hexdigest() == digest
    previous_manifest = json.loads((previous / 'planner-manifest.json').read_text())
    assert previous_manifest['source_sha256'] == manifest['source_sha256']
    assert previous_manifest['cases_sha256'] == manifest['cases_sha256']
    for key in ('model', 'planner_version', 'planner_prompt_version', 'planner_decoder_version',
                'planner_renderer_version', 'max_output_tokens', 'temperature', 'minimum_start_interval_seconds'):
        assert previous_manifest[key] == manifest[key]
    for call in calls:
        assert call['request_sha256'] == fingerprint(json.loads((previous / call['case_id'] / 'request.json').read_text()))
    integration = ROOT / reference['integration_run']
    tests = json.loads((integration / 'tests-manifest.json').read_text())
    assert tests['state'] == 'passed' and tests['exit_code'] == 0
    assert tests['source_sha256'] == manifest['source_sha256']
    assert tests['git_head'] == manifest['git_head']
    snapshots = json.loads((integration / 'index-snapshots.json').read_text())
    assert snapshots['before'] == snapshots['after']
    assert tests['indices_preserved']
    print('Offline replay matches saved artifacts for', len(replayed), 'completed planner cases; previous archive unchanged; prior integration result linked.')


if __name__ == '__main__':
    main()
