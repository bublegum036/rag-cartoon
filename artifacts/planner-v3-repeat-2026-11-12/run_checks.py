"""Authorized live verification: isolated DB tests and planner-only Gemini calls."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from cartoon_rag.adapters.gemini import GeminiHTTP, decode_generated_json
from cartoon_rag.adapters.gemini_planner import (
    GeminiQueryPlanner, PLAN_SCHEMA, PLANNER_DECODER_VERSION, PLANNER_INSTRUCTION,
    PLANNER_PROMPT_VERSION, PLANNER_VERSION,
)
from cartoon_rag.adapters.json_catalog import JsonExactCatalog
from cartoon_rag.adapters.qdrant import QdrantHTTP
from cartoon_rag.application import early_trace
from cartoon_rag.domain import GenerationUnavailable, Intent
from cartoon_rag.early_replies import EARLY_REPLY_VERSION
from cartoon_rag.ingestion import active_snapshot, load_catalog
from cartoon_rag.settings import Settings
from cartoon_rag.snapshots import catalog_fingerprint, fingerprint


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_hashes():
    base = ROOT / 'src/cartoon_rag'
    return {str(path.relative_to(base)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(base.rglob('*.py'))}


def snapshots(settings):
    db = QdrantHTTP(settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=settings.timeout_seconds)
    result = {}
    for alias in db.request('GET', '/aliases')['aliases']:
        if alias['alias_name'] in ('cartoons', 'cartoons-expanded'):
            name = alias['collection_name']
            info = db.request('GET', '/collections/' + name)
            count = db.request('POST', '/collections/' + name + '/points/count', {'exact': True})['count']
            result[alias['alias_name']] = {'collection': name, 'metadata': info['config']['metadata'], 'count': count}
    return result


class AuditTransport:
    def __init__(self, settings):
        self.delegate = GeminiHTTP(api_key=settings.gemini_api_key, model=settings.gemini_model,
                                   timeout=settings.generation_timeout_seconds)
        self.last_started = 0.0
        self.folder = None
        self.calls = []

    def generate(self, body):
        gap = 10 - (time.monotonic() - self.last_started)
        if gap > 0:
            time.sleep(gap)
        self.last_started = time.monotonic()
        write(self.folder / 'request.json', body)  # Body contains no API key or headers.
        record = {'case_id': self.folder.name, 'state': 'started', 'request_sha256': fingerprint(body)}
        self.calls.append(record)
        write(OUTPUT / 'provider-calls.json', self.calls)
        try:
            response = self.delegate.generate(body)  # Exactly one request, no retry/fallback.
        except GenerationUnavailable as exc:
            record.update(state='error', type=type(exc).__name__, message=str(exc), http_status=exc.http_status)
            raise
        else:
            record.update(state='response', response_sha256=fingerprint(response))
            try:
                proposal = decode_generated_json(response)
            except GenerationUnavailable:
                proposal = None
            write(self.folder / 'proposed-plan.json', proposal)
            return response
        finally:
            record['elapsed_seconds'] = round(time.monotonic() - self.last_started, 3)
            write(OUTPUT / 'provider-calls.json', self.calls)


def planner_check(settings, manifest):
    document = json.loads((OUTPUT / 'cases.json').read_text())
    films = active_snapshot(load_catalog(ROOT / 'data/catalog-expanded.json'))
    manifest.update(cases_version=document['version'], cases_sha256=fingerprint(document),
                    catalog_sha256=catalog_fingerprint(films), catalog='data/catalog-expanded.json',
                    planner_version=PLANNER_VERSION, planner_prompt_version=PLANNER_PROMPT_VERSION,
                    planner_decoder_version=PLANNER_DECODER_VERSION, planner_renderer_version=EARLY_REPLY_VERSION,
                    model=settings.gemini_model, max_output_tokens=2048, temperature=0,
                    minimum_start_interval_seconds=10, final_answer_generation=False)
    write(OUTPUT / 'planner-manifest.json', manifest)
    (OUTPUT / 'planner-prompt.txt').write_text(PLANNER_INSTRUCTION)
    write(OUTPUT / 'planner-schema.json', PLAN_SCHEMA)
    transport = AuditTransport(settings)
    planner = GeminiQueryPlanner(transport, {film.movie_id: film.title for film in films})
    exact, records, stop = JsonExactCatalog(films), [], False
    for case in document['cases']:
        folder = OUTPUT / case['id']
        folder.mkdir(exist_ok=False)
        transport.folder = folder
        record = {'id': case['id']}
        if stop:
            record.update(state='not_executed', reason='Stopped after access/quota error')
        else:
            try:
                plan = planner.plan(case['question'])  # Only question and titles enter the model.
                write(folder / 'plan.json', asdict(plan))
                projection = {'intent': plan.intent, 'filters': asdict(plan.filters),
                              'time_budget_minutes': plan.time_budget_minutes, 'early_status': plan.early_status}
                projection = json.loads(json.dumps(projection))
                raw = json.loads((folder / 'proposed-plan.json').read_text())
                checks = {'plan_matches': projection == case['expected_plan'],
                          'reason_matches': raw.get('early_reason') == case['expected_reason'],
                          'provider_free_text_absent': 'early_text' not in raw}
                reply = early_trace(case['question'], plan)
                if reply is not None:
                    write(folder / 'early-answer.json', asdict(reply.answer))
                else:
                    write(folder / 'year-candidates.json', [
                        {'movie_id': film.movie_id, 'year': film.year, 'duration_minutes': film.duration_minutes}
                        for film in films if plan.filters.matches(film)])
                if case['expected_count'] is not None:
                    value = exact.execute(plan)[1] if plan.intent is Intent.COUNT and plan.early_status is None else None
                    write(folder / 'exact-result.json', asdict(value) if value else None)
                    checks['exact_count_matches'] = value is not None and value.value == case['expected_count']
                record.update(state='completed', checks=checks, automatic_pass=all(checks.values()))
            except (GenerationUnavailable, ValueError) as exc:
                record.update(state='error', type=type(exc).__name__, message=str(exc),
                              http_status=getattr(exc, 'http_status', None))
                write(folder / 'error.json', record)
                if isinstance(exc, GenerationUnavailable) and exc.http_status in (401, 403, 429):
                    stop = True
        records.append(record)
        write(OUTPUT / 'planner-results.json', records)
        print(case['id'] + ': ' + record['state'], flush=True)
    summary = {'cases': len(records), 'completed': sum(r['state'] == 'completed' for r in records),
               'errors': sum(r['state'] == 'error' for r in records),
               'not_executed': sum(r['state'] == 'not_executed' for r in records),
               'automatic_pass': sum(r.get('automatic_pass', False) for r in records),
               'provider_calls': len(transport.calls), 'final_answer_generation_calls': 0}
    manifest.update(state='completed' if not summary['errors'] and not stop else 'completed_with_errors',
                    finished_at=datetime.now(timezone.utc).isoformat())
    write(OUTPUT / 'planner-summary.json', summary)
    write(OUTPUT / 'planner-manifest.json', manifest)
    print(json.dumps(summary), flush=True)
    return 1 if summary['errors'] or summary['not_executed'] or summary['automatic_pass'] != len(records) else 0


def test_check(settings, manifest):
    before = snapshots(settings)
    environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'QDRANT_INTEGRATION_URL': settings.qdrant_url,
                   'RUN_SEMANTIC_SMOKE': '1', 'PYTHONPATH': 'src', 'QDRANT_API_KEY': settings.qdrant_api_key}
    command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']
    with (OUTPUT / 'tests-live.log').open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    after = snapshots(settings)
    write(OUTPUT / 'index-snapshots.json', {'before': before, 'after': after})
    manifest.update(state='passed' if result.returncode == 0 and before == after else 'failed',
                    exit_code=result.returncode, indices_preserved=before == after,
                    finished_at=datetime.now(timezone.utc).isoformat())
    write(OUTPUT / 'tests-manifest.json', manifest)
    print('Integration suite: ' + manifest['state'], flush=True)
    return result.returncode or int(before != after)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('tests', 'planner'), required=True)
    phase = parser.parse_args().phase
    manifest = {'phase': phase, 'state': 'started', 'created_at': datetime.now(timezone.utc).isoformat(),
                'python': platform.python_version(), 'source_sha256': source_hashes(),
                'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    path = OUTPUT / (phase + '-manifest.json')
    with path.open('x') as stream:
        json.dump(manifest, stream, indent=2)
    settings = Settings.load()
    result = test_check(settings, manifest) if phase == 'tests' else planner_check(settings, manifest)
    raise SystemExit(result)


if __name__ == '__main__':
    main()
