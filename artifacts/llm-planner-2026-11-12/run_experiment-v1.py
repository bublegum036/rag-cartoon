"""Authorized experiment: no retries/fallback; pace every Gemini request."""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from cartoon_rag.adapters.qdrant import QdrantHTTP
from cartoon_rag.adapters.gemini_planner import PLAN_SCHEMA, PLANNER_INSTRUCTION
from cartoon_rag.bootstrap import build_rag_runtime
from cartoon_rag.evaluation import load_cases
from cartoon_rag.experiments import run_evaluation
from cartoon_rag.ingestion import load_catalog
from cartoon_rag.settings import Settings
from cartoon_rag.snapshots import fingerprint


class AuditTransport:
    def __init__(self, transport, phase, ledger, pacing):
        self.transport, self.phase, self.ledger, self.pacing = transport, phase, ledger, pacing

    def generate(self, body):
        gap = 10 - (time.monotonic() - self.pacing[0])
        if gap > 0:
            time.sleep(gap)
        self.pacing[0] = time.monotonic()
        record = {'number': len(self.ledger) + 1, 'phase': self.phase,
                  'body_sha256': fingerprint(body), 'state': 'started'}
        self.ledger.append(record)
        self.save()
        started = time.monotonic()
        try:
            result = self.transport.generate(body)
        except Exception as exc:
            record.update(state='error', error_type=type(exc).__name__, http_status=getattr(exc, 'http_status', None))
            raise
        else:
            record.update(state='response', response_sha256=fingerprint(result))
            return result
        finally:
            record['elapsed_seconds'] = round(time.monotonic() - started, 3)
            self.save()

    def save(self):
        path = self.output / 'provider-calls.json'
        path.write_text(json.dumps(self.ledger, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('rules', 'gemini'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--questions', type=Path, default=Path('questions.jsonl'))
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    settings = Settings.load()
    ledger, pacing = [], [0.0]
    database = QdrantHTTP(settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=settings.timeout_seconds)

    def snapshot():
        result = {}
        for alias in database.request('GET', '/aliases')['aliases']:
            if alias['alias_name'] in ('cartoons', 'cartoons-expanded'):
                name = alias['collection_name']
                info = database.request('GET', '/collections/' + name)
                count = database.request('POST', '/collections/' + name + '/points/count', {'exact': True})['count']
                result[alias['alias_name']] = {'collection': name, 'count': count,
                                             'metadata': info['config']['metadata']}
        return result

    before = snapshot()

    def factory():
        service, identity = build_rag_runtime(load_catalog(Path('catalog.json')), settings, planner_mode=args.mode)
        if args.mode == 'gemini':
            wrapper = AuditTransport(service.planner.transport, 'planner', ledger, pacing)
            wrapper.output = args.output
            service.planner.transport = wrapper
        wrapper = AuditTransport(service.generator.transport, 'answer', ledger, pacing)
        wrapper.output = args.output
        service.generator.transport = wrapper
        return service, identity

    metrics = run_evaluation(load_cases(args.questions), factory, args.output, label=args.mode,
                             delay_seconds=0, progress=lambda case, state: print(case, state, flush=True))
    after = snapshot()
    (args.output / 'index-snapshots.json').write_text(json.dumps({'before': before, 'after': after}, indent=2) + '\n')
    assert before == after, 'Published index changed during experiment'
    (args.output / 'provider-calls.json').write_text(json.dumps(ledger, indent=2) + '\n')
    (args.output / 'planner-prompt.txt').write_text(PLANNER_INSTRUCTION)
    (args.output / 'planner-schema.json').write_text(json.dumps(PLAN_SCHEMA, indent=2) + '\n')
    (args.output / 'runner-sha256.txt').write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() + '\n')
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    if metrics['technical_errors'] or metrics['not_executed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
