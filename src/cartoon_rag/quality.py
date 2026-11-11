"""Offline stage diagnostics with independent, versioned routing annotations."""

import hashlib
import json
import platform
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from cartoon_rag.application import validate_answer
from cartoon_rag.bootstrap import build_planner
from cartoon_rag.domain import Intent, QueryPlan
from cartoon_rag.evaluation import accepted
from cartoon_rag.experiments import load_run
from cartoon_rag.ingestion import active_snapshot, load_catalog
from cartoon_rag.snapshots import catalog_fingerprint, fingerprint


REPORT_VERSION = "stage-quality-v2"


def plan_fields(plan: QueryPlan) -> dict:
    """Route is distinct from final answered/insufficient/clarify status."""
    return json.loads(json.dumps({
        "intent": plan.intent, "filters": {name: value for name, value in asdict(plan.filters).items()
                                         if name not in ("year_gte", "year_lte") or value is not None},
        "time_budget_minutes": plan.time_budget_minutes, "early_status": plan.early_status,
    }))


def stage_report(run_dir: Path, expectations_path: Path, catalog_path: Path,
                 reviews_path: Path) -> dict:
    manifest, records, cases, traces, preparations, judge = load_run(run_dir, reviews_path)
    gold = json.loads(expectations_path.read_text(encoding="utf-8"))
    films = load_catalog(catalog_path)
    catalog_hash = catalog_fingerprint(films)
    if (gold.get("schema_version") != 1 or not isinstance(gold.get("version"), str)
            or not gold["version"].strip()):
        raise ValueError("Stage expectations require schema 1 and a version")
    if gold["case_set_sha256"] != manifest["case_set_sha256"]:
        raise ValueError("Stage expectations belong to another case set")
    runtime_catalog = manifest.get("runtime", {}).get("catalog_sha256")
    if gold["catalog_sha256"] != catalog_hash or runtime_catalog not in (None, catalog_hash):
        raise ValueError("Stage expectations/run belong to another catalog")
    if set(gold["expectations"]) != {case.id for case in cases}:
        raise ValueError("Stage expectations must cover exactly the archived cases")
    for expected in gold["expectations"].values():
        if set(expected) != {"plan", "exact_result"}:
            raise ValueError("Invalid stage expectation fields")
        plan = expected["plan"]
        if (set(plan) != {"intent", "filters", "time_budget_minutes", "early_status"}
                or plan["intent"] not in {item.value for item in Intent}
                or plan["early_status"] not in (None, "clarify", "insufficient")):
            raise ValueError("Invalid expected plan")
    planner = build_planner(films)
    cards = {film.movie_id: film for film in active_snapshot(films)}
    evidence = {**preparations, **traces}
    results = []
    for case, record in zip(cases, records, strict=True):
        expected = gold["expectations"][case.id]
        plan = expected["plan"]
        prepared, trace = evidence.get(case.id), traces.get(case.id)
        search = plan["early_status"] is None and plan["intent"] == "search"
        exact = plan["early_status"] is None and plan["intent"] in ("count", "compare_duration")
        checks = {}

        def check(name, applicable, score):
            checks[name] = {"state": ("not_applicable" if not applicable else "missing" if score is None
                                      else "pass" if score == 1 else "fail"),
                            "score": score if applicable else None}

        check("current_planner", True, float(plan_fields(planner.plan(case.question)) == plan))
        check("archived_planner", True, float(plan_fields(prepared.plan) == plan) if prepared else None)
        check("completion", True, float(record["state"] == "ok"))
        check("retrieval_recall_at_k", search and bool(case.relevant_source_ids),
              len(set(case.relevant_source_ids)
                  & {hit.chunk.film.source_id for hit in prepared.retrieved[:manifest["recall_k"]]})
              / len(set(case.relevant_source_ids)) if prepared and case.relevant_source_ids else None)
        check("retrieval_filters", search,
              float(all(case.filters.matches(hit.chunk.film) for hit in prepared.retrieved))
              if prepared else None)
        check("context_recall", bool(case.relevant_source_ids),
              len(set(case.relevant_source_ids) & {item.film.source_id for item in prepared.context})
              / len(set(case.relevant_source_ids)) if prepared and case.relevant_source_ids else None)
        check("context_filters", plan["early_status"] is None,
              float(all(case.filters.matches(item.film) for item in prepared.context)) if prepared else None)
        if prepared:
            observed = [hit.chunk.film for hit in prepared.retrieved] + [item.film for item in prepared.context]
            catalog_ok = (
                all(cards.get(film.movie_id) == film for film in observed)
                and all(chunk.film == item.film
                        and cards.get(chunk.film.movie_id) == chunk.film
                        for item in prepared.context for chunk in item.chunks)
            )
        check("catalog_evidence", plan["early_status"] is None, float(catalog_ok) if prepared else None)
        check("exact_result", exact,
              float((json.loads(json.dumps(asdict(prepared.exact_result)))
                     if prepared.exact_result else None) == expected["exact_result"]) if prepared else None)
        check("answer_status", True, float(trace.answer.status is case.expected_status) if trace else None)
        contract = None
        if trace:
            try:
                validate_answer(trace.answer, trace.context,
                                exact_count=trace.exact_result if trace.plan.intent is Intent.COUNT else None)
            except ValueError:
                contract = 0.0
            else:
                contract = 1.0
        check("answer_contract", True, contract)
        reviewed = trace is not None and case.id in judge.decisions
        check("answer_semantics", True, float(judge.accepts(case, trace)) if reviewed else None)
        check("answer_acceptance", True, float(accepted(case, trace, judge)) if reviewed else None)
        results.append({"id": case.id, "expected_route": plan["early_status"] or plan["intent"],
                        "actual_route": (prepared.plan.early_status or prepared.plan.intent) if prepared else None,
                        "expected_status": case.expected_status,
                        "actual_status": trace.answer.status if trace else None, "checks": checks})
    metrics = {}
    for name in results[0]["checks"]:
        checks = [result["checks"][name] for result in results]
        counts = Counter(check["state"] for check in checks)
        scores = [check["score"] for check in checks if check["score"] is not None]
        applicable = len(checks) - counts["not_applicable"]
        metrics[name] = {state: counts[state] for state in ("pass", "fail", "missing", "not_applicable")}
        metrics[name].update(evaluated=len(scores), applicable=applicable,
                             observed_mean=sum(scores) / len(scores) if scores else None,
                             coverage=len(scores) / applicable if applicable else None)
    sources = {str(path.relative_to(Path(__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(Path(__file__).parent.rglob("*.py"))}
    return {"report_version": REPORT_VERSION, "expectations_version": gold["version"],
            "provenance": {"python": platform.python_version(), "current_planner_mode": "rules",
                           "catalog_sha256": catalog_hash,
                           "expectations_sha256": fingerprint(gold), "manifest_sha256": fingerprint(manifest),
                           "results_sha256": fingerprint(records), "reviews_sha256": fingerprint(judge.document),
                           "source_fingerprints": sources, "original_runtime": manifest.get("runtime", {})},
            "metrics": metrics, "cases": results}
