"""Read-only provider check; archives retrieval and exact evidence, never answers.

Run from the repository root with HF_HUB_OFFLINE=1 and PYTHONPATH=src.
Requires the already published cartoons-expanded alias and index-before.json.
Refuses to overwrite a completed verification.
"""

import json
from dataclasses import asdict, replace
from pathlib import Path
from urllib.parse import quote

from cartoon_rag.application import RetrievalService, execute_exact
from cartoon_rag.bootstrap import build_exact_catalog, build_planner, build_search
from cartoon_rag.domain import Filters, Intent, QueryPlan
from cartoon_rag.evaluation import load_cases
from cartoon_rag.ingestion import active_snapshot, load_catalog
from cartoon_rag.settings import Settings
from cartoon_rag.snapshots import catalog_fingerprint, fingerprint


ROOT = Path("artifacts/expanded-retrieval-2026-10-26")


def save(name, data):
    (ROOT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    if (ROOT / "retrieval-metrics.json").exists():
        raise ValueError("Verification already completed; use a new output directory")
    before = json.loads((ROOT / "index-before.json").read_text())
    settings = replace(Settings.load(), qdrant_alias="cartoons-expanded")
    films = active_snapshot(load_catalog(Path("data/catalog-expanded.json")))
    cards = {film.movie_id: film for film in films}
    cases = load_cases(Path("data/questions-expanded.jsonl"))
    _, index = build_search(settings)
    pinned = index.bind_snapshot(catalog_fingerprint(films))
    retrieval = RetrievalService(pinned)
    planner, catalog = build_planner(films), build_exact_catalog(films)
    reference_scores, semantic_scores, rows = [], [], []
    filtered, compliant = 0, 0
    for case in cases:
        plan = planner.plan(case.question)
        hits, context, exact = (), (), None
        if plan.early_status is not None:
            assert plan.early_status is case.expected_status
            route = "early"
        elif plan.intent is Intent.SEARCH:
            result = retrieval.search(plan, limit=10)
            hits, context = result.retrieved, result.context[:3]
            route = "semantic"
        else:
            result = execute_exact(catalog, plan)
            context, exact = result.context, result.exact_result
            route = "exact"
        for film in [hit.chunk.film for hit in hits] + [item.film for item in context]:
            assert film == cards[film.movie_id]
            assert case.filters.matches(film), case.id
        if case.filters != Filters():
            filtered += 1
            compliant += 1  # The independent reference filters passed above.
        if case.id == "no-match-under19":
            assert not hits and not context
            assert not any(case.filters.matches(film) for film in films)
        sources = [hit.chunk.film.source_id for hit in hits] if route == "semantic" else [item.film.source_id for item in context]
        recall = None
        if case.relevant_source_ids:
            relevant = set(case.relevant_source_ids)
            recall = {k: len(relevant & set(sources[:k])) / len(relevant) for k in (1, 3)}
            reference_scores.append(recall)
            if route == "semantic":
                semantic_scores.append(recall)
        folder = ROOT / case.id
        folder.mkdir(exist_ok=True)
        for name, value in (("case", asdict(case)), ("plan", asdict(plan)),
                            ("retrieved", [asdict(hit) for hit in hits]),
                            ("context", [asdict(item) for item in context]),
                            ("exact-result", asdict(exact) if exact is not None else None)):
            save(f"{case.id}/{name}.json", value)
        rows.append({"case_id": case.id, "case_sha256": fingerprint(asdict(case)),
                     "route": route, "retrieved_source_ids": sources,
                     "recall_at_1": recall[1] if recall else None,
                     "recall_at_3": recall[3] if recall else None})

    boundaries = {}
    for label, filters in (("lt90", Filters(duration_minutes_lt=90)),
                           ("lte90", Filters(duration_minutes_lte=90))):
        hits = pinned.search(QueryPlan("коралловый риф свежая вода", filters=filters), limit=40)
        expected = {film.movie_id for film in films if filters.matches(film)}
        actual = {hit.chunk.film.movie_id for hit in hits}
        assert actual == expected
        assert ("film-26" in actual) == (label == "lte90")
        boundaries[label] = [asdict(hit) for hit in hits]
    save("duration-boundaries.json", boundaries)

    aliases = index.transport.request("GET", "/aliases")["aliases"]
    mapping = {item["alias_name"]: item["collection_name"] for item in aliases}
    assert all(mapping[item["alias_name"]] == item["collection_name"] for item in before["aliases"])
    primary = index.transport.request("GET", "/collections/" + quote(before["primary_collection"], safe=""))
    assert primary["points_count"] == before["primary_points_count"]
    assert primary["config"]["metadata"] == before["primary_metadata"]
    expanded = index.transport.request("GET", "/collections/" + quote(pinned.collection, safe=""))
    assert expanded["points_count"] == 40
    assert expanded["config"]["metadata"]["catalog_sha256"] == catalog_fingerprint(films)
    save("index-after.json", {"aliases": aliases, "primary_preserved": True,
                               "expanded_points_count": expanded["points_count"],
                               "expanded_metadata": expanded["config"]["metadata"]})
    save("retrieval-results.json", rows)
    metrics = {"cases": len(cases), "reference_cases": len(reference_scores),
               "semantic_reference_cases": len(semantic_scores),
               "recall_at_1": sum(r[1] for r in reference_scores) / len(reference_scores),
               "recall_at_3": sum(r[3] for r in reference_scores) / len(reference_scores),
               "semantic_recall_at_1": sum(r[1] for r in semantic_scores) / len(semantic_scores),
               "semantic_recall_at_3": sum(r[3] for r in semantic_scores) / len(semantic_scores),
               "strict_filter_cases": filtered, "strict_filter_compliant": compliant,
               "answer_acceptance_rate": None, "gemini_calls": 0,
               "primary_index_preserved": True, "alias": settings.qdrant_alias,
               "snapshot_id": pinned.collection, "catalog_sha256": pinned.catalog_sha256}
    save("retrieval-metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
