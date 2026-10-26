"""Auditable evaluation runs and separately supplied semantic reviews."""

import json
import hashlib
import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from cartoon_rag.application import validate_answer
from cartoon_rag.domain import GenerationUnavailable, Intent, PreparedQuery, Trace
from cartoon_rag.evaluation import (
    Case, ReviewedSemanticJudge, accepted, case_from_dict, recall_at_k,
    prepared_filters_compliant, prepared_from_dict, strict_filters_compliant, trace_from_dict,
)
from cartoon_rag.ports import QuestionAnswerer
from cartoon_rag.snapshots import fingerprint


def _write(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _error(exc: Exception) -> dict:
    # Project/provider exceptions have redacted messages. Unexpected exceptions
    # retain their type without exposing arbitrary request/header state.
    from cartoon_rag.domain import EmbeddingUnavailable, RetrievalUnavailable

    known = isinstance(exc, (GenerationUnavailable, EmbeddingUnavailable, RetrievalUnavailable, ValueError, OSError))
    return {"type": type(exc).__name__, "message": str(exc) if known else "Unexpected application error",
            "http_status": getattr(exc, "http_status", None)}


def _retrieved_ids(data: dict) -> list[dict]:
    return [{"rank": i, "chunk_id": hit["chunk"]["chunk_id"],
             "source_id": hit["chunk"]["film"]["source_id"],
             "movie_id": hit["chunk"]["film"]["movie_id"], "score": hit["score"]}
            for i, hit in enumerate(data["retrieved"], 1)]


def _save_prepared(folder: Path, data: dict) -> None:
    for name, value in (("plan", data["plan"]), ("retrieved", data["retrieved"]),
                        ("context", data["context"]), ("exact-result", data["exact_result"]),
                        ("retrieved-ids", _retrieved_ids(data)), ("prepared", data)):
        _write(folder / (name + ".json"), value)


def _check_prepared_files(folder: Path, data: dict) -> None:
    for name, value in (("plan", data["plan"]), ("retrieved", data["retrieved"]),
                        ("context", data["context"]), ("exact-result", data["exact_result"]),
                        ("retrieved-ids", _retrieved_ids(data))):
        if json.loads((folder / (name + ".json")).read_text(encoding="utf-8")) != value:
            raise ValueError("Separate prepared artifacts differ from preparation")


def summarize(
    cases: Sequence[Case], records: Sequence[dict], traces: dict[str, Trace], *, k: int,
    judge: ReviewedSemanticJudge | None = None,
    preparations: dict[str, PreparedQuery] | None = None,
) -> dict:
    if k < 1:
        raise ValueError("Recall k must be positive")
    reference_cases = [case for case in cases if case.relevant_source_ids]
    filtered = [case for case in cases if case.filters.duration_minutes_lt is not None
                or case.filters.duration_minutes_lte is not None or case.filters.movie_ids]
    reviewed, passed = 0, 0
    for case in cases:
        if case.id in traces and judge is not None and case.id in judge.decisions:
            # Validate the review binding even if an automatic check already fails.
            judge.accepts(case, traces[case.id])
            reviewed += 1
            passed += int(accepted(case, traces[case.id], judge))
    pending = len(traces) - reviewed

    evidence = {**(preparations or {}), **traces}

    def recall(limit):
        scores = [recall_at_k(case, evidence[case.id], limit) if case.id in evidence else 0.0
                  for case in reference_cases]
        return sum(scores) / len(scores) if scores else None

    def complies(case):
        if case.id in traces:
            return strict_filters_compliant(case, traces[case.id])
        if preparations is not None and case.id in preparations:
            return prepared_filters_compliant(case, preparations[case.id])
        return False

    compliant = sum(complies(case) for case in filtered)
    metrics = {
        "cases": len(cases), "completed": len(traces),
        "technical_errors": sum(record["state"] == "error" for record in records),
        "not_executed": sum(record["state"] == "not_executed" for record in records),
        "reference_cases": len(reference_cases), "recall_k": k, "recall_at_k": recall(k),
        "recall_at_1": recall(1), "strict_filter_cases": len(filtered),
        "strict_filter_compliant": compliant,
        "strict_filter_compliance": compliant / len(filtered) if filtered else None,
        "reviewed_cases": reviewed, "pending_reviews": pending, "accepted_answers": passed,
        "answer_acceptance_rate": passed / len(cases) if cases and not pending else None,
    }
    if preparations is not None:
        metrics["prepared_cases"] = len(preparations)
    return metrics


def run_evaluation(
    cases: Sequence[Case], factory: Callable[[], tuple[QuestionAnswerer, dict]], output: Path, *,
    label: str, k: int = 3, delay_seconds: float = 0,
    wait: Callable[[float], None] = time.sleep, progress: Callable[[str, str], None] | None = None,
) -> dict:
    if k < 1 or not cases or not math.isfinite(delay_seconds) or delay_seconds < 0:
        raise ValueError("Evaluation requires cases, positive k and nonnegative finite delay")
    ids = [case.id for case in cases]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}", item) for item in ids):
        raise ValueError("Case IDs must be unique safe file names")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "artifact_schema_version": 2,
        "label": label, "created_at": datetime.now(timezone.utc).isoformat(), "state": "running",
        "case_ids": ids, "case_set_sha256": fingerprint([asdict(case) for case in cases]),
        "recall_k": k, "delay_seconds": delay_seconds,
        "implementation_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("application.py", "bootstrap.py", "domain.py", "ports.py", "evaluation.py", "experiments.py",
                         "extractive.py", "adapters/rule_planner.py", "adapters/gemini.py",
                         "adapters/gemini_extractive.py", "adapters/qdrant.py")
        },
    }
    for case in cases:
        folder = output / case.id
        folder.mkdir()
        _write(folder / "case.json", asdict(case))
    _write(output / "manifest.json", manifest)
    records, traces, preparations = [], {}, {}
    startup_error = None
    try:
        service, identity = factory()
        allowed = {"index_name", "snapshot_id", "catalog_sha256", "embedding", "generator_model",
                   "prompt_version", "planner_version", "retrieval_k", "context_movies",
                   "max_output_tokens", "query_mode", "generator_mode", "renderer_version"}
        manifest["runtime"] = {name: identity[name] for name in allowed if name in identity}
    except Exception as exc:
        startup_error = _error(exc)
        manifest["startup_error"] = startup_error
    stopped = None
    for number, case in enumerate(cases):
        record = {"id": case.id, "case_sha256": fingerprint(asdict(case))}
        folder = output / case.id
        if startup_error is not None or stopped is not None:
            record.update(state="not_executed", error={"type": "EvaluationStopped",
                          "message": "Setup failure" if startup_error else "Stopped after quota/access error"})
        else:
            if number and delay_seconds:
                wait(delay_seconds)
            try:
                # Only the question enters preparation. Reference labels never
                # reach the RAG, and persistence finishes before generation.
                prepared = service.prepare(case.question)
                if not isinstance(prepared, PreparedQuery) or prepared.question != case.question:
                    raise ValueError("Preparation must describe the evaluation question")
                prepared_data = asdict(prepared)
                _save_prepared(folder, prepared_data)
                record["prepared_sha256"] = fingerprint(prepared_data)
                preparations[case.id] = prepared
                trace = service.answer(prepared)
                if trace.question != case.question:
                    raise ValueError("Trace question differs from evaluation question")
                validate_answer(trace.answer, trace.context,
                                exact_count=trace.exact_result if trace.plan.intent is Intent.COUNT else None)
                trace_data = asdict(trace)
                if any(trace_data[name] != value for name, value in prepared_data.items()):
                    raise ValueError("Completed trace differs from archived preparation")
                trace_hash = fingerprint(trace_data)
            except Exception as exc:
                record.update(state="error", error=_error(exc))
                if isinstance(exc, GenerationUnavailable) and exc.http_status in (401, 403, 429):
                    stopped = record["error"]
            else:
                traces[case.id] = trace
                record.update(state="ok", trace_sha256=trace_hash)
                _write(folder / "trace.json", trace_data)
                _write(folder / "answer.json", trace_data["answer"])
        if record["state"] != "ok":
            _write(folder / "error.json", record["error"])
        records.append(record)
        _write(output / "results.json", records)
        if progress is not None:
            progress(case.id, record["state"])
    manifest["state"] = ("failed" if startup_error else "stopped" if stopped else
                         "completed_with_errors" if any(r["state"] == "error" for r in records) else "completed")
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    metrics = summarize(cases, records, traces, k=k, preparations=preparations)
    _write(output / "metrics.json", metrics)
    _write(output / "manifest.json", manifest)
    return metrics


def score_run(output: Path, reviews_path: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    records = json.loads((output / "results.json").read_text(encoding="utf-8"))
    schema = manifest.get("artifact_schema_version", 1)
    if type(schema) is not int or schema not in (1, 2):
        raise ValueError("Unsupported artifact schema version")
    if (manifest["state"] not in {"completed", "completed_with_errors", "stopped", "failed"}
            or [r["id"] for r in records] != manifest["case_ids"]
            or any(r["state"] not in {"ok", "error", "not_executed"} for r in records)):
        raise ValueError("Run is unfinished or its case ledger is inconsistent")
    if any(not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}", item) for item in manifest["case_ids"]):
        raise ValueError("Unsafe case ID in run manifest")
    judge = ReviewedSemanticJudge(json.loads(reviews_path.read_text(encoding="utf-8")))
    if not set(judge.decisions) <= set(manifest["case_ids"]):
        raise ValueError("Semantic reviews include cases outside this run")
    cases, traces, preparations = [], {}, {}
    for record in records:
        folder = output / record["id"]
        case = case_from_dict(json.loads((folder / "case.json").read_text(encoding="utf-8")))
        if case.id != record["id"] or fingerprint(asdict(case)) != record["case_sha256"]:
            raise ValueError("Archived case/rubric differs from the run ledger")
        cases.append(case)
        prepared_data = None
        if schema == 2:
            if "prepared_sha256" in record:
                if record["state"] == "not_executed":
                    raise ValueError("Unexecuted case cannot contain preparation")
                prepared_data = json.loads((folder / "prepared.json").read_text(encoding="utf-8"))
                if fingerprint(prepared_data) != record["prepared_sha256"]:
                    raise ValueError("Preparation checksum differs from the run ledger")
                _check_prepared_files(folder, prepared_data)
                prepared = prepared_from_dict(prepared_data)
                if fingerprint(asdict(prepared)) != record["prepared_sha256"]:
                    raise ValueError("Preparation is not in canonical form")
                if prepared.question != case.question:
                    raise ValueError("Preparation belongs to another question")
                preparations[case.id] = prepared
            elif record["state"] == "ok":
                raise ValueError("Completed case requires archived preparation")
        if record["state"] == "ok":
            data = json.loads((folder / "trace.json").read_text(encoding="utf-8"))
            if fingerprint(data) != record["trace_sha256"]:
                raise ValueError("Trace checksum differs from the run ledger")
            if prepared_data is not None and any(data[name] != value for name, value in prepared_data.items()):
                raise ValueError("Completed trace differs from archived preparation")
            for name in ("retrieved", "context", "answer"):
                if fingerprint(json.loads((folder / (name + ".json")).read_text(encoding="utf-8"))) != fingerprint(data[name]):
                    raise ValueError("Separate run artifacts differ from their trace")
            trace = trace_from_dict(data)
            if trace.question != case.question:
                raise ValueError("Trace belongs to another question")
            validate_answer(trace.answer, trace.context,
                            exact_count=trace.exact_result if trace.plan.intent is Intent.COUNT else None)
            traces[case.id] = trace
        elif case.id in judge.decisions:
            raise ValueError("Cannot review a case without a completed trace")
    if fingerprint([asdict(case) for case in cases]) != manifest["case_set_sha256"]:
        raise ValueError("Archived case set differs from manifest")
    metrics = summarize(cases, records, traces, k=manifest["recall_k"], judge=judge,
                        preparations=preparations if schema == 2 else None)
    for case in cases:
        if case.id in traces and case.id in judge.decisions:
            trace = traces[case.id]
            _write(output / case.id / "judgment.json", {
                "state": "reviewed",
                "reviewer": judge.document["reviewer"], "semantic": judge.decisions[case.id],
                "status_matches": trace.answer.status is case.expected_status,
                "strict_filters_compliant": strict_filters_compliant(case, trace),
                "accepted": accepted(case, trace, judge),
            })
        else:
            # Rescoring with fewer decisions must not leave an old accepted
            # judgment beside a new pending acceptance rate.
            _write(output / case.id / "judgment.json", {
                "state": "pending_review" if case.id in traces else "no_answer",
                "reviewer": None, "semantic": None,
                "accepted": None if case.id in traces else False,
            })
    _write(output / "reviews.json", judge.document)
    _write(output / "metrics.json", metrics)
    manifest["scoring"] = {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "reviewer": judge.document["reviewer"], "reviews_sha256": fingerprint(judge.document),
        "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                          for name in ("evaluation.py", "experiments.py")},
    }
    _write(output / "manifest.json", manifest)
    return metrics
