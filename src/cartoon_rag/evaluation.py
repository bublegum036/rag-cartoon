"""Evaluation keeps retrieval metrics separate from semantic answer acceptance."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from cartoon_rag.domain import (
    Answer, Chunk, Evidence, ExactResult, Film, Filters, Hit, Intent, QueryPlan,
    PreparedQuery, Recommendation, Status, Trace,
)
from cartoon_rag.snapshots import fingerprint


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    expected_status: Status
    relevant_source_ids: tuple[str, ...]
    filters: Filters
    rubric: str


def load_cases(path: Path) -> tuple[Case, ...]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(Case(
            id=row["id"], question=row["question"],
            expected_status=Status(row["expected_status"]),
            relevant_source_ids=tuple(row["relevant_source_ids"]),
            filters=Filters(**row["filters"]), rubric=row["rubric"],
        ))
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Duplicate evaluation case IDs")
    return tuple(cases)


def recall_at_k(case: Case, trace: Trace | PreparedQuery, k: int) -> float | None:
    if k < 1:
        raise ValueError("k must be positive")
    relevant = set(case.relevant_source_ids)
    if not relevant:
        return None  # Exclude these cases from the macro-average denominator.
    retrieved = {hit.chunk.film.source_id for hit in trace.retrieved[:k]}
    if trace.plan.intent is not Intent.SEARCH:
        retrieved = {item.film.source_id for item in trace.context[:k]}
    return len(relevant & retrieved) / len(relevant)


class SemanticJudge(Protocol):
    def accepts(self, case: Case, trace: Trace) -> bool:
        """Review rubric, factual support and alternative valid recommendations.

        Implement with human review or a separately configured judge; never use
        literal string equality as a substitute for semantic assessment.
        """
        ...


def retrieval_filters_compliant(case: Case, trace: Trace) -> bool:
    """Check every retrieved hit against reference filters, not the planner's.

    This remains independent of context selection and the final answer so a
    lost planner constraint cannot be masked by a correct recommendation.
    """
    return all(case.filters.matches(hit.chunk.film) for hit in trace.retrieved)


def prepared_filters_compliant(case: Case, prepared: PreparedQuery) -> bool:
    """Check evidence against reference constraints even without a final answer."""
    return (all(case.filters.matches(hit.chunk.film) for hit in prepared.retrieved)
            and all(case.filters.matches(item.film) for item in prepared.context))


def accepted(case: Case, trace: Trace, judge: SemanticJudge) -> bool:
    if trace.answer.status is not case.expected_status:
        return False
    if not strict_filters_compliant(case, trace):
        return False
    # Judge also checks constraints for prose answers without recommendations.
    return judge.accepts(case, trace)


def strict_filters_compliant(case: Case, trace: Trace) -> bool:
    if not retrieval_filters_compliant(case, trace):
        return False
    recommended = {item.movie_id for item in trace.answer.recommendations}
    cited = set(trace.answer.source_ids)
    for item in trace.context:
        own = {item.film.source_id, *(chunk.chunk_id for chunk in item.chunks)}
        if (item.film.movie_id in recommended or cited & own) and not case.filters.matches(item.film):
            return False
    return True


def case_from_dict(row: dict) -> Case:
    filters = dict(row["filters"])
    filters["movie_ids"] = tuple(filters.get("movie_ids", ()))
    return Case(row["id"], row["question"], Status(row["expected_status"]),
                tuple(row["relevant_source_ids"]), Filters(**filters), row["rubric"])


def prepared_from_dict(row: dict) -> PreparedQuery:
    def film(data):
        return Film(**{**data, "tags": tuple(data.get("tags", ()))})

    def chunk(data):
        return Chunk(data["chunk_id"], film(data["film"]), data["text"])

    raw_plan = row["plan"]
    filters = {**raw_plan["filters"], "movie_ids": tuple(raw_plan["filters"]["movie_ids"])}
    plan = QueryPlan(raw_plan["semantic_query"], Intent(raw_plan["intent"]), Filters(**filters),
                     raw_plan["time_budget_minutes"],
                     Status(raw_plan["early_status"]) if raw_plan["early_status"] is not None else None,
                     raw_plan["early_text"])
    raw_exact = row["exact_result"]
    exact = ExactResult(raw_exact["value"], raw_exact["unit"], tuple(raw_exact["source_ids"])) if raw_exact else None
    return PreparedQuery(row["question"], plan,
                         tuple(Hit(chunk(item["chunk"]), item["score"]) for item in row["retrieved"]),
                         tuple(Evidence(film(item["film"]), tuple(chunk(c) for c in item["chunks"]))
                               for item in row["context"]), exact)


def trace_from_dict(row: dict) -> Trace:
    prepared = prepared_from_dict(row)
    raw_answer = row["answer"]
    answer = Answer(Status(raw_answer["status"]), raw_answer["text"], tuple(raw_answer["source_ids"]), tuple(
        Recommendation(item["movie_id"], item["title"], item["reason"], tuple(item["source_ids"]))
        for item in raw_answer["recommendations"]
    ))
    return Trace(prepared.question, prepared.plan, prepared.retrieved, prepared.context,
                 prepared.exact_result, answer)


class ReviewedSemanticJudge:
    """Explicit semantic decisions, bound to the exact case and generated trace.

    A reviewer reads the rubric and source facts. This class checks the decision
    format and provenance; it does not replace semantic review with text matching.
    """

    checks = ("rubric_passed", "facts_supported", "constraints_passed", "references_supported")

    def __init__(self, document: dict) -> None:
        if not isinstance(document.get("reviewer"), str) or not document["reviewer"].strip():
            raise ValueError("Semantic reviews require a named reviewer")
        if not isinstance(document.get("decisions"), list):
            raise ValueError("Semantic reviews require a decisions array")
        self.document = document
        self.decisions = {}
        required = {"case_id", "case_sha256", "trace_sha256", "explanation", *self.checks}
        for decision in document["decisions"]:
            if not isinstance(decision, dict) or set(decision) != required:
                raise ValueError("Invalid semantic review fields")
            if any(type(decision[name]) is not bool for name in self.checks):
                raise ValueError("Semantic review checks must be booleans")
            if not isinstance(decision["explanation"], str) or not decision["explanation"].strip():
                raise ValueError("Semantic review requires an explanation")
            if (not isinstance(decision["case_id"], str)
                    or any(not isinstance(decision[name], str) or not re.fullmatch(r"[0-9a-f]{64}", decision[name])
                           for name in ("case_sha256", "trace_sha256"))):
                raise ValueError("Semantic review requires a case ID and SHA-256 bindings")
            if decision["case_id"] in self.decisions:
                raise ValueError("Duplicate semantic review")
            self.decisions[decision["case_id"]] = decision

    def accepts(self, case: Case, trace: Trace) -> bool:
        decision = self.decisions[case.id]
        if decision["case_sha256"] != fingerprint(asdict(case)):
            raise ValueError("Semantic review belongs to another case/rubric")
        if decision["trace_sha256"] != fingerprint(asdict(trace)):
            raise ValueError("Semantic review belongs to another generated trace")
        return all(decision[name] for name in self.checks)


def save_trace(path: Path, trace: Trace) -> None:
    """Store complete hits, selected context, exact computation and final answer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(trace), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
