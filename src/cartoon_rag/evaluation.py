"""Evaluation keeps retrieval metrics separate from semantic answer acceptance."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from cartoon_rag.domain import Filters, Intent, Status, Trace


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


def recall_at_k(case: Case, trace: Trace, k: int) -> float | None:
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


def accepted(case: Case, trace: Trace, judge: SemanticJudge) -> bool:
    if trace.answer.status is not case.expected_status:
        return False
    if not retrieval_filters_compliant(case, trace):
        return False
    recommended = {item.movie_id for item in trace.answer.recommendations}
    for evidence in trace.context:
        if evidence.film.movie_id in recommended and not case.filters.matches(evidence.film):
            return False
    # Judge also checks constraints for prose answers without recommendations.
    return judge.accepts(case, trace)


def save_trace(path: Path, trace: Trace) -> None:
    """Store complete hits, selected context, exact computation and final answer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(trace), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
