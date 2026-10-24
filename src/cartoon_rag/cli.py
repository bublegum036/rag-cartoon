"""Offline planning/exact operations, data validation and explicit-filter search."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from cartoon_rag.application import RetrievalService, early_trace, execute_exact
from cartoon_rag.evaluation import load_cases, save_trace
from cartoon_rag.domain import EmbeddingUnavailable, Filters, GenerationUnavailable, QueryPlan, RetrievalUnavailable
from cartoon_rag.ingestion import SentenceChunker, active_snapshot, index_catalog, load_catalog
from cartoon_rag.settings import Settings


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Cartoon RAG project tools")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-data", help="Validate catalog and evaluation cases")
    validate.add_argument("--catalog", type=Path, default=Path("catalog.json"))
    validate.add_argument("--questions", type=Path, default=Path("questions.jsonl"))
    plan = commands.add_parser("plan", help="Parse a question without retrieval or answer generation")
    plan.add_argument("question")
    plan.add_argument("--catalog", type=Path, default=Path("catalog.json"))
    exact = commands.add_parser("exact", help="Execute exact operations; does not generate an answer")
    exact.add_argument("question")
    exact.add_argument("--catalog", type=Path, default=Path("catalog.json"))
    ask = commands.add_parser("ask", help="Answer with Gemini using checked catalog evidence")
    ask.add_argument("question")
    ask.add_argument("--catalog", type=Path, default=Path("catalog.json"))
    ask.add_argument("--trace", type=Path, help="Save retrieval, context, exact result and answer separately")
    evaluate = commands.add_parser("evaluate", help="Run cases or score an archived run with semantic reviews")
    evaluation_commands = evaluate.add_subparsers(dest="evaluation_command", required=True)
    run = evaluation_commands.add_parser("run", help="Generate artifacts; acceptance stays pending until review")
    run.add_argument("--questions", type=Path, default=Path("questions.jsonl"))
    run.add_argument("--catalog", type=Path, default=Path("catalog.json"))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--k", type=positive_int, default=3)
    run.add_argument("--retrieval-k", type=positive_int, default=10)
    run.add_argument("--context-movies", type=positive_int, default=3)
    run.add_argument("--delay-seconds", type=float, default=10)
    run.add_argument("--prompt-version", choices=("grounded-ru-v1", "grounded-ru-v2"), default=None)
    score = evaluation_commands.add_parser("score", help="Check review provenance and calculate final metrics")
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--reviews", type=Path, required=True)
    index = commands.add_parser("index", help="Publish a complete active catalog snapshot")
    index.add_argument("--catalog", type=Path, default=Path("catalog.json"))
    search = commands.add_parser("search", help="Retrieve chunks; does not generate an answer")
    search.add_argument("query")
    search.add_argument("--limit", type=positive_int, default=5, help="Maximum number of chunks")
    search.add_argument("--duration-lt", type=positive_int)
    search.add_argument("--duration-lte", type=positive_int)
    search.add_argument("--movie-id", action="append", default=[])
    args = parser.parse_args()
    try:
        if args.command == "evaluate":
            from cartoon_rag.experiments import run_evaluation, score_run

            if args.evaluation_command == "score":
                output = score_run(args.run_dir, args.reviews)
            else:
                from cartoon_rag.bootstrap import build_rag_runtime

                films = load_catalog(args.catalog)
                cases = load_cases(args.questions)
                settings = Settings.load(args.env_file)
                output = run_evaluation(
                    cases, lambda: build_rag_runtime(films, settings, retrieval_k=args.retrieval_k,
                                                     context_movies=args.context_movies,
                                                     prompt_version=args.prompt_version),
                    args.output, label=args.label, k=args.k, delay_seconds=args.delay_seconds,
                    progress=lambda case_id, state: print(f"{case_id}: {state}", file=sys.stderr),
                )
        elif args.command == "validate-data":
            films = active_snapshot(load_catalog(args.catalog))
            cases = load_cases(args.questions)
            known = {film.source_id for film in films}
            for case in cases:
                if not set(case.relevant_source_ids) <= known:
                    raise ValueError(f"Unknown reference source in case {case.id}")
            output = {"active_films": len(films), "evaluation_cases": len(cases)}
        elif args.command == "plan":
            from cartoon_rag.bootstrap import build_planner

            output = asdict(build_planner(load_catalog(args.catalog)).plan(args.question))
        elif args.command == "exact":
            from cartoon_rag.bootstrap import build_exact_catalog, build_planner

            # Load once so planner references and exact evidence share revisions.
            films = load_catalog(args.catalog)
            plan = build_planner(films).plan(args.question)
            output = asdict(execute_exact(build_exact_catalog(films), plan))
        elif args.command == "ask":
            from cartoon_rag.bootstrap import build_planner, build_rag

            films = load_catalog(args.catalog)
            plan = build_planner(films).plan(args.question)
            trace = early_trace(args.question, plan)
            if trace is None:
                trace = build_rag(films, Settings.load(args.env_file)).execute_plan(args.question, plan)
            if args.trace is not None:
                save_trace(args.trace, trace)
            output = asdict(trace.answer)
        else:
            from cartoon_rag.bootstrap import build_search

            settings = Settings.load(args.env_file)
            # Validate the catalog before downloading weights or contacting the DB.
            films = load_catalog(args.catalog) if args.command == "index" else ()
            active_snapshot(films)
            embedder, store = build_search(settings)
            if args.command == "index":
                publication = index_catalog(films, SentenceChunker(settings.chunk_chars), embedder, store)
                output = asdict(publication)
            else:
                plan = QueryPlan(args.query, filters=Filters(
                    duration_minutes_lt=args.duration_lt, duration_minutes_lte=args.duration_lte,
                    movie_ids=tuple(args.movie_id),
                ))
                output = asdict(RetrievalService(store).search(plan, limit=args.limit))
        print(json.dumps(output, ensure_ascii=False, indent=2))
        if (args.command == "evaluate" and args.evaluation_command == "run"
                and (output["technical_errors"] or output["not_executed"])):
            raise SystemExit(1)
    except (EmbeddingUnavailable, RetrievalUnavailable, GenerationUnavailable, ValueError, OSError) as exc:
        print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}},
                         ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
