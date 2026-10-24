"""Data validation, standalone planning, publication and explicit-filter search."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from cartoon_rag.application import RetrievalService
from cartoon_rag.evaluation import load_cases
from cartoon_rag.domain import EmbeddingUnavailable, Filters, QueryPlan, RetrievalUnavailable
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
        if args.command == "validate-data":
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
    except (EmbeddingUnavailable, RetrievalUnavailable, ValueError, OSError) as exc:
        print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}},
                         ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
