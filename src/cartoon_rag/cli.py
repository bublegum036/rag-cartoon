"""Local validation entry point; live provider wiring is intentionally separate."""

import argparse
import json
from pathlib import Path

from cartoon_rag.evaluation import load_cases
from cartoon_rag.ingestion import active_snapshot, load_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Cartoon RAG project tools")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-data", help="Validate catalog and evaluation cases")
    validate.add_argument("--catalog", type=Path, default=Path("catalog.json"))
    validate.add_argument("--questions", type=Path, default=Path("questions.jsonl"))
    args = parser.parse_args()
    films = active_snapshot(load_catalog(args.catalog))
    cases = load_cases(args.questions)
    known = {film.source_id for film in films}
    for case in cases:
        if not set(case.relevant_source_ids) <= known:
            parser.error(f"Unknown reference source in case {case.id}")
    print(json.dumps({"active_films": len(films), "evaluation_cases": len(cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
