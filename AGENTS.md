# Project guidance

## Current state

This directory contains a Russian-language RAG assignment and a Python 3.11+
architecture with Qdrant REST and local FastEmbed E5 search adapters. Read
`README.txt` for the assignment, `README.md` for commands and verification limits,
and `docs/architecture.md` for boundaries and next steps. Provider integration
tests are opt-in and have not yet run successfully in this environment.
An end-to-end RAG system (planner, exact operations and generator) is not yet implemented.

- `catalog.json`: 12 fictional film cards; the sole source of film facts.
- `questions.jsonl`: 10 evaluation cases with expected statuses, source IDs,
  strict filters, and semantic acceptance rubrics.

Do not claim installation, launch, or test commands exist before implementing
and verifying them. Preserve the supplied catalog and evaluation cases unless
the user requests changes.

Run scaffold checks from the project root:
`PYTHONPATH=src python3 -m unittest discover -s tests -v` and
`PYTHONPATH=src python3 -m cartoon_rag.cli validate-data`.
Keep SDK imports in adapters and compose services through the protocols in
`ports.py`. Do not present scaffold tests as completed A/B experiments.
Use `.venv` for Python dependencies and `.env` for local settings. Do not commit
either directory/file, model caches or API keys. `search` currently accepts
explicit CLI filters; do not claim natural-language constraint parsing exists.
Live checks require `QDRANT_INTEGRATION_URL`; semantic checks additionally require
`RUN_SEMANTIC_SMOKE=1`. They use random test aliases and must preserve the main index.

## Agent profiles

Reusable role instructions live in `agents/`; see `agents/README.md`.
For code review requests, read and follow `agents/code-reviewer.md` before
reviewing changes. Every review includes both Python architect and Senior Python
developer perspectives, with confirmed defects separated from design suggestions.
Use the profile in the current session; it does not require
spawning a separate agent. Apply it only to review work, not implementation.

## Commit convention

Use Conventional Commits 1.0.0 for all new commits:
https://www.conventionalcommits.org/ru/v1.0.0/

- Format: `<type>[optional scope][!]: <description>`.
- Use `feat` for new functionality and `fix` for bug fixes. Other appropriate
  types include `docs`, `refactor`, `test`, `build`, `ci`, and `chore`.
- Use lowercase types consistently. Scope is optional and names the affected
  component, for example `feat(search): add strict duration filters`.
- Mark breaking changes with `!` before the colon or a `BREAKING CHANGE:` footer.
- Separate an optional body and footers from the preceding section by a blank line.
- Keep commits focused. Do not rewrite existing commit history unless requested.

## Implementation requirements

- Embed the title together with the synopsis. Preserve metadata including
  `movie_id`, `source_id`, `year`, `duration_minutes`, `version`, and `active`.
- Keep short descriptions intact. Split long descriptions at meaningful
  boundaries, append part identifiers, and retain the parent card source ID
  and movie ID for evaluation and deduplication.
- Use pgvector or Qdrant as described in the assignment; select the stack when
  implementation is requested.
- Apply exact constraints during retrieval, before semantic ranking.
  “Shorter than 90” means `duration_minutes < 90`.
- Group retrieved chunks by `movie_id` before recommending films.
- Ground generated answers in the supplied cards and include source IDs.
  Never invent ratings, age restrictions, or voice actors.
- Return `status`, `text`, and `source_ids`. For lists, include
  `recommendations` entries with `movie_id`, `title`, `reason`, and `source_ids`.
- Use `answered` for supported answers, `insufficient` when facts are missing,
  and `clarify` when the question needs clarification. Handle database errors
  separately: a timeout is not evidence that no matching film exists.
- Calculate exact counts and duration comparisons in code or SQL.

## Evaluation and delivery

Run all 10 supplied cases for versions A and B. Save retrieved IDs, context,
and final answers separately. Identify an observed error in A, change one
element, and repeat the same cases for B.

Report Recall@k for cases with reference sources, answer acceptance rate, and
strict-filter compliance. Acceptance must check the rubric, status, constraints,
and supporting references; string matching alone is insufficient. Map chunk
IDs back to their original card IDs when evaluating retrieval.

Deliver runnable code, documented launch commands, dependencies, the catalog,
the 10 cases, A/B run records, and a brief explanation of the error and fix.
Catalog expansion, hybrid retrieval, and the version-2 duration update are
optional extensions, not baseline requirements.
