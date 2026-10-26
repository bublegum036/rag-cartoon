# Project guidance

## Current state

This directory contains a Russian-language RAG assignment and a Python 3.11+
architecture with Qdrant REST and local FastEmbed E5 search adapters. Read
`README.txt` for the assignment, `README.md` for commands and verification limits,
and `docs/architecture.md` for boundaries and next steps. Provider integration
tests are opt-in; Qdrant 1.17.0 and real E5 checks passed on 2026-10-25
with Python 3.13.15 on macOS arm64 (see artifacts/verification-2026-10-25).
The rule-based Russian planner and offline `plan` CLI are implemented.
Planner ru-rules-v1.1 includes duration-fact and age-topic review fixes;
64 local tests passed, 3 integration tests skipped after these fixes (artifacts/planner-fixes-2026-10-25).
JsonExactCatalog and offline `exact` CLI implement lookup, full filtered counts
and signed duration comparisons. `ask` composes these with Gemini structured generation
and a Qdrant collection pinned to the same catalog fingerprint.
Latest semantic recheck (2026-10-26): all 20 archived answers re-reviewed by
the same Codex, confirming A 9/10 and B 10/10; not an independent assessment.
Original A/B files unchanged; 124 tests passed, 3 integration tests skipped;
no provider calls (artifacts/semantic-recheck-2026-10-26).
Previous generator identity fix: 124 tests passed, 3 integration tests skipped;
providers were not rerun (artifacts/generator-identity-fix-2026-10-25).
Injected generators require explicit GeneratorIdentity; never infer their metadata from Settings.
Previously after evaluate/A-B: all 123 tests passed with real Qdrant/E5;
local checks passed 120 and skipped 3. All 10 supplied cases ran for A and B
with 14 Gemini generation requests total. Codex semantic review accepted 9/10
in A and 10/10 in B; Recall@3 and strict-filter compliance were 100% in both.
Only the generator prompt changed; v2 is now default, v1 remains replayable.
See docs/ab-report.md and docs/evaluation.md. Reviews are explicit decisions
bound to case/trace hashes, not automatic text-matching judgments.
Previously after the Gemini HTTP truncation fix: 105 passed, 3 integration tests skipped
(artifacts/gemini-http-fix-2026-10-25). Real providers were not rerun after this fix.
Previously, 106 tests passed with real Qdrant/E5, plus two live Gemini smoke answers
(artifacts/rag-verification-2026-10-25).
Gemini uses a Free Tier project key in `.env`; never enable billing, switch models
as a fallback, retry quota errors automatically, or log keys. Unit tests mock Gemini;
real Gemini calls must be explicitly authorized (as in the current implementation session).
For CPython 3.13 on macOS arm64, use requirements/search-macos-arm64-py313.lock
and the installation steps in requirements/README.md. A clean install and all
41 tests passed (artifacts/lock-verification-2026-10-25); other lock targets are unverified.

Learning notes live in `docs/rag-learning-notes.md`. Update them after meaningful
implementation stages, review fixes or experiments: explain the RAG principle,
how it applies here, what was verified and what remains unverified. Keep the
current status accurate and never present planned or skipped checks as passed.

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
explicit CLI filters and does not call the planner. `plan` supports a limited
Russian grammar; do not present it as a universal natural-language parser.
The planner receives only active movie IDs/titles, never evaluation labels or film facts.
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
