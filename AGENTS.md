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
Experimental generator mode extractive selects existing fragment IDs only;
extractive.py validates ownership and renders quotations/metadata, exact operations
without LLM. CLI ask/evaluate support --generator-mode extractive, prompt
extractive-ru-v1. Grounded/v2 remains default. Provenance includes generator_mode
and renderer_version; custom injected generators remain explicitly identified.
Initial extractive checks: 144 tests passed, 3 skipped. Real four-case experiment: 2 accepted,
2 HTTP 503, 3 Gemini calls plus one local count. No overall quality improvement
claimed; source fidelity does not prove relevance/completeness. See docs/extractive.md
and artifacts/expanded-extractive-2026-10-26.
Extractive LOOKUP fix: renderer v2 appends the full verbatim synopsis to card
metadata because named-film LOOKUP includes plot questions, not only facts.
145 tests passed, 3 skipped; new end-to-end offline test covers three named
plot queries. No providers rerun; old renderer source retained to match historical
v1 run hashes (artifacts/extractive-lookup-fix-2026-10-26).
Latest implementation: evaluation artifact schema 2 saves PreparedQuery before
generation; QuestionAnswerer factories now provide prepare(question)/answer(prepared).
Verified preparation survives generation failure and supports Recall/filter checks
without a fake answer; answer acceptance still includes failures. Legacy schema 1
metrics stay unchanged. 132 tests passed, 3 skipped; providers not rerun for this
change (artifacts/pre-generation-evidence-2026-10-26).
Separate manual rerun of the four expanded technical failures used schema 2:
4 prepared queries and answers, no technical errors, Codex accepted 3/4.
mechanical-whale omits the required use of measurements for safe boat passage.
Original 16/20 run unchanged; do not merge attempts into a one-pass score.
See artifacts/expanded-retry-v2-2026-10-26 and data/questions-expanded-retry.jsonl.
Experimental grounded-ru-v3 adds a generic full causal-chain instruction; CLI
evaluate run can select it. On the same four cases, mechanical-whale was fixed
but sea-short omitted 78 minutes (and added an unsupported gratitude motive).
Acceptance stayed 3/4, no technical failures. v2 remains default; do not claim
overall improvement or a full 20-case v3 evaluation. See artifacts/expanded-retry-v3-2026-10-26.
Experimental v4 adds numeric-fact preservation and forbids inferred motives.
Four-case gate returned 2 accepted answers and 2 HTTP 503 (sea-short/count-all):
50% overall acceptance, all 4 preparations retained, Recall/filter checks 100%.
At that stage numeric-answer preservation was unverified; conditional full 20-case
v4 run was not started. See artifacts/expanded-retry-v4-2026-10-26.
Separate v4 recovery of sea-short/count-all returned 2 answers without technical
errors: 78 and 40 are explicit, but sea-short adds the unsupported gratitude motive
already rejected in v3. Accepted 1/2; full-20 gate still not passed. v2 stays default.
See artifacts/expanded-v4-recovery-2026-10-26 and data/questions-expanded-v4-recovery.jsonl.
Latest semantic recheck (2026-10-26): all 20 archived answers re-reviewed by
the same Codex, confirming A 9/10 and B 10/10; not an independent assessment.
Original A/B files unchanged; 124 tests passed, 3 integration tests skipped;
no provider calls (artifacts/semantic-recheck-2026-10-26).
User supplied a separate Kimi K2.8 textual review, archived 2026-10-26:
A 9/10, B 10/10 confirmed. Not blind or human; execution date and exact
model ID unknown, no Kimi call observed by Codex. See artifacts/kimi-review-2026-10-26.
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
- `data/catalog-expanded.json`: optional 40-film catalog, preserving the original
  12 plus 28 fictional films dated 1970–2000. Offline validation/exact operations
  passed; indexed separately as cartoons-expanded on 2026-10-26 with real E5/Qdrant.
  data/questions-expanded.jsonl contains 20 cases; retrieval/exact checks passed,
  semantic Recall@1/@3 = 100% on 11 reference cases (artifacts/expanded-retrieval-2026-10-26).
  First expanded Gemini run saved 16 answers, all accepted by Codex; 4 technical
  failures (3 HTTP 503, 1 request/JSON failure) remain in the denominator: 80%
  acceptance. See artifacts/expanded-run-v2-2026-10-26. Missing traces after
  generation errors lower run Recall/filter metrics; do not claim observed
  retrieval regressions from those zeros. No automatic retries or fallback.
  Baseline no-match rubric is
  invalid here: film-13 is a marine story lasting 24 minutes. Use separate cases
  and a separate Qdrant alias for expansion experiments; see data/README.md.

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
