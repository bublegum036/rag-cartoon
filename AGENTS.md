# Project guidance

## Current state

This directory contains a Russian-language RAG assignment, not an implemented
application. Read `README.txt` for the full requirements. There is currently no
dependency manifest, application code, or test runner.

- `catalog.json`: 12 fictional film cards; the sole source of film facts.
- `questions.jsonl`: 10 evaluation cases with expected statuses, source IDs,
  strict filters, and semantic acceptance rubrics.

Do not claim installation, launch, or test commands exist before implementing
and verifying them. Preserve the supplied catalog and evaluation cases unless
the user requests changes.

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
