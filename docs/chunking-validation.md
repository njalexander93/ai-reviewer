# Chunking validation — 2026-10-23

Current local checks: **168 tests pass; 94.02% branch-aware coverage**. Ruff,
Mypy, lockfile and whitespace checks pass. Tests use synthetic provider responses.
No live-provider quality comparison was run.

This follow-up unifies execution/replay transitions, preflights both providers on
all local scopes, and tests capacity fallback above and below whole-packet size.
Integration reports retain one shape and explicit assignment/provider identity
through paging. Publication labels summaries from loader metadata. Packet ceilings
are required parameters from the plan, no longer copied into each assignment.

CoveredReviews is trusted internal data, not an unforgeable proof. The CLI obtains
it only from the validating loader. Existing run artifacts must be regenerated.
Required evidence that cannot fit still fails explicitly; bytes do not guarantee
token capacity. Relationships are heuristic and unchanged dependencies remain
outside the snapshot. No tests establish model recall or summary fidelity.

Historical benchmarks, immutable snapshot IDs and earlier audit results remain in
the [59a2475 validation record](https://github.com/njalexander93/ai-reviewer/blob/59a24758a183986c414a65176ddf91b59372dcc9/docs/chunking-validation.md).
Its timings are single-run offline smoke observations, not current benchmarks or
worst-case guarantees. Current SDK boundaries remain mocked. Cross-batch duplicate
findings may remain; GitHub's final revision check and write remain non-atomic.
