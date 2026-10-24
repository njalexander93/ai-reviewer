# Chunking validation — 2026-10-23

These measurements cover preparation and deterministic planning, not paid model
judgment. No benchmark review was posted. No repository code was executed.

## PR #3 feedback repair

The repair removes the obsolete branch-test workflow and uses one manifest-backed
review/consolidation/publication flow for both packet sizes. Small initial payloads
remain byte-identical. A provider-counted capacity rejection before generation
selects a bounded chunk plan. Publication consumes the canonical loader's covered
reports once, with snapshot binding and the existing live GitHub safeguards.

Relations now use a multi-pattern literal index over each captured version and a
normalized source/test-name index. The plan reuses that evidence for packing and
integration. Integration expands indexed file relationships and adjacent units,
not every unrelated unit pair; expansion fails above 20,000 distinct cross-owner
pairs. Shared overview and discussion metadata are compiled once per plan.
Manifest updates contain plan/payload hashes, ownership and completion records,
not full source packets. Attempt payloads remain the replay evidence.

Validation on the repair: 148 tests passed; 92.87% branch-aware coverage. Ruff,
Mypy and whitespace checks passed. New cases cover report insertion-order changes,
compact manifest tampering, both providers' token-overflow fallback before
generation, count-only calls, overlapping references and renamed paths. Existing
CLI rejection tests now create real manifest artifacts before corrupting reports,
so they exercise completeness/anchor rejection rather than missing-file failure.

The static test audit examined 59 test functions in five changed files, with no
cannot-fail findings or parser gaps. Fresh-context expectation review covered 12
changed functions in the planner, orchestration and provider tests: 3 PASS,
9 UNKNOWN, no FLAG. UNKNOWN includes synthetic provider-response provenance and
exact fixture-size/range expectations. The repeated-reference fixture was then
tightened to distinguish the first occurrence from later occurrences. Model-family
independence is unverified; these reviews do not establish live-provider behavior.

Offline reruns used the same historical snapshots below. Their complete indexed
relationship/evidence results matched the previous implementation exactly.
PR #9 retained 11 chunks and 24 integration assignments (largest 592,218 bytes);
PR #18 retained 8 chunks and 15 integration assignments (largest 575,039 bytes).
Planning/integration took 1.113/0.617 seconds and 0.565/0.149 seconds respectively.
A separate 3,000-file fixture with 24,000,000 captured source bytes and no matching
references completed relationship discovery in 0.759 seconds. These are single
development-Mac measurements, not comparative speedup or worst-case guarantees.
Dense matching remains bounded by relation/work limits.

The 180-minute Actions job timeout is the effective wall-clock ceiling. Work and
per-call maxima do not promise completion within it. Cost/evidence steps use
`always()`, but their completion after cancellation remains unverified.

## Historical snapshots

Read PR metadata, original file patches, previous inline comments and linked
GitHub issues using GitHub's API. Read before/after blobs from a temporary bare
clone at the immutable revisions below. Blob reading used `git show` and
`git ls-tree`; the clone was never checked out. This measures an offline
collection equivalent, not the production REST blob-fetch latency.

| Snapshot | PR #9 | PR #18 |
| --- | --- | --- |
| Source | [Repository rename](https://github.com/njalexander93/nplet/pull/9) | [Scoped Cursor children](https://github.com/njalexander93/nplet/pull/18) |
| Head | `3bc4b03f1c7c2f324edf25486b15c0cc2237789c` | `67ea982ee14394da0af6124abdab7b9d9bfd08f2` |
| Base and merge base | `94c7eb880fdb7068af7d8b7a109a9472795edb16` | `1451b0bf9c4ccd66ce734faa37a25d055469068d` |
| Files | 125 | 67 |
| Additions/deletions | 387 / 377 | 14,155 / 426 |
| Captured inline comments | 0 | 97 |
| Linked GitHub issues | 0 | 0 |
| Serialized snapshot bytes | 4,504,447 | 3,103,878 |
| Collection time, seconds | 8.979 | 5.585 |
| Planning time, seconds | 1.462 | 0.623 |
| Units / local chunks | 125 / 11 | 69 / 8 |
| Largest local packet, bytes | 419,888 | 419,596 |
| Integration assignments with empty synthetic reports | 24 | 15 |
| Largest integration packet, bytes | 592,218 | 575,039 |

Single measurements on the development Mac, Python 3.14.8. No warm-up or repeated
performance study. They establish that these snapshots fit the planner, not a
latency guarantee. All 192 captured file patches passed complete before/after
validation. Linear was not configured for these measurements. Current discussion
was captured once; this does not reconstruct discussion as it existed at merge.

Both PRs are closed and authored by `njalexander93-ai`. Production preparation
still rejects closed/self-authored PRs. Offline measurement bypasses no live
publication gate and requests no reviewers.

Integration metrics use synthetic empty completed reports solely to measure
planning. Real reports, token counts and source requests can enlarge the work or
make it exceed a bound. The resulting baseline is 70 reviewer calls for #9 and
46 for #18 before follow-ups or Sonnet. No tokens or paid-review costs were
measured. The limits are safeguards, not a claim that maximum runs are cheap.

## Regression evidence

Tests exercise production planning, artifact replay, CLI handoff and publication
against synthetic provider/application reports and a GitHub HTTP fixture:

- Exact patch-record coverage, stable packing, bounded UTF-8 serialization,
  oversized hunks, metadata overflow and unsplittable lines.
- Renames, mode-only changes, Unicode separators, CRLF and final-newline changes.
- Full-blob references outside changed hunks, for both same-chunk and cross-chunk integration.
- Accumulated report paging with actual source repeated; consolidation retry protection.
- Missing, duplicate, failed or altered work; unresolved cross-file requests;
  all original findings retained across many reports and consolidation batches.
- Actual-source integration, stale-head publication, nested attempt costs and
  provider token preflight rejection.
- Existing whole-packet tests retain the two independent calls and one review.
- A known signature/caller defect is available to a deterministic recognizer in
  both a whole packet and forced cross-chunk integration. This tests evidence
  availability, not a model's ability to diagnose the defect.

The paid SDK boundaries are mocked. Synthetic model reports are valid application
contracts, not recorded proof that live providers will produce those reports.
Provider count methods and argument shapes were checked against the installed
SDKs and the official documentation linked in the README.

## Remaining limits

No live paid comparison establishes equal recall, severity judgment or false-
positive rates. Python is the only syntax-aware splitter. Direct references and
adjacent directory sections are heuristic relationship evidence; no repository-
wide dependency crawler is built. Required unchanged dependencies remain outside
the snapshot and must produce an incomplete review.

Summary reduction asks Sonnet to preserve disagreements, but cannot mechanically
prove semantic fidelity. Original reports and findings remain available in the
artifacts; finding partition and severity preservation are enforced in code.
Cross-batch duplicates can appear as separate comments. GitHub's final revision
check and review write remain non-atomic; stale approval dismissal is required.

## Review checks

Required local checks passed: lock consistency, Ruff formatting/lint, Mypy, and
141 tests with 92.07% branch-aware coverage. The bounded static test audit examined
54 functions across five changed test files; no cannot-fail pattern was reported.

A separate read-only safety review found report-overflow, omitted distant-reference
and consolidation-retry accounting defects. All were repaired and regression-tested,
including same-chunk references outside changed hunks.

Fresh-context Astra test judges checked 17 functions in the two new test files.
The evidence validator accepted 10 PASS and 7 UNKNOWN verdicts; no FLAG verdicts.
Model-family independence from the writer was not established. The nine planner
PASS verdicts cover whole-packet preservation, patch coverage, metadata overflow,
Unicode/coordinates, deterministic packing, mode-only source, indivisible lines,
valid Git patches and partial-patch rejection. Integration report paging also passed.

UNKNOWN remains for the six orchestration tests that use synthetic provider/GitHub
responses: full pipeline, corrupt manifest, cross-file requests, worker failure,
many-report source IDs and the known-defect comparison. The distant-reference test
was also UNKNOWN for its exact one-chunk expectation, although its expected import
text comes directly from the fixture. These are limits of oracle provenance, not
observed runtime failures or evidence of live model quality. Other changed test
files received static audit and execution, not a separate expected-value judge.
