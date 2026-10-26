# AI Reviewer

Review with the providers whose API keys are configured. All models use **medium**
reasoning effort. One GitHub account publishes the validated result.

| Configured keys | Independent reviewers | Consolidation |
| --- | --- | --- |
| OpenAI only | GPT-6.1 Sol | GPT-6.1 Sol |
| Anthropic only | Claude Opus 5.5 | Claude Sonnet 5.5 |
| Both | Opus 5.5 and GPT-6.1 Sol | Sonnet 5.5 |
| Neither | Configuration error before provider calls | None |

Missing, empty and whitespace-only keys are unconfigured. A nonblank invalid key
remains configured: authentication, credit and other provider failures fail the run.
They never downgrade required coverage. There is no provider-mode setting.

The manifest records the enabled set and binds it into the plan digest. Replay and
publication use that set, never current credentials or the artifacts that happen to
exist. Every enabled reviewer must finish every required assignment and follow-up.
Older manifests without this selection are rejected; run artifacts are not migrated.

Model IDs and settings verified on **2026-10-25**:
[`gpt-6.1-sol`](https://developers.openai.com/api/docs/models/gpt-6.1-sol) serves
both OpenAI review and consolidation. Both roles use
`reasoning={"effort": "medium"}` through Responses.
[`claude-opus-5-5`](https://platform.claude.com/docs/en/models/overview) and
[`claude-sonnet-5-5`](https://platform.claude.com/docs/en/models/sonnet-5-5/overview)
use `thinking={"type": "adaptive"}` and `output_config={"effort": "medium"}`.
`models.py` owns the role-to-provider/model/pricing table. The canonical coverage
loader carries the validated provider set into `consolidation.py`; consolidation
does not reread the manifest.

Both consolidation stages use the selected consolidation model and the same validated
finding-partition contract. Every source finding must survive exactly once, including
unique findings as singleton groups. Model choice never decides approval.

## Install in a repository

1. Give your reviewer account collaborator access to the target repository.
   Create its personal access token with review permissions. The original setup
   uses `njalexander93-ai` and a classic token with `public_repo` scope.
   The username is resolved from the token; it is not hard-coded.
2. Add these **repository Actions secrets to each target repository**:

   | Secret | Value |
   | --- | --- |
   | `ANTHROPIC_API_KEY` | Optional; access to Opus 5.5 and Sonnet 5.5 |
   | `OPENAI_API_KEY` | Optional; access to GPT-6.1 Sol for review and consolidation |
   | `AI_REVIEWER_TOKEN` | Reviewer's GitHub personal access token |
   | `LINEAR_API_KEY` (optional) | Linear personal API key with read access to relevant issues |

   Configure at least one model-provider key. Keep the forwarding entries for both;
   an unset optional secret passes an empty value.

3. Copy [the caller workflow](examples/ai-review.yml) into the target repository
   as `.github/workflows/ai-review.yml`. Replace both `REVIEWER_RELEASE_COMMIT_SHA` placeholders with the same reviewed,
   full commit SHA from a release of this repository. The template is not deployable
   until those placeholders are replaced.
   Merge the caller into the target's default branch.
4. Enable **dismiss stale pull request approvals when new commits are pushed**
   in the target's branch protection or ruleset. Keep existing tests and required
   human reviews. This automation does not configure merge rules.
5. Open or update a non-draft PR. Check the workflow's **Report API costs** step for usage.
   An existing PR needs an update or manual workflow dispatch after installation.

The shared workflow runs **inside the calling repository's Actions run**. It uses
that repository's secrets, permissions, logs and artifacts. It does not discover
or enable itself on every repository in an account.

For a fork of this reviewer, update the caller's `uses:` repository and supply
`implementation-repository: your-account/your-reviewer` under `with:`. Replace both
commit pins and supply your own secrets. Never put credentials in workflow YAML.

See GitHub's documentation for [reusable workflows and explicit secrets](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows),
[token scopes](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/scopes-for-oauth-apps),
and [stale approval dismissal](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).
Private targets need a credential with access to them; a `public_repo` token is
insufficient.

## Review policy

- Runs on PR opening, reopening, publication from draft, and new commits.
- Same-repository PRs run automatically. Fork PRs require a maintainer to select
  **Run workflow** and enter the PR number. Draft, closed and reviewer-authored
  PRs are rejected. Requests serialize per repository and PR.
- Small packets retain one whole-packet review per enabled provider with complete before/after
  text. Both sizes use the same manifest, source validation and consolidation flow.
  Larger snapshots use deterministic chunks and one review per enabled provider of every required
  local/integration assignment. Each packet includes PR metadata, the full issue
  requirements, a changed-file inventory and relevant previous inline discussion.
  Before versions come from the merge base, matching the PR diff.
- Context uses validated file/comment records and omits repeated GitHub diff hunks
  and URL/user metadata. Symlinks retain their actual link text and Git mode.
- The imported [thermo-nuclear review skill](src/ai_reviewer/prompts/review.md)
  supplies the structural quality standard. Correctness and security findings
  also count. Models must report insufficient context explicitly.
- The consolidator sees completed reports and findings in bounded batches. It must preserve every finding exactly
  once. Original wording and explicit severity labels remain in feedback. Different
  locations receive separate threads; large groups split between findings at a
  conservative 60,000 UTF-8 byte limit per inline comment.
- Any blocking finding means **REQUEST_CHANGES**. All required complete reports with no
  blockers mean **APPROVE**. Nonblocking observations may accompany approval.
- Missing, invalid or incomplete results fail the run. They never become a clean
  verdict. Both head and base commits are rechecked immediately before posting.
- Re-running the same head/base pair does not duplicate a published review.
  Model calls still incur charges on reruns.

## Issue context

Each preparation collects issue titles, descriptions, states and source URLs once
and passes the same snapshot to enabled reviewers. Neither provider needs an issue:
GitHub only, Linear only, both and neither are supported. Duplicate references
within a provider produce one issue record.

- **GitHub:** uses the existing `GH_TOKEN`, with `issues: read` in the caller and
  reusable workflow. Reads linked issues (including manual Development links),
  plus `owner/repo#123` and `https://github.com/owner/repo/issues/123`
  references in the PR title/body. Local references require an issue keyword,
  such as `Fixes #123`, `Refs #123`, `Issue #123` or `Related to #123`, so colours
  and numeric URL fragments are not mistaken for tickets. Other PRs are excluded.
  GitHub authentication remains required for the review itself.
- **Linear:** add `LINEAR_API_KEY` to the target repository's Actions secrets.
  Missing or empty means no Linear lookup. With a key, reads Linear attachments
  matching the PR URL and ticket IDs such as `ENG-123` in the PR title, body or
  branch name, including Linear issue URLs and lowercase branch IDs. IDs must
  match a team visible to the key, so ordinary terms such as `UTF-8` are ignored.
  Team discovery is bounded to 100 teams and fails if the list is incomplete.
- **Failures:** configured-provider errors, inaccessible referenced issues and
  truncated results fail preparation. They never silently mean "no issue."
  Collection allows at most 20 issues and 20 explicit references per provider;
  issue text counts toward every packet’s 600,000-byte context limit.
- **Scope:** refreshes on review runs, not on issue edits alone. Does not crawl
  project boards, infer tickets from similar wording, or read issue comments.
  A GitHub token must have access to every referenced repository; the default
  workflow token may not access issues in another private repository.

Issue contents are untrusted requirements, not reviewer instructions. Linear
content is sent to configured model providers and saved in the calling repository's
review artifacts. Use a key scoped to data appropriate for that repository;
review feedback may quote issue requirements.

API contracts: [GitHub linked issues](https://docs.github.com/en/graphql/reference/pulls),
[Linear attachments by URL](https://linear.app/developers/attachments), and
[Linear issue lookup and personal API keys](https://linear.app/developers/graphql).

The Linear attachment query uses `attachmentsForURL(url: String!, first: Int)`
returning `AttachmentConnection!`, with `nodes` and `pageInfo`. This is verified
against Linear's [published SDK schema](https://github.com/linear/linear/blob/618a737a4e6aa279b7853c650f79f209a5142bbc/packages/sdk/src/schema.graphql#L39017-L39046),
not a flat attachment list. Mock fixtures preserve that provider contract;
authenticated live validation remains deferred.

## Rejected advisory locations

A schema-valid nonblocking finding outside the captured diff is excluded from
integration, consolidation and publication. Full source visibility does not make
an unchanged line a valid inline anchor. Invalid blocking locations still fail
the run, as do schema, completeness and coverage errors. No correction call is made.

The exact response remains in `work/<assignment>/<provider>/<attempt>/<provider>.json`.
`rejected-findings.json` beside it records each rejected finding, its zero-based
index in that response and the validation reason. These diagnostics are saved before
completion and continuation checks, including attempts that later fail. Accepted findings retain their
original order, wording and severity. Replay derives the same selection from the
raw response; the diagnostic sidecar is not trusted as review input. An affected
report's summary is replaced with a neutral completion/context statement so its
original prose cannot publish the rejected finding. The original summary remains
in the raw response. Usage and cost artifacts are unchanged.

## Failure diagnostics

Failed stages and model calls emit bounded JSON diagnostics to Actions logs and
`*.error.json` files in the evidence artifact. Model records identify the role,
model and phase (`preflight`, `generation`, or `validate_response`). Worker logs
identify assignment, provider and zero-based attempt. Numeric HTTP status and up
to 20 schema errors include known field names, array indexes and validation codes.
Unknown field names are redacted; response bodies, exception messages, validation
inputs/context, credentials and headers are never included. SDK parse failures
can still leave token usage unknown. Logging does not make a failed review pass.

These diagnostics run only when the workflow uses this implementation revision.
The production caller uses the implementation on `main`; updating an unmerged PR
does not change the reviewer executing against that PR.

## Security and limits

The workflow checks out only its pinned reviewer implementation. It reads PR
content through GitHub's API and never executes PR code, installs its packages,
or exposes shell tools to models. One job performs setup once and runs enabled independent reviews concurrently in a
single step. Configured provider API keys are available to that step; only preparation
and publication receive the reviewer credential. All stages share the runner and
workspace, so step-scoped environment variables are not a sandbox boundary.

**This is a bounded static review, not a replacement for the interactive agents.**
It can inspect tracked text files from immutable head, comparison and target-base
snapshots, including unchanged dependencies. It cannot run tests or browse links.
Model judgment can still be wrong or affected by malicious instructions in source;
structured output does not eliminate that risk.

### Large PRs

The **600,000-byte ceiling applies to serialized UTF-8 context per call**, including
source, patches, requirements, replies and the shared overview. It is a starting
safeguard, not a measured optimum. Preparation now permits up to 3,000 changed
files, 8 MB per blob and 32 MB per serialized snapshot. These collection bounds
replace the old 80-file and 150 KB per-version rejection.

1. Capture immutable head, base and merge-base evidence. Validate that original
   patches account for the complete before/after change. Reconstruct missing
   text patches with Git over inert blobs. Binary, unreadable, truncated or
   unsupported changed-file objects still fail without approval. Separately save
   complete tracked-file indexes and best-effort bounded UTF-8 blobs for all three
   revisions. Blob and byte limits leave uncaptured files indexed as unavailable.
   If fewer than eight GitHub requests remain, fail preparation explicitly.
2. Keep a whole file when it fits the unit budget. For oversized text, prefer
   Python function/class boundaries, then diff sections, then physical patch
   lines. Other languages use diff boundaries. Retain original patch records,
   old/new cursors, source offsets, modes, renames and final-newline state.
3. Compile the inventory and discussion index once per plan. Scan each captured
   version with a shared reference index using complete path/module tokens. Join
   normalized names only between test and non-test files. Reuse this relationship evidence during packing and integration.
   Pack direct references, source/test pairs and directory peers together using
   stable ordering. Fit checks count precomputed item bytes incrementally; full
   packets are serialized only after their scopes are chosen. A unit is capped at 160,000 bytes so two actual source units
   can accompany reports during integration. Local packets target 420,000 bytes,
   leaving 180,000 bytes for required follow-ups under the 600,000-byte ceiling.
   Indivisible oversized lines or required metadata fail explicitly; nothing is
   silently removed to fit.
4. Run each enabled reviewer at medium effort on every chunk with **two concurrent
   workers** in the same Actions job. Each provider gets the same list, literal-search
   and inclusive line-range tools over those snapshots. Tool results are bounded and
   recorded beside each attempt, then replayed against the captured Git blobs before
   coverage is accepted. If a tool budget is exhausted, the required assignment fails.
5. Enabled providers perform integration passes with actual source and reports.
   Every chunk gets an integration pass. Additional packets join direct references,
   source/test pairs and neighboring directory/file sections across chunks. Direct
   references come from full captured blobs, with their actual source lines included
   even outside changed hunks. Accumulated reports are paged into additional required
   integration assignments. This
   is a bounded heuristic, **not a complete dependency graph**. Models can search
   the complete captured tracked snapshots for dependencies outside these heuristics.
6. Recompute the plan and validate the coverage manifest once before consolidation
   and once before publication. The manifest stores assignment and plan hashes;
   full payloads stay in attempt artifacts. Reports are keyed by assignment and
   provider. Compile immutable diff anchors once per stage and reuse them across
   workers and replay. Whole, local and integration assignments receive explicit
   scope instructions. Repository-enabled reviews use bounded tools; contexts without
   a repository snapshot retain the changed-file request protocol. Require every
   enabled provider for all assignments.
   Coverage proves assignment/completion; it does not prove model understanding.
7. The consolidator groups up to 40 original findings per batch with global source IDs.
   All accepted findings, locations and severity survive exactly once. Cross-batch duplicate
   findings remain separate. Bounded summary passes carry assessments
   and disagreements into one overview, using neutral summaries for reports with
   rejected advisory locations as described above. Publication submits **one GitHub review**.

Chunked local and integration packets include a declaration index in
`overview.inventory[].declarations`. Python classes, functions and async functions
have qualified names and one-based inclusive LEFT/RIGHT ranges, including decorators.
Use the inventory's current filename for either side of a renamed file. The index
is compiled once per plan; unit splitting and packet navigation share that discovery.
Navigation fields are capped at 16,000 UTF-8 bytes across the inventory and trimmed
further when needed to fit a required source unit. Oversized integration and follow-up
packets omit navigation before paging reports or rejecting required evidence.
`declaration_navigation: truncated` means entries or entire file indexes were omitted; absence is not evidence that a
function or section does not exist. Unit splitting consumes complete discovery before navigation
is projected. Projections return new metadata; packet sizers retain unchanged inputs.
Discovery never executes source and does not replace reading it. Unsupported languages,
malformed Python and bare-CR line endings have explicit unavailable statuses; changed-file
source requests still work when repository tools are unavailable. Whole-source bodies stay
out of the shared overview, which carries only the repository snapshot digest. Index bytes
count toward existing packet limits; required source and review metadata still fail closed
if oversized.

Markdown `.md`/`.markdown` files additionally expose LEFT/RIGHT `sections`, using
raw heading names, levels and inclusive section ranges. Each range includes nested
sections until the next equal/shallower indexed heading or EOF. Both reviewers are
taught to request the relevant source using those locations. These sections share
the existing navigation byte limit and omission marker; they do not alter chunk
splitting. Status `partial` explicitly identifies a conservative syntax subset:
ATX (`#`) and single-line setext (underlined) headings, with up to three leading
spaces, excluding fenced code. Backtick/tilde fences require matching character and
sufficient closing length. Containers and HTML blocks are skipped conservatively
until a blank line (comments through their terminator); they are not parsed.
Multiline setext titles are omitted and inline markup is retained. This is not a
full CommonMark index: absence never establishes that a section does not exist.

Final incomplete responses, including `complete=false` with no requests, record
`FollowupExhausted` and reason `followup_budget_exhausted` in worker logs and
`work/<assignment>/<provider>/failure.json`, with validated attempt count, completed
follow-up rounds and bounded outstanding coordinates. Reasons, summaries, source and
provider error bodies are excluded. Unknown paths are redacted. The manifest retains
the exhausted attempt count and failed status. An earlier incomplete response without
requests records `IncompleteWithoutRequests`, reason `missing_source_requests`, and
the observed attempt count. Both are classified as `unresolved_evidence`, not API
transport failures. Neither triggers an extra call. For other failures, zero attempts
still means unknown; per-attempt artifacts remain the evidence of provider activity.
Incomplete work cannot proceed to approval. Offline navigation tests do not guarantee
that a model will finish within the three-attempt budget.

Provider token counts are checked before generation, including instructions and
output schemas. Each call allows at most **250,000 input tokens** and **16,000
output/reasoning tokens**. This conservative policy leaves headroom below the
verified [Claude 1M context](https://platform.claude.com/docs/en/models/overview)
and [Sol 1.05M context](https://developers.openai.com/api/docs/models/gpt-6.1-sol)
and stays below Sol's 272k long-context price threshold. Bytes are not converted
into guessed tokens. See [Claude token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting)
and [OpenAI token counting](https://developers.openai.com/api/docs/guides/token-counting).
Count every enabled provider input for every planned local scope before any reviewer generates
a report, including packets already routed to chunks.
If any exceeds capacity, re-plan with a 200,000-byte ceiling for every review
packet, including integration pages and follow-ups. Local chunks target half that
ceiling and source units at most one quarter, leaving integration headroom.
Each resulting call still undergoes provider token counting; there is no byte/token
conversion or guarantee that required metadata can fit. The fallback plan is bound
to the manifest and replayed by downstream stages. Other provider failures remain
failures, not partitioning signals.

Operational defaults are deliberate finite safeguards, not performance optima:

| Boundary | Default | Reason |
| --- | --- | --- |
| Changed-file collection | 3,000 files; 8 MB/blob; 32 MB snapshot; 10,000 API requests/client; 20,000 direct relations | Bound PR evidence collection |
| Repository snapshots | 50,000 tracked entries per revision; at most 4,000 unique blob requests, reserving six tree reads and two preparation requests; 2 MB per captured blob; 100 MB total unique captured bytes; recursive Git trees must be complete | Bound immutable head/comparison/base capture; binary, symlink, submodule, oversized and uncaptured blobs stay listed with diagnostics |
| Local planning | 64 chunks | Bound partition growth |
| Local + integration work | 128 assignments, each reviewed by all enabled providers | Bound relationship expansion and spending |
| Follow-ups | 3 total attempts/provider/assignment | Resolve missing evidence without an open-ended loop |
| Repository tools | 12 provider turns; 32 calls; 16,000 bytes/call; 256,000 returned bytes; 3,000,000 cumulative counted input tokens; 600 seconds per provider assignment | Bound iterative investigation; exhaustion fails review coverage |
| Repository reads | 50 files/list page; 25 search matches and 500 scanned files/call; 200 lines/read | Bound each visible result and filesystem work |
| Provider output | 40 findings and 60,000 serialized bytes/attempt | Bound report handling and integration input |
| Consolidation work | 128 finding batches + 128 summary calls | Bound aggregate consolidation |
| Publication | 1,000 comments; 8 MB comment payload; 60,000-byte body | Fail before any oversized or partial review write |
| Job | 2 workers; 180 minutes; no automatic API retries | Bound concurrency and execution time |

The initial packet remains capped at 600,000 bytes. Tool continuations have a
1,576,000-byte serialized-history ceiling, alongside the per-call and cumulative
token limits above. These are independent ceilings, not guaranteed usable capacity.
Line reads page complete lines by serialized JSON size and return `next_start`.
A single line exceeding the result budget returns an explicit diagnostic.

The repository-tool deadline is checked before tools/counting and after generation.
An in-flight provider request can extend one assignment by up to its 300-second
timeout; a response arriving after the deadline cannot complete the assignment.
Missing transcript files fail replay, including when a review used zero tools.
Repository-mode reports have no excerpt-request field and cannot enter legacy
follow-ups. Preparation fails if it cannot reserve requests for repository snapshots;
it never silently downgrades a requested repository review to excerpt-only access.

Application work limits live in `src/ai_reviewer/limits.py`; the job timeout lives
in the workflow. Increasing one requires reviewing downstream bounds. Equal values
with different purposes remain separate. The work limits are **not a dollar budget**. Large PRs can cost substantially more through repeated overview/source,
follow-ups and integration. There is no promise that an accepted snapshot will
finish: required evidence, output or total work may exceed a later bound, which
blocks approval. The **180-minute job timeout is the effective wall-clock limit**;
the assignment and per-call maxima cannot all be exhausted within it. Timeout may
interrupt a stage. Cost reporting and artifact upload use `always()`, but their
completion after job cancellation is not established. A timed-out run may therefore
lose usage/evidence and must never be treated as a clean review. A new run needs a fresh
artifact directory, as supplied by a new Actions job.


GitHub review submission is not atomic with the final revision check. A push in
that small interval can leave a review attached to the old commit. Enable stale
approval dismissal as described above. A failed subsequent run does not retract
older reviews by itself.

Review context and model reports are Actions artifacts retained for one day.
They contain changed source, tracked text blobs, tool results, discussion and issue
context. Treat their access as part of the calling repository's trust boundary. Public
review text and Actions logs must not contain secrets. The CLI deliberately suppresses
provider exception bodies.

Repository review preflight caps initial input at 150,000 tokens, reserving 100,000
for tool continuations under the 250,000 per-request limit. The 3,000,000 cumulative
ceiling counts full repeated inputs across at most 12 turns; it is not a billing
estimate. Requests retain the 300-second timeout; the 600-second assignment deadline
is checked before counting and after provider calls. A running call may finish after
the deadline, but its review is rejected. Time or result limits can bind before all
12 turns. No maximum is a promise of completion.


## Cost reporting

The final **Report API costs** step reads every nested attempt artifact and prints
Opus, Sonnet and separate Sol review/consolidation estimates, Claude/OpenAI
subtotals and a combined total. It runs after success or failure. An attempted
call with unavailable usage is **unknown**, never silently counted as free. A
stage never attempted is reported as **not run ($0)**. Published reviews end with
the same report in a collapsible **API costs** section.

Estimates use provider-reported input, output and cache token counts. Rates are
recorded in [models.py](src/ai_reviewer/models.py), checked on **2026-10-25** against
[Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) and
[OpenAI pricing](https://developers.openai.com/api/docs/pricing). They exclude tax,
credits and GitHub Actions charges. Update rates when provider pricing changes;
provider billing remains authoritative. Sonnet 5.5 and GPT-6.1 Sol consolidation use
$2 input, $10 output, $0.10 cache-read and $2.50 cache-write per million tokens.
The shared 250,000-token
input ceiling stays below the OpenAI long-prompt threshold. Per-attempt artifacts
use `sonnet` or `sol` names so costs follow the actual provider and model. Configure spending limits in each API
project independently.

## Development

Use **Python 3.14.8** and **Poetry 2.4.2**, matching the nplet development baseline.

```sh
poetry env use python3.14
poetry sync
poetry run pre-commit install
poetry check --lock
poetry run ruff format --check .
poetry run ruff check --no-fix .
poetry run mypy
poetry run pytest --cov --cov-report=term-missing
```

Open the repository root in VS Code, install `.vscode/extensions.json`
recommendations and select `.venv`. Editor settings, Python ignore rules,
formatting, typing and test tools are adapted from nplet. Commit `poetry.lock`.

Tests use synthetic GitHub and provider boundaries. They exercise artifact
handoff, approval guards, stale revisions, inline anchors, consolidation fidelity
and cost calculations without paid API calls. CI runs formatting, linting,
typing and tests, with an 80% branch-aware coverage floor.

### Source documentation

Follow the source-documentation pattern used in nplet and plainspoken across
production code, scripts and tests:

- Begin each module with its responsibility, boundary and relevant side effects.
- Give every named class, function and method a meaningful Google-style docstring.
  Use `Args`, `Returns`, `Raises` and `Attributes` where the contract needs them.
  Document constraints and failure conditions rather than repeating type hints.
- Introduce each logical phase of a multi-step routine with a concise comment.
  Together, the comments should explain the flow: input selection, validation,
  transformation, I/O, state changes and failure handling. Group related statements;
  do not narrate each assignment. Single-operation helpers need only a docstring.
- Describe what the code actually does, including limitations. Preserve vendored
  prompts and licenses rather than restyling third-party material.
- Review documentation manually as well as running Ruff. For documentation-only
  Python edits, compare syntax trees with docstrings removed and inspect runtime
  consumers of documentation. Here, argparse uses the CLI module docstring and
  Pydantic includes contract class docstrings in provider-facing schema descriptions.

## License and attribution

[MIT](LICENSE). The vendored review prompt is copied from the supplied local
`thermo-nuclear-code-quality-review` skill, based on the
[Cursor team kit skill](https://github.com/cursor/plugins/blob/main/cursor-team-kit/skills/thermo-nuclear-code-quality-review/SKILL.md).
Its upstream [MIT license](src/ai_reviewer/prompts/LICENSE.cursor) is included.

The [prompt manifest](src/ai_reviewer/prompts/manifest.json) records the verified
upstream commit, immutable source URLs, SHA-256 hashes and local modifications.

## Final merge checklist

- [ ] Complete the last candidate review and record its passing run.
- [ ] Keep required code-quality checks passing on the final revision.
