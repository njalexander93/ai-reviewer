"""Define model-output contracts and preserve findings through consolidation.

Pydantic rejects extra fields and type coercion. Reconciliation checks required
reports and source-ID partition without performing I/O or changing findings.
Class docstrings also become descriptions in provider-facing JSON schemas.
"""

from collections import Counter
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ai_reviewer.limits import (
    MAX_COMPLETED_FINDINGS,
    MAX_CONSOLIDATION_GROUPS,
    MAX_FILES,
    MAX_FINAL_GROUPS,
    MAX_FINDINGS,
    MAX_GROUP_SOURCES,
    MAX_SOURCE_REQUESTS,
)
from ai_reviewer.patch import diff_lines

Provider = Literal["claude", "openai"]
# Only canonical nonempty selections are accepted in persisted run manifests.
Providers = (
    tuple[Literal["claude"]]
    | tuple[Literal["openai"]]
    | tuple[Literal["claude"], Literal["openai"]]
)


class Contract(BaseModel):
    """Reject unexpected fields at artifact and model boundaries."""

    model_config = ConfigDict(extra="forbid", strict=True)


class Finding(Contract):
    """Describe one actionable finding anchored to a reviewed diff line.

    Attributes:
        path: Repository-relative filename from the diff.
        line: One-based line number on the selected side.
        side: LEFT for the old version or RIGHT for the new version.
        title: Short issue heading.
        body: Evidence and consequences supporting the finding.
        blocking: Whether the finding requires changes before approval.
    """

    path: str = Field(min_length=1, max_length=1000)
    line: int = Field(ge=1)
    side: Literal["LEFT", "RIGHT"]
    title: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=4000)
    blocking: bool


class Review(Contract):
    """Record an independent assessment of the supplied context.

    Attributes:
        complete: Whether the reviewer could assess all supplied changes.
        summary: Overall assessment, including context limitations when incomplete.
        findings: At most 40 actionable observations; an empty list means none found.
    """

    complete: bool
    summary: str = Field(min_length=1, max_length=6000)
    findings: list[Finding] = Field(max_length=MAX_FINDINGS)


class SourceRequest(Contract):
    """Request inclusive Git lines from a captured changed-file version only."""

    # Current captured filename; LEFT reads before, RIGHT reads after, including renames.
    path: str
    side: Literal["LEFT", "RIGHT"]
    # One-based inclusive bounds; source_sections checks ordering and availability.
    start: int = Field(ge=1)
    end: int = Field(ge=1)
    # Nonempty explanation of the evidence needed; never executable instructions.
    reason: str = Field(min_length=1, max_length=1000)


class ChunkReview(Review):
    """Review assigned sections; required missing evidence must be requested explicitly."""

    # Required missing ranges. advance rejects complete reports with outstanding requests.
    requests: list[SourceRequest] = Field(max_length=MAX_SOURCE_REQUESTS)


class CompletedReview(Review):
    """Preserve findings from up to three attempts of one assigned provider review."""

    # Retain findings from all successful attempts; summary/complete follow Review.
    findings: list[Finding] = Field(max_length=MAX_COMPLETED_FINDINGS)


class Group(Contract):
    """Identify duplicate findings without granting permission to delete them.

    Attributes:
        source_ids: Nonempty list of zero-based "review:finding" IDs, such as "0:1".
            Reconciliation validates membership and uniqueness.
    """

    source_ids: list[str] = Field(min_length=1, max_length=MAX_GROUP_SOURCES)


class Consolidation(Contract):
    """Hold the consolidator's summary and proposed partition of original findings.

    Attributes:
        summary: Combined assessment; it does not control the approval verdict.
        groups: Duplicate groups covering every source exactly once. Empty when
            neither independent report contains findings.
    """

    summary: str = Field(min_length=1, max_length=6000)
    groups: list[Group] = Field(max_length=MAX_CONSOLIDATION_GROUPS)


class FinalConsolidation(Consolidation):
    """Hold bounded batches after each consolidator response preserves its source partition."""

    # Aggregate partition across bounded batches; summary retains Consolidation semantics.
    groups: list[Group] = Field(max_length=MAX_FINAL_GROUPS)


class ChangedFile(Contract):
    """Carry only changed-file evidence used by reviewers and publication.

    Attributes:
        filename: Current repository-relative path.
        previous_filename: Old path for renames, otherwise None.
        status: GitHub change status.
        patch: Unified diff with Git newline coordinates.
        before: Exact old blob text, empty for added files.
        after: Exact new blob text, empty for removed files.
        before_mode: Old Git mode; None when the old version does not exist.
        after_mode: New Git mode; 120000 identifies symlink text, not its target.
    """

    filename: str = Field(min_length=1)
    previous_filename: str | None
    status: str
    patch: str
    before: str
    after: str
    before_mode: str | None
    after_mode: str | None


class PriorComment(Contract):
    """Keep discussion content and coordinates without repeated diff hunks or API metadata.

    Attributes:
        path: Commented file path.
        line: Current line, or None for an outdated comment.
        side: Comment's diff side, or None for file-level discussion.
        body: Original discussion text, treated as untrusted input.
        author: GitHub login, or None for a deleted account.
    """

    path: str
    line: int | None
    side: str | None
    body: str
    author: str | None


class IssueContext(Contract):
    """Carry untrusted issue requirements and their source, without provider metadata.

    Attributes:
        provider: Service that supplied the issue.
        url: Canonical issue URL for attribution and deduplication.
        title: Issue title.
        description: Full issue body, or None when absent.
        state: Provider's issue state label.
    """

    provider: Literal["github", "linear"]
    url: str = Field(min_length=1)
    title: str
    description: str | None
    state: str


class ReviewContext(Contract):
    """Bind minimal review evidence to an immutable PR revision pair.

    Attributes:
        repo: Owner/repository pair validated by the GitHub adapter.
        number: Positive PR number.
        head: Reviewed head commit SHA.
        base: Base branch tip observed during preparation.
        diff_base: Merge-base commit used to read old file versions.
        reviewer: GitHub login expected to publish the review.
        title: PR title.
        description: PR body, or None when absent.
        files: Complete bounded list of changed-file evidence.
        previous_comments: Minimal prior inline discussion records.
        issues: Linked or explicitly referenced issue requirements from both providers.
    """

    repo: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    number: int = Field(ge=1)
    head: str = Field(pattern=r"^[0-9a-f]{40}$")
    base: str = Field(pattern=r"^[0-9a-f]{40}$")
    diff_base: str = Field(pattern=r"^[0-9a-f]{40}$")
    reviewer: str = Field(min_length=1)
    title: str
    description: str | None
    files: list[ChangedFile] = Field(max_length=MAX_FILES)
    previous_comments: list[PriorComment]
    issues: list[IssueContext] = Field(default_factory=list, max_length=40)


def source_findings(reviews: list[Review]) -> dict[str, Finding]:
    """Validate completeness and assign stable IDs; load_reviews enforces exact coverage.

    Args:
        reviews: Complete reports in deterministic assignment/provider order.

    Returns:
        Original findings keyed by zero-based review:finding IDs.

    Raises:
        ValueError: No reports are supplied or a report is incomplete.
            Exact assignment/provider coverage is enforced by load_reviews, not here.
    """
    # Enforce the same completeness boundary before consolidation and publication.
    if not reviews or not all(review.complete for review in reviews):
        raise ValueError("Incomplete review; approval prohibited")
    # Index original objects once for both the payload and reconciliation contract.
    return {
        f"{i}:{j}": finding
        for i, review in enumerate(reviews)
        for j, finding in enumerate(review.findings)
    }


def reconcile(reviews: list[Review], result: Consolidation) -> list[list[Finding]]:
    """Require complete reviews and an exact partition of all source findings.

    Args:
        reviews: Complete reports in deterministic assignment/provider order.
        result: Proposed duplicate groups from the consolidator.

    Returns:
        Original findings grouped for publication, with their severity unchanged.

    Raises:
        ValueError: Required complete reports are absent, or a source is missing,
            repeated or invented.
    """
    # Reuse the canonical completeness check and source-index contract.
    sources = source_findings(reviews)
    # Require an exact partition so grouping cannot discard or invent evidence.
    ids = [source for group in result.groups for source in group.source_ids]
    if Counter(ids) != Counter({key: 1 for key in sources}):
        raise ValueError("Consolidation must preserve every source exactly once")
    # Return original objects, preserving wording and blocker flags.
    return [[sources[key] for key in group.source_ids] for group in result.groups]


class IndependentSummary(Contract):
    """Label an original assessment where assignment/provider identity is known.

    Attributes:
        assignment: Whole-work assignment ID supplied by the canonical loader.
        provider: Independent reviewer whose label publication must display.
        summary: Original report assessment, without consolidation rewriting.
    """

    assignment: str
    provider: Literal["claude", "openai"]
    summary: str


class CoveredReviews(Contract):
    """Carry trusted canonical-loader data without replaying artifacts in publication.

    The CLI obtains this only after exact work and anchor validation. It is internal
    application data, not an unforgeable capability or accepted wire format.

    Attributes:
        providers: Validated canonical reviewer set from the run manifest.
        reviews: Complete reports in stable assignment/provider order for source IDs.
        independent_summaries: Explicit whole-review identity and original assessments.
        snapshot: Hash binding these reports to the captured context's exact JSON.
        summary: Human-readable coverage statement; not evidence of model understanding.
    """

    providers: Providers
    reviews: list[Review]
    independent_summaries: list[IndependentSummary]
    snapshot: str
    summary: str


# Read-only current path to immutable (LEFT/RIGHT, one-based line) coordinates.
AnchorIndex = Mapping[str, frozenset[tuple[str, int]]]


def compile_anchors(context: ReviewContext) -> AnchorIndex:
    """Return a read-only path-to-frozen-coordinate index for context's captured patches.

    Context must have unique current paths; callers validate patch completeness during
    preparation. The index is shared across workers without mutation and contains
    LEFT/RIGHT one-based inline locations, not a proof of model understanding.
    """
    return MappingProxyType(
        {file.filename: frozenset(diff_lines(file.patch)) for file in context.files}
    )


def validate_anchors(anchors: AnchorIndex, reviews: list[Review]) -> None:
    """Reject invalid model locations before purchasing consolidation.

    Args:
        anchors: Precompiled immutable snapshot coordinates.
        reviews: Independent reports with locations to validate.

    Raises:
        ValueError: A finding is outside the captured diff. Reports remain on disk
            for diagnosis; this fails the run rather than silently dropping findings.
    """
    # Reject any unsupported location rather than filtering findings out of the verdict.
    for review in reviews:
        for finding in review.findings:
            if (finding.side, finding.line) not in anchors.get(finding.path, frozenset()):
                raise ValueError("Finding is outside the captured diff")
