"""Define model-output contracts and preserve findings through consolidation.

Pydantic rejects extra fields and type coercion. Reconciliation checks the two
reports and source-ID partition without performing I/O or changing findings.
Class docstrings also become descriptions in provider-facing JSON schemas.
"""

from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
    findings: list[Finding] = Field(max_length=40)


class Group(Contract):
    """Identify duplicate findings without granting permission to delete them.

    Attributes:
        source_ids: Nonempty list of zero-based "review: finding" IDs, without
            spaces. Reconciliation validates membership and uniqueness.
    """

    source_ids: list[str] = Field(min_length=1, max_length=80)


class Consolidation(Contract):
    """Hold Sonnet's summary and proposed partition of original findings.

    Attributes:
        summary: Combined assessment; it does not control the approval verdict.
        groups: Duplicate groups covering every source exactly once. Empty when
            neither independent report contains findings.
    """

    summary: str = Field(min_length=1, max_length=6000)
    groups: list[Group] = Field(max_length=80)


def reconcile(reviews: list[Review], result: Consolidation) -> list[list[Finding]]:
    """Require complete reviews and an exact partition of all source findings.

    Args:
        reviews: Exactly two complete reports, ordered Claude then OpenAI.
        result: Proposed duplicate groups from Sonnet.

    Returns:
        Original findings grouped for publication, with their severity unchanged.

    Raises:
        ValueError: There are not two complete reports, or a source is missing,
            repeated or invented.
    """
    # Both independent assessments must be present and complete.
    if len(reviews) != 2 or not all(review.complete for review in reviews):
        raise ValueError("Incomplete review; approval prohibited")
    # Give each original finding an ID independent of Sonnet's grouping choices.
    sources = {
        f"{i}:{j}": finding
        for i, review in enumerate(reviews)
        for j, finding in enumerate(review.findings)
    }
    # Require an exact partition so grouping cannot discard or invent evidence.
    ids = [source for group in result.groups for source in group.source_ids]
    if Counter(ids) != Counter({key: 1 for key in sources}):
        raise ValueError("Consolidation must preserve every source exactly once")
    # Return original objects, preserving wording and blocker flags.
    return [[sources[key] for key in group.source_ids] for group in result.groups]
