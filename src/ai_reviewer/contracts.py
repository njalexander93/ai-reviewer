"""Validate model output and preserve every finding through consolidation."""

from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    """Reject unexpected fields at artifact and model boundaries."""

    model_config = ConfigDict(extra="forbid", strict=True)


class Finding(Contract):
    """An actionable finding anchored to a reviewed diff line."""

    path: str = Field(min_length=1, max_length=1000)
    line: int = Field(ge=1)
    side: Literal["LEFT", "RIGHT"]
    title: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=4000)
    blocking: bool


class Review(Contract):
    """A reviewer must explicitly attest that supplied context was sufficient."""

    complete: bool
    summary: str = Field(min_length=1, max_length=6000)
    findings: list[Finding] = Field(max_length=40)


class Group(Contract):
    """Source IDs identify duplicates without granting permission to delete findings."""

    source_ids: list[str] = Field(min_length=1, max_length=80)


class Consolidation(Contract):
    """Sonnet writes a summary and partitions the original findings."""

    summary: str = Field(min_length=1, max_length=6000)
    groups: list[Group] = Field(max_length=80)


def reconcile(reviews: list[Review], result: Consolidation) -> list[list[Finding]]:
    """Require complete reviews and an exact partition of all source findings.

    Args:
        reviews: Independently completed reports in stable provider order.
        result: Proposed duplicate groups from Sonnet.

    Returns:
        Original findings grouped for publication, with their severity unchanged.

    Raises:
        ValueError: A review is incomplete or a source is missing, repeated or invented.
    """
    if len(reviews) != 2 or not all(review.complete for review in reviews):
        raise ValueError("Incomplete review; approval prohibited")
    sources = {
        f"{i}:{j}": finding
        for i, review in enumerate(reviews)
        for j, finding in enumerate(review.findings)
    }
    ids = [source for group in result.groups for source in group.source_ids]
    if Counter(ids) != Counter({key: 1 for key in sources}):
        raise ValueError("Consolidation must preserve every source exactly once")
    return [[sources[key] for key in group.source_ids] for group in result.groups]
