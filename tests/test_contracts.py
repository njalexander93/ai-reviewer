"""Publication safety contracts derived from the agreed review policy."""

import pytest

from ai_reviewer.contracts import (
    Consolidation,
    Finding,
    Group,
    Review,
    ReviewContext,
    reconcile,
    source_findings,
)
from tests.test_github import Server, bundle


def test_consolidator_cannot_discard_a_blocker() -> None:
    """A clean summary must not erase an independently reported blocker."""
    # Create an independent report containing a concrete blocking finding.
    review = Review(
        complete=True,
        summary="Bug found",
        findings=[
            Finding(
                path="app.py",
                line=2,
                side="RIGHT",
                title="Lost data",
                body="This overwrites existing data.",
                blocking=True,
            )
        ],
    )
    # Reject a clean consolidation that omits the original blocker.
    with pytest.raises(ValueError, match="exactly once"):
        reconcile(
            [review, Review(complete=True, summary="Independent", findings=[])],
            Consolidation(summary="Looks good", groups=[]),
        )


@pytest.mark.parametrize("ids", [["0:0", "0:0"], ["invented"], []])
def test_consolidation_requires_exact_partition(ids: list[str]) -> None:
    """Duplicate or invented IDs cannot satisfy preservation of the source findings."""
    # Pair one finding with a duplicated, invented or missing source-ID assignment.
    finding = Finding(
        path="app.py", line=1, side="RIGHT", title="Bug", body="Evidence", blocking=True
    )
    review = Review(complete=True, summary="Done", findings=[finding])
    groups = [Group(source_ids=ids)] if ids else []
    # Reject every assignment that fails to preserve the source exactly once.
    with pytest.raises(ValueError, match="exactly once"):
        reconcile(
            [review, Review(complete=True, summary="Independent", findings=[])],
            Consolidation(summary="Done", groups=groups),
        )


def test_merged_duplicates_retain_both_original_findings() -> None:
    """Sonnet grouping cannot overwrite the severity or evidence of either reviewer."""
    # Give two reviewers different evidence and severity for a grouped issue.
    one = Finding(path="app.py", line=1, side="RIGHT", title="Bug", body="Evidence", blocking=True)
    two = Finding(path="app.py", line=2, side="RIGHT", title="Maybe", body="Other", blocking=False)
    reviews = [
        Review(complete=True, summary="Done", findings=[one]),
        Review(complete=True, summary="Done", findings=[two]),
    ]
    # Require the original findings to survive grouping without modification.
    assert reconcile(
        reviews, Consolidation(summary="Done", groups=[Group(source_ids=["0:0", "1:0"])])
    ) == [[one, two]]


@pytest.mark.parametrize("count", [0, 1, 3])
def test_source_index_requires_exactly_two_reports(count: int) -> None:
    """The shared source-index boundary rejects missing or extra reviewer reports."""
    # Empty findings do not make an incorrect reviewer count acceptable.
    reports = [Review(complete=True, summary="Complete", findings=[]) for _ in range(count)]
    with pytest.raises(ValueError, match="Incomplete"):
        source_findings(reports)


def test_context_rejects_opaque_or_missing_fields() -> None:
    """Artifact loading cannot accept arbitrary metadata or silently missing evidence."""
    # Obtain the valid fixture, then remove an invariant or add API passthrough noise.
    context, _, _ = bundle(Server())
    for invalid in [
        {**context.model_dump(), "raw_url": "extra"},
        {k: v for k, v in context.model_dump().items() if k != "files"},
    ]:
        with pytest.raises(ValueError):
            ReviewContext.model_validate(invalid)
