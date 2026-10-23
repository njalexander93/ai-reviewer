"""Publication safety contracts derived from the agreed review policy."""

import pytest

from ai_reviewer.contracts import Consolidation, Finding, Group, Review, reconcile


def test_consolidator_cannot_discard_a_blocker() -> None:
    """A clean summary must not erase an independently reported blocker."""
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
    with pytest.raises(ValueError, match="exactly once"):
        reconcile(
            [review, Review(complete=True, summary="Independent", findings=[])],
            Consolidation(summary="Looks good", groups=[]),
        )


@pytest.mark.parametrize("ids", [["0:0", "0:0"], ["invented"], []])
def test_consolidation_requires_exact_partition(ids: list[str]) -> None:
    """Duplicate or invented IDs cannot satisfy preservation of the source findings."""
    finding = Finding(
        path="app.py", line=1, side="RIGHT", title="Bug", body="Evidence", blocking=True
    )
    review = Review(complete=True, summary="Done", findings=[finding])
    groups = [Group(source_ids=ids)] if ids else []
    with pytest.raises(ValueError, match="exactly once"):
        reconcile(
            [review, Review(complete=True, summary="Independent", findings=[])],
            Consolidation(summary="Done", groups=groups),
        )


def test_merged_duplicates_retain_both_original_findings() -> None:
    """Sonnet grouping cannot overwrite the severity or evidence of either reviewer."""
    one = Finding(path="app.py", line=1, side="RIGHT", title="Bug", body="Evidence", blocking=True)
    two = Finding(path="app.py", line=2, side="RIGHT", title="Maybe", body="Other", blocking=False)
    reviews = [
        Review(complete=True, summary="Done", findings=[one]),
        Review(complete=True, summary="Done", findings=[two]),
    ]
    assert reconcile(
        reviews, Consolidation(summary="Done", groups=[Group(source_ids=["0:0", "1:0"])])
    ) == [[one, two]]
