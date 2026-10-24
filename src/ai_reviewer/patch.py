"""Walk Git LF-delimited patch records with one canonical hunk/cursor interpretation."""

import re
from collections.abc import Iterator
from dataclasses import dataclass


def lines(text: str) -> list[str]:
    """Split only at Git's LF boundaries, retaining termination and Unicode."""
    return [part + "\n" for part in text.split("\n")[:-1]] + (
        [text.rsplit("\n", 1)[-1]] if text and not text.endswith("\n") else []
    )


@dataclass(frozen=True)
class PatchRecord:
    """Carry one record's one-based cursors, optional hunk counts and consumed sides."""

    text: str
    left: int
    right: int
    counts: tuple[int, int] | None
    in_hunk: bool

    @property
    def consumes(self) -> tuple[bool, bool]:
        """Headers, metadata and newline markers consume neither source side."""
        body = self.in_hunk and self.counts is None
        return body and self.text.startswith((" ", "-")), body and self.text.startswith((" ", "+"))

    @property
    def end(self) -> tuple[int, int]:
        """Return cursors immediately after this record."""
        old, new = self.consumes
        return self.left + old, self.right + new


def walk_patch(patch: str) -> Iterator[PatchRecord]:
    """Yield every original record; zero-count headers retain their Git coordinates."""
    left = right = 1
    in_hunk = False
    for text in lines(patch):
        match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", text)
        counts = None
        if match:
            a, ac, b, bc = match.groups()
            left, right = int(a), int(b)
            counts = int(ac or 1), int(bc or 1)
            in_hunk = True
        record = PatchRecord(text, left, right, counts, in_hunk)
        yield record
        left, right = record.end


def diff_lines(patch: str) -> set[tuple[str, int]]:
    """Extract inline coordinates without claiming that the patch is complete."""
    anchors = set()
    for record in walk_patch(patch):
        old, new = record.consumes
        if old:
            anchors.add(("LEFT", record.left))
        if new:
            anchors.add(("RIGHT", record.right))
    return anchors
