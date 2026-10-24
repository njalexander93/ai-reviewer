"""Walk Git patch records with one canonical LF/hunk/cursor interpretation.

Consumers share coordinates for source splitting, completeness checks and inline
anchors. This module preserves records without fetching source or proving that a
patch is complete; validate_patch owns that separate boundary.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass


def lines(text: str) -> list[str]:
    """Return LF-delimited records from text, retaining terminators and Unicode content.

    An empty string yields no records. A final unterminated line remains unterminated;
    carriage returns and Unicode line separators never create additional Git lines.
    """
    return [part + "\n" for part in text.split("\n")[:-1]] + (
        [text.rsplit("\n", 1)[-1]] if text and not text.endswith("\n") else []
    )


@dataclass(frozen=True)
class PatchRecord:
    """Retain immutable cursor metadata for one original patch record.

    Attributes:
        text: Exact LF-delimited record, including any terminator.
        left: Old-side coordinate before consumption; zero-count headers may use zero.
        right: New-side coordinate before consumption, with the same zero-count rule.
        counts: Old/new hunk counts on a header, otherwise None.
        in_hunk: Whether a valid hunk header has appeared before or at this record.
    """

    text: str
    left: int
    right: int
    counts: tuple[int, int] | None
    in_hunk: bool

    @property
    def consumes(self) -> tuple[bool, bool]:
        """Return (old, new) consumption flags for this record.

        Context consumes both sides, deletion only old, addition only new. Headers,
        metadata outside a hunk and newline markers consume neither side.
        """
        body = self.in_hunk and self.counts is None
        return body and self.text.startswith((" ", "-")), body and self.text.startswith((" ", "+"))

    @property
    def end(self) -> tuple[int, int]:
        """Return old/new cursors immediately after this record's source consumption."""
        old, new = self.consumes
        return self.left + old, self.right + new


def walk_patch(patch: str) -> Iterator[PatchRecord]:
    """Yield every LF-delimited record from patch with canonical cursor metadata.

    Valid unified headers reset coordinates/counts; omitted counts mean one and
    explicit zero counts retain Git coordinates. Other text is retained, not rejected.
    The caller must validate completeness separately; this iterator performs no I/O.
    """
    left = right = 1
    in_hunk = False
    # Headers reset cursors; every other original record remains visible to validators.
    for text in lines(patch):
        match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", text)
        counts = None
        if match:
            a, ac, b, bc = match.groups()
            left, right = int(a), int(b)
            counts = int(ac or 1), int(bc or 1)
            in_hunk = True
        # Yield pre-consumption coordinates, then advance only the affected sides.
        record = PatchRecord(text, left, right, counts, in_hunk)
        yield record
        left, right = record.end


def diff_lines(patch: str) -> set[tuple[str, int]]:
    """Return a set of (LEFT/RIGHT, line) anchors consumed by the supplied patch.

    Includes context on both sides. No completeness or source validation occurs;
    malformed/incomplete patches may yield partial anchors. Patch is not modified.
    """
    # Only source-consuming records create valid inline review coordinates.
    anchors = set()
    for record in walk_patch(patch):
        old, new = record.consumes
        if old:
            anchors.add(("LEFT", record.left))
        if new:
            anchors.add(("RIGHT", record.right))
    return anchors
