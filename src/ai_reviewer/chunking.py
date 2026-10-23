"""Plan deterministic bounded reviews over an immutable changed-file snapshot.

Units own contiguous original patch records exactly once. Source excerpts retain
Git line coordinates; additional source can only come from the same snapshot.
No reviewed code is imported or executed. Coverage establishes work, not judgment.
"""

import ast
import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import Field

from ai_reviewer.contracts import ChangedFile, Contract, ReviewContext, SourceRequest

PACKET_BYTES = 600_000
UNIT_BYTES = 160_000
CHUNK_BYTES = 420_000
MAX_CHUNKS = 64
MAX_SNAPSHOT_BYTES = 32_000_000
MAX_BLOB_BYTES = 8_000_000
MAX_WORK = 128


def encode(value: Any) -> str:
    """Serialize exact Unicode evidence consistently without ASCII expansion."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def digest(text: str) -> str:
    """Bind persisted work to the exact UTF-8 payload."""
    return hashlib.sha256(text.encode()).hexdigest()


def lines(text: str) -> list[str]:
    """Split only at Git's LF boundaries, retaining termination and Unicode."""
    # Python splitlines recognizes Unicode separators too; Git does not.
    return [part + "\n" for part in text.split("\n")[:-1]] + (
        [text.rsplit("\n", 1)[-1]] if text and not text.endswith("\n") else []
    )


class Unit(Contract):
    """Own a half-open patch-record interval and its supporting source excerpts."""

    id: str
    filename: str
    previous_filename: str | None
    status: str
    before_mode: str | None
    after_mode: str | None
    patch_start: int
    patch_end: int
    patch: str
    old_cursor: int
    new_cursor: int
    before_start: int
    before: str
    after_start: int
    after: str


class Chunk(Contract):
    """Group related units under a stable job identifier."""

    id: str
    units: list[Unit]

    def payload(self, context: ReviewContext) -> str:
        """Combine local evidence with shared requirements and relevant discussion."""
        paths = {u.filename for u in self.units} | {u.previous_filename for u in self.units}
        return encode(
            {
                "overview": overview(context),
                "scope": self.model_dump(),
                "previous_comments": [
                    c.model_dump()
                    for c in context.previous_comments
                    if c.path in paths or c.path not in {f.filename for f in context.files}
                ],
            }
        )


class Plan(Contract):
    """Select the unchanged whole-packet path or exact partitioned work."""

    snapshot: str
    whole: str | None
    chunks: list[Chunk] = Field(max_length=MAX_CHUNKS)


def overview(context: ReviewContext) -> dict[str, Any]:
    """Include fixed revisions, complete requirements and a changed-file inventory."""
    data = context.model_dump(exclude={"files", "previous_comments"})
    data["inventory"] = [
        {
            "filename": f.filename,
            "previous_filename": f.previous_filename,
            "status": f.status,
            "before_lines": f.before.count("\n")
            + int(bool(f.before) and not f.before.endswith("\n")),
            "after_lines": f.after.count("\n") + int(bool(f.after) and not f.after.endswith("\n")),
        }
        for f in sorted(context.files, key=lambda f: f.filename)
    ]
    return data


def declarations(filename: str, text: str) -> list[tuple[int, int]]:
    """Recognize Python declaration boundaries without importing reviewed code.

    Unsupported or invalid syntax falls back to diff and physical-line boundaries.
    """
    if not filename.endswith(".py"):
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError, ValueError, RecursionError:
        return []
    return sorted(
        (node.lineno, node.end_lineno or node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


def units(file: ChangedFile) -> list[Unit]:
    """Prefer a whole file, then declarations, hunks and bounded patch fragments.

    Every original patch record belongs to one unit, even newline markers and
    headers. A fragment carries its starting old/new cursor independently of any
    hunk header. An indivisible oversized physical line fails explicitly.
    """
    records = lines(file.patch)
    old, new = lines(file.before), lines(file.after)
    old_defs, new_defs = (
        declarations(file.filename, file.before),
        declarations(file.filename, file.after),
    )
    cursors: list[tuple[int, int]] = []
    left = right = 1
    boundaries = []
    ends: list[tuple[int, int]] = []
    definition_starts = {start for start, _ in new_defs}
    in_hunk = False
    # Capture cursor coordinates before every record, including split hunks.
    for index, record in enumerate(records):
        match = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", record)
        if match:
            left, right = map(int, match.groups())
            boundaries.append(index)
            in_hunk = True
        if right in definition_starts:
            boundaries.append(index)
        cursors.append((left, right))
        if in_hunk and not match:
            left += int(record.startswith((" ", "-")))
            right += int(record.startswith((" ", "+")))
        ends.append((left, right))
    cursors.append((left, right))

    def excerpt(
        source: list[str], start: int, end: int, definitions: list[tuple[int, int]], expand: bool
    ) -> tuple[int, str]:
        """Keep nearby physical lines, optionally extending to enclosing declarations."""
        lo, hi = max(1, start - 20), min(len(source), end + 20)
        if expand:
            for a, b in definitions:
                if a <= start <= b or a <= end <= b:
                    lo, hi = min(lo, a), max(hi, b)
        return lo, "".join(source[lo - 1 : hi])

    def make(start: int, end: int, whole: bool = False, expand: bool = False) -> Unit:
        """Build one interval with source offsets and unchanged file metadata."""
        a, b = cursors[start]
        c, d = ends[end - 1] if end > start else (a, b)
        before_start, before = (1, file.before) if whole else excerpt(old, a, c, old_defs, expand)
        after_start, after = (1, file.after) if whole else excerpt(new, b, d, new_defs, expand)
        return Unit(
            id=f"{file.filename}:{start}-{end}",
            filename=file.filename,
            previous_filename=file.previous_filename,
            status=file.status,
            before_mode=file.before_mode,
            after_mode=file.after_mode,
            patch_start=start,
            patch_end=end,
            patch="".join(records[start:end]),
            old_cursor=a,
            new_cursor=b,
            before_start=before_start,
            before=before,
            after_start=after_start,
            after=after,
        )

    whole = make(0, len(records), whole=True)
    if len(whole.model_dump_json().encode()) <= UNIT_BYTES:
        return [whole]
    if not records:
        # Mode-only changes and identical renames still review all blob text.
        result = []
        start = 0
        total = max(len(old), len(new))
        while start < total:
            end = total
            while True:
                unit = whole.model_copy(
                    update={
                        "id": f"{file.filename}:source-{start}-{end}",
                        "before_start": start + 1,
                        "after_start": start + 1,
                        "before": "".join(old[start:end]),
                        "after": "".join(new[start:end]),
                    }
                )
                if len(unit.model_dump_json().encode()) <= UNIT_BYTES:
                    break
                if end - start <= 1:
                    raise ValueError("Indivisible source line exceeds unit budget")
                end = (start + end) // 2
            result.append(unit)
            start = end
        return result
    pending = [(0, len(records))]
    result = []
    # Split deterministically; preserve the input order rather than stack order.
    while pending:
        start, end = pending.pop()
        candidate = make(start, end, expand=True)
        if len(candidate.model_dump_json().encode()) > UNIT_BYTES:
            candidate = make(start, end)
        if len(candidate.model_dump_json().encode()) <= UNIT_BYTES:
            result.append(candidate)
            continue
        if end - start <= 1:
            raise ValueError("Indivisible source or patch line exceeds unit budget")
        midpoint = (start + end) // 2
        cuts = [i for i in boundaries if start < i < end]
        cut = min(cuts, key=lambda i: (abs(i - midpoint), i)) if cuts else midpoint
        pending.extend([(cut, end), (start, cut)])
    return result


def relation_key(left: str, right: str) -> tuple[str, str]:
    """Identify an undirected file relationship independently of input ordering."""
    return (left, right) if left < right else (right, left)


def direct_relations(context: ReviewContext) -> dict[tuple[str, str], list[SourceRequest]]:
    """Find direct references in full captured blobs, never only in hunk excerpts.

    Source/test names imply a relationship. Textual path references additionally
    retain a short actual source range showing the reference on each side found.
    The 20,000-relation bound prevents a dense name/reference graph exhausting work.
    """
    files = sorted(context.files, key=lambda f: f.filename)
    result: dict[tuple[str, str], list[SourceRequest]] = {}
    for index, left in enumerate(files):
        for right in files[index + 1 :]:
            names = [
                PurePosixPath(f.filename).stem.removeprefix("test_").removesuffix("_test")
                for f in (left, right)
            ]
            evidence = []
            for source, target in ((left, right), (right, left)):
                references = [
                    target.filename,
                    str(PurePosixPath(target.filename).with_suffix("")).replace("/", "."),
                ]
                if target.previous_filename:
                    references.append(target.previous_filename)
                versions: list[tuple[Literal["LEFT", "RIGHT"], str]] = [
                    ("LEFT", source.before),
                    ("RIGHT", source.after),
                ]
                for side, text in versions:
                    positions = [text.find(ref) for ref in references if ref]
                    positions = [position for position in positions if position >= 0]
                    if not positions:
                        continue
                    line = text.count("\n", 0, min(positions)) + 1
                    total = text.count("\n") + int(bool(text) and not text.endswith("\n"))
                    evidence.append(
                        SourceRequest(
                            path=source.filename,
                            side=side,
                            start=max(1, line - 10),
                            end=min(total, line + 10),
                            reason="Captured direct path reference",
                        )
                    )
            if evidence or names[0] == names[1]:
                result[relation_key(left.filename, right.filename)] = evidence
                if len(result) > 20_000:
                    raise ValueError("Captured relationship budget exceeded")
    return result


def related(a: Unit, b: Unit, links: dict[tuple[str, str], list[SourceRequest]]) -> bool:
    """Prefer captured direct relations and shared directories when packing local work."""
    return (
        PurePosixPath(a.filename).parent == PurePosixPath(b.filename).parent
        or relation_key(a.filename, b.filename) in links
    )


def plan(context: ReviewContext) -> Plan:
    """Plan exact ownership without fetching, calling models or modifying the snapshot.

    Reject duplicate paths, oversized shared metadata or exhausted collection/work
    budgets. Small packets preserve the existing serialization and two-call path.
    """
    serialized = context.model_dump_json()
    if len(serialized.encode()) > MAX_SNAPSHOT_BYTES:
        raise ValueError("Snapshot exceeds 32 MB collection budget")
    if len({f.filename for f in context.files}) != len(context.files):
        raise ValueError("Duplicate changed-file path")
    if len(serialized.encode()) <= PACKET_BYTES:
        return Plan(snapshot=digest(serialized), whole=serialized, chunks=[])
    pending = [u for f in sorted(context.files, key=lambda f: f.filename) for u in units(f)]
    links = direct_relations(context)
    chunks: list[Chunk] = []
    # Start with the stable first unit, then prefer visible related evidence.
    while pending:
        chunk = Chunk(id=f"chunk-{len(chunks):04d}", units=[pending.pop(0)])
        if len(chunk.payload(context).encode()) > CHUNK_BYTES:
            raise ValueError("Shared metadata, replies or unit exceeds chunk budget")
        candidates = sorted(
            pending,
            key=lambda u: (
                not related(chunk.units[0], u, links),
                u.filename,
                u.patch_start,
                u.before_start,
            ),
        )
        for unit in candidates:
            trial = Chunk(id=chunk.id, units=[*chunk.units, unit])
            if len(trial.payload(context).encode()) <= CHUNK_BYTES:
                chunk = trial
                pending.remove(unit)
        chunks.append(chunk)
        if len(chunks) > MAX_CHUNKS:
            raise ValueError("Chunk work budget exceeded")
    if not chunks:
        raise ValueError("Shared metadata exceeds packet budget")
    return Plan(snapshot=digest(serialized), whole=None, chunks=chunks)


def validate_patch(file: ChangedFile) -> None:
    """Prove captured hunks account for the complete before/after text change.

    Validate old/new counts, coordinates, payload lines and unchanged gaps. This
    rejects truncated API patches instead of silently blessing omitted changes.
    File headers are ignored; only LF separates Git lines.
    """
    old, new = lines(file.before), lines(file.after)
    records = lines(file.patch)
    left = right = 0
    expected_left = expected_right = 0
    in_hunk = False
    for record in records:
        header = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", record)
        if header:
            if in_hunk and (left != expected_left or right != expected_right):
                raise ValueError("Incomplete captured patch hunk")
            a, ac, b, bc = header.groups()
            old_count, new_count = int(ac or 1), int(bc or 1)
            old_start = int(a) - int(old_count > 0)
            new_start = int(b) - int(new_count > 0)
            if (
                old_start < left
                or new_start < right
                or old_start > len(old)
                or new_start > len(new)
                or old[left:old_start] != new[right:new_start]
            ):
                raise ValueError("Captured patch omits changed sections")
            left, right = old_start, new_start
            expected_left, expected_right = left + old_count, right + new_count
            in_hunk = True
            continue
        if not in_hunk or record.startswith("\\"):
            continue
        sign = record[:1]
        if sign not in (" ", "+", "-"):
            raise ValueError("Unreadable captured patch record")
        text = record[1:].removesuffix("\n")
        if sign in (" ", "-"):
            if left >= len(old) or old[left].removesuffix("\n") != text:
                raise ValueError("Captured patch differs from old source")
            left += 1
        if sign in (" ", "+"):
            if right >= len(new) or new[right].removesuffix("\n") != text:
                raise ValueError("Captured patch differs from new source")
            right += 1
        if sign == " " and old[left - 1] != new[right - 1]:
            raise ValueError("Captured patch omits newline change")
    if left != expected_left or right != expected_right or old[left:] != new[right:]:
        raise ValueError("Captured patch is incomplete")
