"""Plan deterministic bounded reviews over an immutable changed-file snapshot.

Units own contiguous original patch records exactly once. Source excerpts retain
Git line coordinates; additional source can only come from the same snapshot.
No reviewed code is imported or executed. Coverage establishes work, not judgment.
"""

import ast
import hashlib
import json
import re
from bisect import bisect_right
from collections import defaultdict, deque
from itertools import combinations
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

    def payload(self, metadata: "PacketMetadata") -> str:
        """Serialize candidate evidence using snapshot metadata compiled once."""
        paths = {u.filename for u in self.units} | {u.previous_filename for u in self.units}
        indices = sorted(
            set(metadata.global_comments).union(
                *(metadata.comments_by_path.get(path, []) for path in paths)
            )
        )
        return encode(
            {
                "overview": metadata.overview,
                "scope": self.model_dump(),
                "previous_comments": [metadata.comments[i] for i in indices],
            }
        )


class PacketMetadata(Contract):
    """Cache inventory and indexed discussion for one immutable snapshot."""

    overview: dict[str, Any]
    comments: list[dict[str, Any]]
    comments_by_path: dict[str | None, list[int]]
    global_comments: list[int]


class Relation(Contract):
    """Persist deterministic file endpoints and captured reference evidence."""

    left: str
    right: str
    evidence: list[SourceRequest]


class Plan(Contract):
    """Describe whole or partitioned assignments with reusable snapshot evidence."""

    snapshot: str
    whole: str | None
    chunks: list[Chunk] = Field(max_length=MAX_CHUNKS)
    metadata: PacketMetadata
    relations: list[Relation]


def packet_metadata(context: ReviewContext) -> PacketMetadata:
    """Compile full-blob line counts and discussion membership once per plan."""
    changed = {f.filename for f in context.files}
    by_path: dict[str | None, list[int]] = defaultdict(list)
    global_comments = []
    for i, comment in enumerate(context.previous_comments):
        by_path[comment.path].append(i)
        if comment.path not in changed:
            global_comments.append(i)
    return PacketMetadata(
        overview=overview(context),
        comments=[c.model_dump() for c in context.previous_comments],
        comments_by_path=dict(by_path),
        global_comments=global_comments,
    )


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
    """Index all known literal references with one multi-pattern scan per version.

    A failure-linked trie preserves substring matches (including overlapping paths).
    Only the first occurrence per target/version contributes source evidence.
    Normalized source/test names are joined separately, without source scans.
    """
    files = sorted(context.files, key=lambda f: f.filename)
    targets: dict[str, set[str]] = defaultdict(set)
    names: dict[str, list[str]] = defaultdict(list)
    for file in files:
        names[PurePosixPath(file.filename).stem.removeprefix("test_").removesuffix("_test")].append(
            file.filename
        )
        for ref in (
            file.filename,
            str(PurePosixPath(file.filename).with_suffix("")).replace("/", "."),
            file.previous_filename,
        ):
            if ref:
                targets[ref].add(file.filename)
    # Failure links let each character advance the index without rescanning blobs.
    transitions: list[dict[str, int]] = [{}]
    failures = [0]
    outputs: list[list[str]] = [[]]
    for ref in sorted(targets):
        state = 0
        for char in ref:
            if char not in transitions[state]:
                transitions[state][char] = len(transitions)
                transitions.append({})
                failures.append(0)
                outputs.append([])
            state = transitions[state][char]
        outputs[state].append(ref)
    queue = deque(transitions[0].values())
    while queue:
        state = queue.popleft()
        for char, child in transitions[state].items():
            queue.append(child)
            fallback = failures[state]
            while fallback and char not in transitions[fallback]:
                fallback = failures[fallback]
            failures[child] = transitions[fallback].get(char, 0)
            outputs[child].extend(outputs[failures[child]])
    result: dict[tuple[str, str], list[SourceRequest]] = {}

    def add(left: str, right: str, evidence: SourceRequest | None = None) -> None:
        """Bound discovered relationships before expanding integration work."""
        key = relation_key(left, right)
        result.setdefault(key, [])
        if evidence is not None:
            result[key].append(evidence)
        if len(result) > 20_000:
            raise ValueError("Captured relationship budget exceeded")

    for group in names.values():
        for left, right in combinations(group, 2):
            add(left, right)
    for file in files:
        versions: list[tuple[Literal["LEFT", "RIGHT"], str]] = [
            ("LEFT", file.before),
            ("RIGHT", file.after),
        ]
        for side, text in versions:
            first: dict[str, int] = {}
            newlines = []
            state = 0
            for position, char in enumerate(text):
                if char == "\n":
                    newlines.append(position)
                while state and char not in transitions[state]:
                    state = failures[state]
                state = transitions[state].get(char, 0)
                for ref in outputs[state]:
                    for target in targets[ref]:
                        if target != file.filename:
                            start = position - len(ref) + 1
                            first[target] = min(first.get(target, start), start)
            total = len(newlines) + int(bool(text) and not text.endswith("\n"))
            for target, position in sorted(first.items()):
                line = bisect_right(newlines, position) + 1
                add(
                    file.filename,
                    target,
                    SourceRequest(
                        path=file.filename,
                        side=side,
                        start=max(1, line - 10),
                        end=min(total, line + 10),
                        reason="Captured direct path reference",
                    ),
                )
    return dict(sorted(result.items()))


def related(a: Unit, b: Unit, links: dict[tuple[str, str], list[SourceRequest]]) -> bool:
    """Prefer captured direct relations and shared directories when packing local work."""
    return (
        PurePosixPath(a.filename).parent == PurePosixPath(b.filename).parent
        or relation_key(a.filename, b.filename) in links
    )


def plan(context: ReviewContext, *, force_chunks: bool = False) -> Plan:
    """Plan exact ownership without fetching, calling models or modifying the snapshot.

    Reject duplicate paths, oversized shared metadata or exhausted collection/work
    budgets. Small packets preserve the existing serialization and two-call path.
    """
    serialized = context.model_dump_json()
    if len(serialized.encode()) > MAX_SNAPSHOT_BYTES:
        raise ValueError("Snapshot exceeds 32 MB collection budget")
    if len({f.filename for f in context.files}) != len(context.files):
        raise ValueError("Duplicate changed-file path")
    metadata = packet_metadata(context)
    if not force_chunks and len(serialized.encode()) <= PACKET_BYTES:
        return Plan(
            snapshot=digest(serialized),
            whole=serialized,
            chunks=[],
            metadata=metadata,
            relations=[],
        )
    pending = [u for f in sorted(context.files, key=lambda f: f.filename) for u in units(f)]
    links = direct_relations(context)
    chunks: list[Chunk] = []
    # Start with the stable first unit, then prefer visible related evidence.
    while pending:
        chunk = Chunk(id=f"chunk-{len(chunks):04d}", units=[pending.pop(0)])
        if len(chunk.payload(metadata).encode()) > (
            min(CHUNK_BYTES, 200_000) if force_chunks else CHUNK_BYTES
        ):
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
            if len(trial.payload(metadata).encode()) <= (
                min(CHUNK_BYTES, 200_000) if force_chunks else CHUNK_BYTES
            ):
                chunk = trial
                pending.remove(unit)
        chunks.append(chunk)
        if len(chunks) > MAX_CHUNKS:
            raise ValueError("Chunk work budget exceeded")
    if not chunks:
        raise ValueError("Shared metadata exceeds packet budget")
    return Plan(
        snapshot=digest(serialized),
        whole=None,
        chunks=chunks,
        metadata=metadata,
        relations=[Relation(left=a, right=b, evidence=e) for (a, b), e in links.items()],
    )


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
