"""Plan deterministic bounded reviews over an immutable changed-file snapshot.

Units own contiguous original patch records exactly once. Source excerpts retain
Git line coordinates; additional source can only come from the same snapshot.
Shared metadata locates Python declarations for source requests across all packets.
No reviewed code is imported or executed. Coverage establishes work, not judgment.
"""

import ast
from bisect import bisect_right
from collections import defaultdict, deque
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import Field

from ai_reviewer.contracts import ChangedFile, Contract, ReviewContext, SourceRequest
from ai_reviewer.limits import (
    CAPACITY_PACKET_BYTES,
    CHUNK_BYTES,
    MAX_CHUNKS,
    MAX_RELATIONS,
    MAX_SNAPSHOT_BYTES,
    PACKET_BYTES,
    UNIT_BYTES,
)
from ai_reviewer.patch import lines, walk_patch
from ai_reviewer.serialization import digest, encode


class Unit(Contract):
    """Own a half-open patch-record interval with supporting captured source.

    Attributes:
        id: Stable filename/interval identity, unique within a plan.
        filename: Current path; previous_filename is the old rename path or None.
        status: Captured GitHub change classification.
        before_mode: Old Git mode or None for an added file.
        after_mode: New Git mode or None for a removed file.
        patch_start: Zero-based first owned LF-delimited patch record.
        patch_end: Exclusive final owned record; source-only units may own none.
        patch: Exact concatenation of the owned records, not necessarily a full hunk.
        old_cursor: Git old-side coordinate at the fragment start.
        new_cursor: Git new-side coordinate at the fragment start.
        before_start: One-based line of the old excerpt's first physical line.
        before: Captured old text, which may overlap neighboring units' support.
        after_start: One-based line of the new excerpt's first physical line.
        after: Captured new text; only patch ownership must be disjoint.
    """

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
    """Group source units under one deterministic local or integration scope ID.

    Attributes:
        id: Planner-generated identifier also serialized inside the packet.
        units: Ordered evidence with exact section ownership; callers enforce budgets.
    """

    id: str
    units: list[Unit]

    def payload(self, metadata: "PacketMetadata") -> str:
        """Serialize this scope with its indexed discussion and shared snapshot metadata.

        Args:
            metadata: Precompiled data from the same snapshot; unchanged during packing.

        Returns:
            Compact Unicode JSON with overview, scope and original-order relevant comments.
            Global comments and comments matching current/previous unit paths appear once.
            Size is not enforced here; callers must apply their packet budget.
        """
        # Select relevant discussion by index without rescanning captured blobs.
        paths = {u.filename for u in self.units} | {u.previous_filename for u in self.units}
        indices = sorted(
            set(metadata.global_comments).union(
                *(metadata.comments_by_path.get(path, []) for path in paths)
            )
        )
        # Emit the exact shape whose array growth PacketSizer measures.
        return encode(
            {
                "overview": metadata.overview,
                "scope": self.model_dump(),
                "previous_comments": [metadata.comments[i] for i in indices],
            }
        )


class PacketMetadata(Contract):
    """Cache inventory and indexed discussion for one immutable snapshot.

    Attributes:
        overview: Revision/requirement metadata, file inventory and side-specific declarations.
        comments: Original-order discussion dictionaries, shared read-only by builders.
        comments_by_path: Current/old discussion path to indices in comments.
        global_comments: Indices whose path is not a current changed path; included everywhere.
    """

    overview: dict[str, Any]
    comments: list[dict[str, Any]]
    comments_by_path: dict[str | None, list[int]]
    global_comments: list[int]


class ArrayBudget:
    """Track exact compact-JSON array growth without serializing accepted evidence.

    Attributes:
        size: Current UTF-8 packet bytes, including fixed overhead and admitted items.
        sizes: Shared section/item byte lengths; callers must not mutate them.
        selected: Mutable sets of admitted item keys, owned by this budget only.
    """

    def __init__(self, size: int, sizes: dict[str, dict[str, int]]) -> None:
        """Start at size bytes for an empty-array packet with precomputed item sizes.

        Sizes maps section names to stable item keys and their complete JSON byte lengths.
        All tracked arrays must initially be empty. The mapping is shared, not copied.
        """
        self.size = size
        self.sizes = sizes
        self.selected: dict[str, set[str]] = {name: set() for name in sizes}

    def growth(self, additions: dict[str, set[str]]) -> int:
        """Return extra UTF-8 bytes for additions without changing this budget.

        Additions maps declared section names to item keys. Already-selected keys cost
        nothing; commas count only between array elements. Unknown sections or keys
        raise KeyError, so callers must predeclare the complete item-size inventory.
        """
        # Count only new members plus their separators in each independent array.
        total = 0
        for name, keys in additions.items():
            fresh = keys - self.selected[name]
            total += sum(self.sizes[name][key] for key in fresh)
            total += len(fresh) if self.selected[name] else max(0, len(fresh) - 1)
        return total

    def include(self, additions: dict[str, set[str]]) -> None:
        """Admit additions and update size/selected in place after the caller's fit check.

        Additions follows growth's section/key contract. No ceiling is enforced here;
        unknown keys raise KeyError before membership updates.
        """
        self.size += self.growth(additions)
        for name, keys in additions.items():
            self.selected[name].update(keys)


class PacketSizer:
    """Cache immutable evidence lengths for many incremental packet budgets.

    Attributes:
        metadata: Shared snapshot metadata; must remain unchanged after construction.
        sizes: UTF-8 JSON lengths indexed by array section and stable item key.
    """

    def __init__(
        self,
        metadata: PacketMetadata,
        units: list[Unit],
        sections: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Measure each unit, comment and optional extra array item once.

        Args:
            metadata: Immutable snapshot overview and indexed discussion.
            units: Candidate units with unique IDs for every scope this sizer will handle.
            sections: Optional extra array-name/item-key/value maps; names must not replace
                units or previous_comments. Values must be JSON serializable.

        Source values are not retained beyond metadata; only encoded lengths are cached.
        JSON serialization errors propagate for unsupported values.
        """
        # Cache source/comment sizes first, then declare integration-specific arrays.
        self.metadata = metadata
        self.sizes = {
            "units": {u.id: len(encode(u.model_dump()).encode()) for u in units},
            "previous_comments": {
                str(i): len(encode(comment).encode()) for i, comment in enumerate(metadata.comments)
            },
        }
        self.sizes.update(
            {
                name: {key: len(encode(value).encode()) for key, value in values.items()}
                for name, values in (sections or {}).items()
            }
        )

    def budget(self, id: str, extra: dict[str, Any] | None = None) -> ArrayBudget:
        """Return a fresh byte counter including fixed metadata and global comments.

        Args:
            id: Exact scope ID that will appear in the finished packet.
            extra: Optional fixed fields or empty extra arrays matching the final shape.
                Tracked arrays must stay empty here to avoid counting their contents twice.

        Returns:
            Independent mutable budget sharing this sizer's immutable item lengths.
        """
        empty = {
            "overview": self.metadata.overview,
            "scope": {"id": id, "units": []},
            **{name: [] for name in self.sizes if name != "units"},
            **(extra or {}),
        }
        # Global discussion is mandatory in every candidate, even an empty scope.
        budget = ArrayBudget(len(encode(empty).encode()), self.sizes)
        budget.include({"previous_comments": {str(i) for i in self.metadata.global_comments}})
        return budget

    def additions(self, scope: list[Unit]) -> dict[str, set[str]]:
        """Return unit and comment memberships for scope without mutating the sizer.

        Scope units must occur in the precomputed inventory. Current and previous paths
        select indexed comments; global comments were already admitted by budget().
        """
        paths = {u.filename for u in scope} | {u.previous_filename for u in scope}
        return {
            "units": {u.id for u in scope},
            "previous_comments": {
                str(i) for path in paths for i in self.metadata.comments_by_path.get(path, [])
            },
        }


class Relation(Contract):
    """Retain an undirected file relationship and its captured supporting ranges.

    Attributes:
        left: Lexically first current filename.
        right: Lexically second current filename.
        evidence: Inclusive captured-source requests; empty for name-only test/source joins.
    """

    left: str
    right: str
    evidence: list[SourceRequest]


class Plan(Contract):
    """Describe whole or partitioned work for a fixed captured snapshot.

    Attributes:
        snapshot: SHA-256 of the original context's UTF-8 JSON bytes.
        packet_bytes: Selected per-call byte ceiling; independent of token counting.
        whole: Byte-identical context JSON for small work, otherwise None.
        chunks: Ordered local scopes; empty when whole is present.
        metadata: Shared inventory/discussion indexes from the same context.
        relations: Deterministic file evidence reused during integration construction.

    Treat all fields as immutable after planning; the manifest hashes the entire plan.
    """

    snapshot: str
    packet_bytes: int
    whole: str | None
    chunks: list[Chunk] = Field(max_length=MAX_CHUNKS)
    metadata: PacketMetadata
    relations: list[Relation]


def packet_metadata(context: ReviewContext) -> PacketMetadata:
    """Return inventory and discussion indexes for context without mutating it.

    Discussion retains original order. Comments outside current changed paths are
    global, including comments attached to old rename paths. Overview creation scans
    captured blobs once for line counts and declaration navigation. No repository or
    provider reads occur.
    """
    # Index discussion once; non-current paths become mandatory global context.
    changed = {f.filename for f in context.files}
    by_path: dict[str | None, list[int]] = defaultdict(list)
    global_comments = []
    for i, comment in enumerate(context.previous_comments):
        by_path[comment.path].append(i)
        if comment.path not in changed:
            global_comments.append(i)
    # Cache the full inventory separately from packet-specific unit evidence.
    return PacketMetadata(
        overview=overview(context),
        comments=[c.model_dump() for c in context.previous_comments],
        comments_by_path=dict(by_path),
        global_comments=global_comments,
    )


def overview(context: ReviewContext) -> dict[str, Any]:
    """Return context revisions/requirements plus a filename-sorted source inventory.

    Files and prior discussion are excluded from the copied metadata. Inventory line
    counts use only LF boundaries and count a final unterminated line. Context remains
    unchanged; actual blob text is supplied separately in unit evidence. Declaration
    locations use current paths for both sides, even when the old path selected the parser.
    """
    # Keep revisions and requirements intact while replacing blobs with line counts.
    data = context.model_dump(exclude={"files", "previous_comments"})
    data["inventory"] = [
        {
            "filename": f.filename,
            "previous_filename": f.previous_filename,
            "status": f.status,
            "declarations": {
                "LEFT": declaration_index(f.previous_filename or f.filename, f.before),
                "RIGHT": declaration_index(f.filename, f.after),
            },
            "before_lines": f.before.count("\n")
            + int(bool(f.before) and not f.before.endswith("\n")),
            "after_lines": f.after.count("\n") + int(bool(f.after) and not f.after.endswith("\n")),
        }
        for f in sorted(context.files, key=lambda f: f.filename)
    ]
    return data


def declaration_index(filename: str, text: str) -> dict[str, Any]:
    """Locate Python classes/functions without executing captured text.

    Args:
        filename: Version's path, used only to select the Python parser.
        text: Captured blob; never imported or evaluated.

    Returns:
        Status and source-ordered entries with qualified names and one-based inclusive
        LF ranges, including decorators. Empty valid Python is indexed with no entries.
        Unsupported syntax/languages or bare-CR coordinates have explicit statuses and
        no entries; callers retain physical-line source requests as their fallback.
    """
    # Never guess syntax or coordinates that the parser cannot represent faithfully.
    if not filename.endswith(".py"):
        return {"status": "unsupported_language", "entries": []}
    if "\r" in text.replace("\r\n", "\n"):
        return {"status": "unsupported_line_endings", "entries": []}
    try:
        tree = ast.parse(text)
    except SyntaxError, ValueError, RecursionError:
        return {"status": "parse_error", "entries": []}
    # Iterative traversal retains lexical owners without a second recursion limit.
    entries = []
    pending: list[tuple[ast.AST, str]] = [(tree, "")]
    while pending:
        node, owner = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            owner = f"{owner}.{node.name}" if owner else node.name
            entries.append(
                {
                    "name": owner,
                    "start": min([node.lineno, *(d.lineno for d in node.decorator_list)]),
                    "end": node.end_lineno or node.lineno,
                }
            )
        pending.extend((child, owner) for child in ast.iter_child_nodes(node))
    return {"status": "indexed", "entries": sorted(entries, key=lambda entry: entry["start"])}


def declarations(filename: str, text: str) -> list[tuple[int, int]]:
    """Return sorted one-based inclusive Python declaration ranges without imports.

    Args:
        filename: Path used only to select the .py parser.
        text: Captured source to parse, never execute.

    Returns:
        Function, async-function and class ranges, including nested declarations.
        Unsupported paths or syntax/parser failures return no boundaries, allowing
        the caller to fall back to diff and physical-line splitting.
    """
    # Unsupported or unparsable source uses the caller's non-syntax splitting fallback.
    if not filename.endswith(".py"):
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError, ValueError, RecursionError:
        return []
    # Include nested declarations without importing or executing the source.
    return sorted(
        (node.lineno, node.end_lineno or node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


def units(file: ChangedFile, unit_bytes: int = UNIT_BYTES) -> list[Unit]:
    """Split one captured file into bounded units with exact patch-record ownership.

    Args:
        file: Complete captured patch and before/after blobs, already validated.
        unit_bytes: Maximum UTF-8 bytes for each unit's model_dump_json serialization.
            Defaults to UNIT_BYTES; enclosing packet overhead is budgeted separately.

    Returns:
        Deterministic units covering all patch records, including headers and newline
        markers. Supporting excerpts may overlap. Empty-patch changes retain all source
        through source-only slices. Inputs remain unchanged and no code is executed.

    Raises:
        ValueError: An indivisible physical source/patch line plus metadata cannot fit.
    """
    # Compile physical records, declaration boundaries and canonical fragment cursors.
    records = lines(file.patch)
    old, new = lines(file.before), lines(file.after)
    old_defs, new_defs = (
        declarations(file.filename, file.before),
        declarations(file.filename, file.after),
    )
    cursors: list[tuple[int, int]] = []
    boundaries = []
    ends: list[tuple[int, int]] = []
    definition_starts = {start for start, _ in new_defs}
    for index, record in enumerate(walk_patch(file.patch)):
        if record.counts is not None or record.right in definition_starts:
            boundaries.append(index)
        cursors.append((record.left, record.right))
        ends.append(record.end)
    cursors.append(ends[-1] if ends else (1, 1))

    def excerpt(
        source: list[str], start: int, end: int, definitions: list[tuple[int, int]], expand: bool
    ) -> tuple[int, str]:
        """Return a one-based start and LF-preserving source around inclusive coordinates.

        Source is an LF-split captured version. Start/end receive up to 20 surrounding
        lines. If expand is true, overlapping inclusive declaration ranges in definitions
        also extend the excerpt; available source bounds clamp the resulting slice.
        """
        # Start with local context, then optionally widen to enclosing declarations.
        lo, hi = max(1, start - 20), min(len(source), end + 20)
        if expand:
            for a, b in definitions:
                if a <= start <= b or a <= end <= b:
                    lo, hi = min(lo, a), max(hi, b)
        return lo, "".join(source[lo - 1 : hi])

    def make(start: int, end: int, whole: bool = False, expand: bool = False) -> Unit:
        """Build the half-open patch-record interval [start, end) from captured data.

        Whole selects full blobs; otherwise expand optionally includes enclosing
        source declarations. Return a unit with stable identity and absolute cursors.
        This helper does not enforce size; callers decide whether to accept or split it.
        """
        # Bind record ownership to exact old/new source coordinates before serializing.
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

    # Prefer complete files; splitting is needed only when their serialized unit is too large.
    whole = make(0, len(records), whole=True)
    if len(whole.model_dump_json().encode()) <= unit_bytes:
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
                if len(unit.model_dump_json().encode()) <= unit_bytes:
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
        if len(candidate.model_dump_json().encode()) > unit_bytes:
            candidate = make(start, end)
        if len(candidate.model_dump_json().encode()) <= unit_bytes:
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
    """Return left/right filenames in lexical order as an undirected relationship key."""
    return (left, right) if left < right else (right, left)


def direct_relations(context: ReviewContext) -> dict[tuple[str, str], list[SourceRequest]]:
    """Discover bounded literal-reference and normalized source/test relationships.

    Args:
        context: Complete changed-file snapshot, including old rename paths and both versions.

    Returns:
        Sorted undirected path-pair keys with captured reference ranges. A failure-linked
        trie scans each version once; only the first occurrence per target/version adds
        evidence. Test/source stem joins carry no range evidence. Inputs are unchanged.

    Raises:
        ValueError: Distinct relationship pairs exceed MAX_RELATIONS.

    Matching is lexical evidence, not import resolution or a complete dependency graph.
    """
    # Index current paths, old rename paths and normalized test/source stems.
    files = sorted(context.files, key=lambda f: f.filename)
    targets: dict[str, set[str]] = defaultdict(set)
    names: dict[str, dict[bool, list[str]]] = defaultdict(lambda: {False: [], True: []})
    for file in files:
        stem = PurePosixPath(file.filename).stem
        normalized = stem.removeprefix("test_").removesuffix("_test")
        names[normalized][normalized != stem].append(file.filename)
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
        """Record a normalized left/right file pair and optional evidence in result.

        Evidence is appended without deduplication. Name-only pairs use None. Raise
        ValueError when distinct pairs exceed MAX_RELATIONS, before integration expansion.
        """
        key = relation_key(left, right)
        result.setdefault(key, [])
        if evidence is not None:
            result[key].append(evidence)
        if len(result) > MAX_RELATIONS:
            raise ValueError("Captured relationship budget exceeded")

    # Name-only joins need no source scans; literal references retain captured ranges.
    for group in names.values():
        for left in group[False]:
            for right in group[True]:
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
                    start = position - len(ref) + 1
                    # Match complete path/module tokens, never a word inside another name.
                    if (start and (text[start - 1].isalnum() or text[start - 1] in "_./")) or (
                        position + 1 < len(text)
                        and (text[position + 1].isalnum() or text[position + 1] in "_./")
                    ):
                        continue
                    for target in targets[ref]:
                        if target != file.filename:
                            first[target] = min(first.get(target, start), start)
            total = len(newlines) + int(bool(text) and not text.endswith("\n"))
            # Translate first textual matches to bounded, one-based source requests.
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
    """Return whether units a/b share a directory or an indexed undirected file link.

    Links is the captured relationship map; this packing preference neither resolves
    imports nor proves compatibility. Unit and evidence inputs are not mutated.
    """
    return (
        PurePosixPath(a.filename).parent == PurePosixPath(b.filename).parent
        or relation_key(a.filename, b.filename) in links
    )


def plan(context: ReviewContext, *, packet_bytes: int = PACKET_BYTES) -> Plan:
    """Plan exact ownership without fetching, calling models or changing context.

    Args:
        context: Complete immutable snapshot with unique current changed paths.
        packet_bytes: PACKET_BYTES by default, or CAPACITY_PACKET_BYTES to force smaller
            chunks. Other ceilings are rejected; bytes never substitute for token counts.

    Returns:
        Deterministic plan. A small default-budget snapshot retains exact context JSON
        in whole; chunked plans carry reusable metadata and relationship evidence.

    Raises:
        ValueError: Ceiling is unknown, paths repeat, snapshot/relationship/work limits
            are exceeded, or required metadata/source cannot fit a bounded unit/chunk.
    """
    # Reject unsupported policy and incomplete snapshot identity before choosing a route.
    if packet_bytes not in (PACKET_BYTES, CAPACITY_PACKET_BYTES):
        raise ValueError("Unknown plan packet budget")
    serialized = context.model_dump_json()
    if len(serialized.encode()) > MAX_SNAPSHOT_BYTES:
        raise ValueError("Snapshot exceeds 32 MB collection budget")
    if len({f.filename for f in context.files}) != len(context.files):
        raise ValueError("Duplicate changed-file path")
    # Keep the whole-context serialization unchanged when the default ceiling permits it.
    metadata = packet_metadata(context)
    if packet_bytes == PACKET_BYTES and len(serialized.encode()) <= packet_bytes:
        return Plan(
            snapshot=digest(serialized),
            whole=serialized,
            packet_bytes=packet_bytes,
            chunks=[],
            metadata=metadata,
            relations=[],
        )
    # Reserve space for two source endpoints, their reports and reference evidence.
    unit_bytes = min(UNIT_BYTES, packet_bytes // 4) if packet_bytes < PACKET_BYTES else UNIT_BYTES
    pending = [
        u for f in sorted(context.files, key=lambda f: f.filename) for u in units(f, unit_bytes)
    ]
    links = direct_relations(context)
    chunks: list[Chunk] = []
    sizer = PacketSizer(metadata, pending)
    target_bytes = (
        min(CHUNK_BYTES, packet_bytes // 2) if packet_bytes < PACKET_BYTES else CHUNK_BYTES
    )
    # Start with the stable first unit, then prefer visible related evidence.
    while pending:
        chunk = Chunk(id=f"chunk-{len(chunks):04d}", units=[pending.pop(0)])
        budget = sizer.budget(chunk.id)
        budget.include(sizer.additions(chunk.units))
        if budget.size > target_bytes:
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
        smallest = min((sizer.sizes["units"][u.id] for u in candidates), default=0)
        accepted = set()
        for unit in candidates:
            if budget.size + smallest + 1 > target_bytes:
                break
            additions = sizer.additions([unit])
            if budget.size + budget.growth(additions) <= target_bytes:
                budget.include(additions)
                chunk.units.append(unit)
                accepted.add(unit.id)
        pending = [u for u in pending if u.id not in accepted]
        chunks.append(chunk)
        if len(chunks) > MAX_CHUNKS:
            raise ValueError("Chunk work budget exceeded")
    if not chunks:
        raise ValueError("Shared metadata exceeds packet budget")
    return Plan(
        snapshot=digest(serialized),
        whole=None,
        packet_bytes=packet_bytes,
        chunks=chunks,
        metadata=metadata,
        relations=[Relation(left=a, right=b, evidence=e) for (a, b), e in links.items()],
    )


def validate_patch(file: ChangedFile) -> None:
    """Require captured hunks to account for the complete before/after text change.

    Args:
        file: Captured versions and patch to validate without mutation or I/O.

    Raises:
        ValueError: Counts/coordinates disagree, source text differs, unchanged gaps or
            trailing text hide changes, records are unreadable, or the patch is truncated.

    Headers are ignored and only LF separates Git lines. This verifies patch evidence,
    not syntax or semantic correctness of the reviewed code.
    """
    # Track consumed source and declared hunk ends using Git's LF interpretation.
    old, new = lines(file.before), lines(file.after)
    left = right = 0
    expected_left = expected_right = 0
    in_hunk = False
    for record in walk_patch(file.patch):
        # A new hunk must follow a complete hunk and an unchanged source gap.
        if record.counts is not None:
            if in_hunk and (left != expected_left or right != expected_right):
                raise ValueError("Incomplete captured patch hunk")
            old_count, new_count = record.counts
            old_start = record.left - int(old_count > 0)
            new_start = record.right - int(new_count > 0)
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
        if not in_hunk or record.text.startswith("\\"):
            continue
        # Every consuming record must agree with its captured old/new text.
        sign = record.text[:1]
        if sign not in (" ", "+", "-"):
            raise ValueError("Unreadable captured patch record")
        text = record.text[1:].removesuffix("\n")
        consumes_old, consumes_new = record.consumes
        if consumes_old:
            if left >= len(old) or old[left].removesuffix("\n") != text:
                raise ValueError("Captured patch differs from old source")
            left += 1
        if consumes_new:
            if right >= len(new) or new[right].removesuffix("\n") != text:
                raise ValueError("Captured patch differs from new source")
            right += 1
        if sign == " " and old[left - 1] != new[right - 1]:
            raise ValueError("Captured patch omits newline change")
    # Declared counts and the untouched tail must agree after the last record.
    if left != expected_left or right != expected_right or old[left:] != new[right:]:
        raise ValueError("Captured patch is incomplete")
