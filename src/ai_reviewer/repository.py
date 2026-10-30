"""Capture immutable tracked source and expose bounded read-only reviewer tools.

GitHub commit trees and blobs become content-addressed artifacts during preparation.
Provider calls receive indexed source and bounded original PR patch records;
model arguments never become local paths or shell commands.
"""

import base64
import hashlib
import re
import time
from itertools import islice
from pathlib import Path
from typing import Any, Literal

from ai_reviewer.contracts import Contract, ReviewContext
from ai_reviewer.github import GitHub
from ai_reviewer.limits import (
    MAX_REPOSITORY_BLOB_BYTES,
    MAX_REPOSITORY_BLOB_REQUESTS,
    MAX_REPOSITORY_BYTES,
    MAX_REPOSITORY_FILES,
    REPOSITORY_CONTEXT_TOKENS,
    REPOSITORY_PAGE_SIZE,
    REPOSITORY_RANGE_LINES,
    REPOSITORY_RESERVED_REQUESTS,
    REPOSITORY_SEARCH_FILES,
    REPOSITORY_SEARCH_LINE_CHARS,
    REPOSITORY_SEARCH_MATCHES,
    REPOSITORY_SEARCH_QUERY_CHARS,
    REPOSITORY_TOOL_RESULT_BYTES,
    REPOSITORY_TOOL_SECONDS,
    REPOSITORY_TOOL_TOTAL_BYTES,
    REPOSITORY_TOOL_VISIBLE_RESULT_BYTES,
)
from ai_reviewer.patch import lines, walk_patch
from ai_reviewer.serialization import digest, encode

Revision = Literal["head", "comparison", "base"]


class SnapshotEntry(Contract):
    """Describe one tracked file without embedding source into review prompts.

    Attributes:
        path: Git repository-relative path in POSIX form.
        mode: Git file mode; symlinks and submodules are listed but never traversed.
        blob: Content-addressed Git blob SHA, or None for submodules.
        size: Raw Git blob size in bytes, or zero when the tree has no blob.
        status: Text availability, including stable binary and size diagnostics.
    """

    path: str
    mode: str
    blob: str | None
    size: int
    status: Literal["text", "binary", "oversized", "unavailable", "symlink", "submodule"]


class TreeSnapshot(Contract):
    """Bind a sorted tracked-file inventory to one exact commit and root tree.

    Attributes:
        commit: Full immutable commit SHA.
        tree: Git root tree SHA for that commit.
        entries: Complete sorted entries; text bodies live in the blob store.
        inventory_digest: SHA-256 over revision identity and canonical entries.
    """

    commit: str
    tree: str
    entries: list[SnapshotEntry]
    inventory_digest: str


class SnapshotIndex(Contract):
    """Bind all comparison sides and their content-addressed artifact inventory.

    Attributes:
        repo: Owner/repository pair used only for provenance.
        snapshots: Trees keyed by fixed head, comparison and base revision names.
        digest: SHA-256 over repository identity and every immutable tree digest.
    """

    repo: str
    snapshots: dict[Revision, TreeSnapshot]
    digest: str


class RepositorySnapshotError(ValueError):
    """Reject incomplete Git trees, damaged snapshot artifacts or budget exhaustion."""


def _tree_snapshot(gh: GitHub, repo: str, commit: str) -> tuple[str, list[dict[str, Any]]]:
    """Fetch one complete recursive Git tree using only a pinned commit identity.

    Args:
        gh: Prepare-stage authenticated GitHub client.
        repo: Validated owner/repository path.
        commit: Exact commit SHA from the captured pull request.

    Returns:
        Root tree SHA and sorted blob/gitlink entries.

    Raises:
        RepositorySnapshotError: The tree is truncated or exceeds the file limit.
        RuntimeError: GitHub rejects the commit/tree request.
    """
    root = gh.request("GET", f"/repos/{repo}/git/commits/{commit}")["tree"]["sha"]
    result = gh.request("GET", f"/repos/{repo}/git/trees/{root}", params={"recursive": "1"})
    if result.get("truncated"):
        raise RepositorySnapshotError("Repository Git tree is truncated")
    entries = sorted(
        (item for item in result["tree"] if item["type"] in ("blob", "commit")),
        key=lambda item: item["path"],
    )
    if len(entries) > MAX_REPOSITORY_FILES:
        raise RepositorySnapshotError("Repository tracked-file limit exceeded")
    return root, entries


def _git_blob_sha(raw: bytes) -> str:
    """Return Git's SHA-1 identity for raw blob bytes, including its object header."""
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def _read_repository_blob(gh: GitHub, repo: str, sha: str, size: int) -> bytes:
    """Read and verify one bounded text candidate by Git blob identity.

    Args:
        gh: Prepare-stage authenticated GitHub client.
        repo: Validated owner/repository path.
        sha: Expected Git blob SHA.
        size: Expected raw byte size from the tree entry.

    Returns:
        Exact raw UTF-8 candidate bytes.

    Raises:
        RepositorySnapshotError: GitHub returns a different or oversized blob.
        RuntimeError: GitHub rejects the blob request.
    """
    if size > MAX_REPOSITORY_BLOB_BYTES:
        raise RepositorySnapshotError("Repository blob exceeds per-file limit")
    blob = gh.request("GET", f"/repos/{repo}/git/blobs/{sha}")
    if blob.get("encoding") != "base64" or blob.get("size") != size:
        raise RepositorySnapshotError("Repository blob metadata does not match its tree")
    encoded = re.sub(r"[\t\n\r ]", "", blob["content"])
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != size or _git_blob_sha(raw) != sha:
        raise RepositorySnapshotError("Repository blob content does not match its Git SHA")
    return raw


def capture_repository(gh: GitHub, context: ReviewContext, directory: Path) -> SnapshotIndex:
    """Capture comparison, head and target-base source into a private artifact directory.

    Args:
        gh: Prepare-stage GitHub client; its credentials never leave this function.
        context: Already validated pull request revisions and repository identity.
        directory: Fresh path under the run's artifact directory.

    Returns:
        Snapshot index whose digest binds every revision, tree and entry.

    Raises:
        RepositorySnapshotError: A tracked tree is incomplete or exceeds its file limit.
        OSError: The run artifact cannot be written.
        RuntimeError: GitHub rejects a required immutable tree or blob read.
    """
    directory.mkdir(parents=True, exist_ok=True)
    blob_dir = directory / "blobs"
    blob_dir.mkdir(exist_ok=True)
    revisions: dict[Revision, str] = {
        "head": context.head,
        "comparison": context.diff_base,
        "base": context.base,
    }
    trees: dict[Revision, TreeSnapshot] = {}
    trees_by_commit: dict[str, TreeSnapshot] = {}
    cached_status: dict[str, Literal["text", "binary", "unavailable"]] = {}
    total_bytes = 0
    blob_requests = 0
    request_budget = min(
        MAX_REPOSITORY_BLOB_REQUESTS,
        max(0, gh.requests_remaining - REPOSITORY_RESERVED_REQUESTS),
    )

    # Capture each unique revision once; source bytes are deduplicated by blob SHA.
    for side, commit in revisions.items():
        if commit in trees_by_commit:
            trees[side] = trees_by_commit[commit]
            continue
        tree_sha, raw_entries = _tree_snapshot(gh, context.repo, commit)
        entries = []
        for raw_entry in raw_entries:
            mode = raw_entry["mode"]
            blob = raw_entry["sha"] if raw_entry["type"] == "blob" else None
            size = raw_entry.get("size", 0)
            status: Literal["text", "binary", "oversized", "unavailable", "symlink", "submodule"]
            if mode == "120000":
                status = "symlink"
            elif mode == "160000":
                status = "submodule"
            elif size > MAX_REPOSITORY_BLOB_BYTES:
                status = "oversized"
            elif cached_status.get(blob or "") is not None:
                status = cached_status[blob or ""]
            elif blob_requests >= request_budget or total_bytes + size > MAX_REPOSITORY_BYTES:
                status = "unavailable"
                cached_status[blob or ""] = status
            else:
                blob_requests += 1
                raw = _read_repository_blob(gh, context.repo, blob or "", size)
                try:
                    decoded = raw.decode("utf-8")
                    status = "binary" if "\0" in decoded else "text"
                except UnicodeDecodeError:
                    status = "binary"
                total_bytes += len(raw)
                if status == "text":
                    blob_path = blob_dir / (blob or "")
                    if not blob_path.exists():
                        blob_path.write_bytes(raw)
                cached_status[blob or ""] = status
            entries.append(
                SnapshotEntry(
                    path=raw_entry["path"], mode=mode, blob=blob, size=size, status=status
                )
            )
        identity = {
            "commit": commit,
            "tree": tree_sha,
            "entries": [entry.model_dump() for entry in entries],
        }
        trees[side] = TreeSnapshot(
            commit=commit,
            tree=tree_sha,
            entries=entries,
            inventory_digest=digest(encode(identity)),
        )
        trees_by_commit[commit] = trees[side]
    body = {
        "repo": context.repo,
        "snapshots": {side: tree.model_dump() for side, tree in trees.items()},
    }
    index = SnapshotIndex(repo=context.repo, snapshots=trees, digest=digest(encode(body)))
    temporary = directory / "index.tmp"
    temporary.write_text(index.model_dump_json())
    temporary.replace(directory / "index.json")
    return index


def load_repository(context: ReviewContext, directory: Path) -> "RepositoryTools":
    """Load and verify snapshot metadata against the exact revisions in ReviewContext.

    Args:
        context: Captured PR context containing repository_digest.
        directory: Prepare-stage snapshot artifact directory.

    Returns:
        Read-only tool executor over its verified indexed snapshots.

    Raises:
        RepositorySnapshotError: Context identity, index digest or local artifact shape differs.
        OSError: Index artifact cannot be read.
    """
    index = SnapshotIndex.model_validate_json((directory / "index.json").read_text())
    expected = {"head": context.head, "comparison": context.diff_base, "base": context.base}
    if context.repository_digest != index.digest or index.repo != context.repo:
        raise RepositorySnapshotError("Repository snapshot digest does not match review context")
    if {side: item.commit for side, item in index.snapshots.items()} != expected:
        raise RepositorySnapshotError("Repository snapshot revisions do not match review context")
    body = {
        "repo": index.repo,
        "snapshots": {side: tree.model_dump() for side, tree in index.snapshots.items()},
    }
    if digest(encode(body)) != index.digest:
        raise RepositorySnapshotError("Repository snapshot index digest is invalid")
    for snapshot in index.snapshots.values():
        identity = {
            "commit": snapshot.commit,
            "tree": snapshot.tree,
            "entries": [entry.model_dump() for entry in snapshot.entries],
        }
        if digest(encode(identity)) != snapshot.inventory_digest:
            raise RepositorySnapshotError("Repository tree inventory digest is invalid")
    return RepositoryTools(index, directory / "blobs", context=context)


def _valid_path(value: str, *, allow_empty: bool = False) -> bool:
    """Accept canonical Git paths without absolute, dot, parent or NUL components."""
    if not value and allow_empty:
        return True
    parts = value.split("/")
    if allow_empty and parts[-1] == "":
        parts.pop()
    return (
        bool(value)
        and not value.startswith("/")
        and "\0" not in value
        and all(part not in ("", ".", "..") for part in parts)
    )


class RepositoryTools:
    """Serve bounded snapshot reads and append replayable evidence to one attempt log.

    Attributes:
        index: Immutable repository/revision identities and complete file inventories.
        blob_dir: Internal content-addressed artifact directory; model paths never enter it.
        event_path: Attempt-local append-only request/result provenance file.
        context: Captured changed-file patches, bound to the same revisions by load_repository.
    """

    def __init__(
        self,
        index: SnapshotIndex,
        blob_dir: Path,
        event_path: Path | None = None,
        *,
        entries: dict[Revision, dict[str, SnapshotEntry]] | None = None,
        context: ReviewContext,
    ) -> None:
        """Bind verified snapshots and their captured PR context to a worker-owned event log.

        Production callers use load_repository to verify context/index identity first.
        Shared entries and context are read-only for the lifetime of all attempts.
        """
        self.context = context
        self.index = index
        self.blob_dir = blob_dir
        self.event_path = event_path
        self.calls = 0
        self.result_bytes = 0
        self.context_tokens = 0
        self.started = time.monotonic()
        self.entries = entries or {
            side: {entry.path: entry for entry in tree.entries}
            for side, tree in index.snapshots.items()
        }

    def for_attempt(self, event_path: Path) -> "RepositoryTools":
        """Create fresh accounting and an isolated event log over the same snapshots."""
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.touch(exist_ok=False)
        return RepositoryTools(
            self.index, self.blob_dir, event_path, entries=self.entries, context=self.context
        )

    def account_tokens(self, count: int) -> None:
        """Charge a counted provider request against the assignment context-token ceiling."""
        self.check_time()
        self.context_tokens += count
        if self.context_tokens > REPOSITORY_CONTEXT_TOKENS:
            self.exhaust("context_tokens", REPOSITORY_CONTEXT_TOKENS)

    def check_time(self) -> None:
        """Stop between provider or tool requests when the assignment deadline expires."""
        if time.monotonic() - self.started >= REPOSITORY_TOOL_SECONDS:
            self.exhaust("seconds", REPOSITORY_TOOL_SECONDS)

    def exhaust(self, reason: str, limit: int) -> None:
        """Persist a source-free budget diagnostic, then fail this required assignment."""
        if self.event_path:
            (self.event_path.parent / "exhaustion.json").write_text(
                encode(
                    {
                        "reason": reason,
                        "limit": limit,
                        "calls": self.calls,
                        "result_bytes": self.result_bytes,
                        "context_tokens": self.context_tokens,
                    }
                )
            )
        raise RepositorySnapshotError(f"Repository tool {reason} budget exhausted")

    def invoke(self, name: str, arguments: dict[str, Any], call_id: str = "") -> str:
        """Validate, execute and record one bounded source or PR-diff call.

        Args:
            name: One of the fixed repository source/diff tool names.
            arguments: Provider-decoded argument object, validated again at this boundary.
            call_id: Provider call identity retained in the attempt transcript.

        Returns:
            Compact bounded JSON evidence or a safe recoverable argument diagnostic.

        Raises:
            RepositorySnapshotError: Result-byte or wall-clock budget is exhausted;
                artifact content is missing or fails its Git SHA verification.
        """
        self.check_time()
        self.calls += 1
        if not isinstance(arguments, dict):
            return self._record(name, {}, {"error": "arguments must be an object"}, call_id)
        try:
            result = self._execute(name, arguments)
        except RepositorySnapshotError:
            raise
        except (ValueError, KeyError, TypeError) as error:
            result = {"error": str(error)}
        return self._record(name, arguments, result, call_id)

    def _record(
        self, name: str, arguments: dict[str, Any], result: dict[str, Any], call_id: str
    ) -> str:
        """Bound one JSON result, account its bytes, and persist the exact transcript row."""
        serialized = encode(result)
        size = len(serialized.encode())
        if size > REPOSITORY_TOOL_RESULT_BYTES:
            self.exhaust("result_bytes_per_call", REPOSITORY_TOOL_RESULT_BYTES)
        self.result_bytes += size
        if self.result_bytes > REPOSITORY_TOOL_TOTAL_BYTES:
            self.exhaust("result_bytes", REPOSITORY_TOOL_TOTAL_BYTES)
        row = {
            "sequence": self.calls - 1,
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "result": result,
            "result_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "result_bytes": size,
            "snapshot_digest": self.index.digest,
        }
        if self.event_path:
            with self.event_path.open("a") as stream:
                stream.write(encode(row) + "\n")
        return serialized

    def _execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run a tool after enforcing its exact shape, revision and source boundary."""
        if name == "read_pr_diff":
            return self._read_pr_diff(arguments)
        revision = arguments.get("revision")
        if revision not in self.entries:
            raise ValueError("revision must be head, comparison or base")
        entries = self.entries[revision]
        common = {"revision", "path_prefix", "cursor", "limit"}
        if name == "list_repository_files":
            if set(arguments) != common:
                raise ValueError("invalid listing arguments")
            prefix, cursor, limit = (
                arguments["path_prefix"],
                arguments["cursor"],
                arguments["limit"],
            )
            if not isinstance(prefix, str) or not _valid_path(prefix, allow_empty=True):
                raise ValueError("invalid repository path prefix")
            if not isinstance(cursor, str) or not cursor.isdecimal():
                raise ValueError("cursor must be a decimal offset")
            if type(limit) is not int or not 1 <= limit <= REPOSITORY_PAGE_SIZE:
                raise ValueError("limit must be within the repository page bound")
            paths = [p for p in sorted(entries) if p.startswith(prefix)]
            offset = int(cursor)
            page: list[dict[str, Any]] = []
            for path in paths[offset : offset + limit]:
                candidate = [*page, entries[path].model_dump()]
                if len(encode(candidate).encode()) > REPOSITORY_TOOL_VISIBLE_RESULT_BYTES:
                    break
                page.append(entries[path].model_dump())
            if not page and offset < len(paths):
                raise RepositorySnapshotError("A tracked path exceeds the listing result bound")
            next_offset = offset + len(page)
            next_cursor = str(next_offset) if next_offset < len(paths) else ""
            return {
                "revision": revision,
                "commit": self.index.snapshots[revision].commit,
                "files": page,
                "next_cursor": next_cursor,
                "truncated": bool(next_cursor),
            }
        if name == "search_repository":
            if set(arguments) != common | {"query"}:
                raise ValueError("invalid search arguments")
            prefix, cursor, limit, query = (
                arguments["path_prefix"],
                arguments["cursor"],
                arguments["limit"],
                arguments["query"],
            )
            if (
                not isinstance(query, str)
                or not query
                or len(query) > REPOSITORY_SEARCH_QUERY_CHARS
            ):
                raise ValueError(
                    f"query must contain 1 to {REPOSITORY_SEARCH_QUERY_CHARS} literal characters"
                )
            if not isinstance(prefix, str) or not _valid_path(prefix, allow_empty=True):
                raise ValueError("invalid repository path prefix")
            if not isinstance(cursor, str) or not re.fullmatch(r"[0-9]+:[0-9]+", cursor):
                raise ValueError("cursor must be a file:line offset")
            if type(limit) is not int or not 1 <= limit <= REPOSITORY_SEARCH_MATCHES:
                raise ValueError("limit must be within the search page bound")
            paths = [p for p in sorted(entries) if p.startswith(prefix)]
            offset, line_offset = (int(value) for value in cursor.split(":"))
            matches: list[dict[str, Any]] = []
            unavailable = 0
            scanned = 0
            needle = query.casefold()
            next_cursor = ""
            result_bound = REPOSITORY_TOOL_VISIBLE_RESULT_BYTES
            for file_index in range(offset, len(paths)):
                path = paths[file_index]
                if len(matches) >= limit or scanned >= REPOSITORY_SEARCH_FILES:
                    next_cursor = f"{file_index}:0"
                    break
                scanned += 1
                entry = entries[path]
                if entry.status != "text":
                    unavailable += 1
                    continue
                text = self._read_text(entry)
                records = lines(text)
                for line_index in range(line_offset if file_index == offset else 0, len(records)):
                    line = records[line_index]
                    if needle in line.casefold():
                        match = {
                            "path": path,
                            "line": line_index + 1,
                            "text": line.rstrip("\n")[:REPOSITORY_SEARCH_LINE_CHARS],
                            "line_truncated": len(line.rstrip("\n")) > REPOSITORY_SEARCH_LINE_CHARS,
                        }
                        if len(encode([*matches, match]).encode()) > result_bound:
                            next_cursor = f"{file_index}:{line_index}"
                            break
                        matches.append(match)
                        if len(matches) >= limit:
                            next_cursor = (
                                f"{file_index}:{line_index + 1}"
                                if line_index + 1 < len(records)
                                else f"{file_index + 1}:0"
                            )
                            break
                if next_cursor:
                    break
                line_offset = 0
            more = bool(next_cursor) and next_cursor.split(":", 1)[0] != str(len(paths))
            return {
                "revision": revision,
                "commit": self.index.snapshots[revision].commit,
                "matches": matches,
                "scanned_files": scanned,
                "unavailable_files": unavailable,
                "next_cursor": next_cursor if more else "",
                "truncated": more,
            }
        if name == "read_repository_lines":
            if set(arguments) != {"revision", "path", "start", "end"}:
                raise ValueError("invalid line-read arguments")
            path, start, end = arguments["path"], arguments["start"], arguments["end"]
            if not isinstance(path, str) or not _valid_path(path):
                raise ValueError("invalid repository path")
            if type(start) is not int or type(end) is not int or start < 1 or end < start:
                raise ValueError("line range must use positive inclusive coordinates")
            if end - start + 1 > REPOSITORY_RANGE_LINES:
                raise ValueError("line range exceeds the repository read bound")
            found = entries.get(path)
            if found is None:
                return {"revision": revision, "path": path, "diagnostic": "missing from snapshot"}
            if found.status != "text":
                return {"revision": revision, "path": path, "diagnostic": found.status}
            source = lines(self._read_text(found))
            if start > len(source):
                return {"revision": revision, "path": path, "diagnostic": "line range unavailable"}
            # Page whole lines by serialized bytes, including JSON escaping.
            selected = ""
            actual_end = start - 1
            for line_number in range(start, min(end, len(source)) + 1):
                candidate_text = selected + source[line_number - 1]
                if len(encode(candidate_text).encode()) > REPOSITORY_TOOL_VISIBLE_RESULT_BYTES:
                    if not selected:
                        return {
                            "revision": revision,
                            "path": path,
                            "line": line_number,
                            "diagnostic": "line exceeds serialized result limit",
                        }
                    break
                selected = candidate_text
                actual_end = line_number
            clipped = actual_end < min(end, len(source))
            return {
                "revision": revision,
                "commit": self.index.snapshots[revision].commit,
                "path": path,
                "blob": found.blob,
                "start": start,
                "end": actual_end,
                "text": selected,
                "truncated": clipped or end > len(source),
                "next_start": actual_end + 1 if clipped else None,
            }
        raise ValueError("unknown repository tool")

    def _read_pr_diff(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Page exact captured patch records, retaining comparison/head inline coordinates.

        Arguments require current inventory path, decimal record cursor and bounded limit.
        Whole records must fit the visible byte ceiling; an oversized record is diagnosed,
        never silently clipped. Missing context/path and out-of-range cursors are explicit.
        Results carry pinned revisions and patch digest for deterministic audited replay.
        """
        if set(arguments) != {"path", "cursor", "limit"}:
            raise ValueError("invalid PR diff arguments")
        path, cursor, limit = arguments["path"], arguments["cursor"], arguments["limit"]
        if not isinstance(path, str) or not _valid_path(path):
            raise ValueError("invalid repository path")
        if not isinstance(cursor, str) or not cursor.isdecimal():
            raise ValueError("cursor must be a decimal offset")
        if type(limit) is not int or not 1 <= limit <= REPOSITORY_RANGE_LINES:
            raise ValueError("limit must be within the diff record bound")
        file = next((f for f in self.context.files if f.filename == path), None)
        if file is None:
            return {"path": path, "diagnostic": "not a captured changed-file path"}
        # Bind evidence to the original PR comparison, not a freshly generated base-tip diff.
        result: dict[str, Any] = {
            "path": file.filename,
            "previous_filename": file.previous_filename,
            "status": file.status,
            "before_mode": file.before_mode,
            "after_mode": file.after_mode,
            "head": self.context.head,
            "comparison": self.context.diff_base,
            "base": self.context.base,
            "patch_sha256": digest(file.patch),
            "patch_start": int(cursor),
        }
        offset = int(cursor)
        records = islice(walk_patch(file.patch), offset, None)
        page: list[dict[str, Any]] = []
        more = False
        # Stream complete records; do not allocate the entire patch for a bounded read.
        for record in islice(records, limit):
            old, new = record.consumes
            candidate = {
                "text": record.text,
                "LEFT": record.left if old else None,
                "RIGHT": record.right if new else None,
            }
            if len(encode([*page, candidate]).encode()) > REPOSITORY_TOOL_VISIBLE_RESULT_BYTES:
                if not page:
                    return {**result, "diagnostic": "patch record exceeds serialized result limit"}
                more = True
                break
            page.append(candidate)
        else:
            more = next(records, None) is not None
        if not page and offset:
            return {**result, "diagnostic": "patch cursor unavailable"}
        end = offset + len(page)
        return {
            **result,
            "records": page,
            "patch_end": end,
            "next_cursor": str(end) if more else "",
            "truncated": more,
        }

    def _read_text(self, entry: SnapshotEntry) -> str:
        """Read a captured text blob from its validated Git SHA, never from a model path."""
        if not entry.blob or not re.fullmatch(r"[0-9a-f]{40}", entry.blob):
            raise RepositorySnapshotError("Snapshot blob identity is invalid")
        path = self.blob_dir / entry.blob
        if path.is_symlink() or not path.is_file():
            raise RepositorySnapshotError("Snapshot text blob is missing or unsafe")
        raw = path.read_bytes()
        if len(raw) != entry.size or _git_blob_sha(raw) != entry.blob:
            raise RepositorySnapshotError("Snapshot text blob failed Git SHA verification")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RepositorySnapshotError("Snapshot text blob is not UTF-8") from error
        if "\0" in text:
            raise RepositorySnapshotError("Snapshot text blob contains NUL")
        return text
