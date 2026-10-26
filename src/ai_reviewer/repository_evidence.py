"""Own attempt paths and replay repository-tool transcripts against pinned snapshots."""

import json
from pathlib import Path

from ai_reviewer.contracts import ReviewContext
from ai_reviewer.repository import RepositoryTools, load_repository
from ai_reviewer.serialization import digest, encode


def attempt_directory(directory: Path, assignment: str, role: str, round_: int) -> Path:
    """Return one attempt path without I/O; IDs must be trusted internal components."""
    return directory / "work" / assignment / role / str(round_)


def tool_events_digest(
    directory: Path, assignment: str, role: str, attempts: int, *, repository_enabled: bool
) -> str | None:
    """Hash ordered attempt-local repository transcripts for coordinator-owned replay."""
    if not repository_enabled:
        return None
    evidence = []
    for round_ in range(attempts):
        path = attempt_directory(directory, assignment, role, round_) / "repository-tools.jsonl"
        data = path.read_bytes()
        evidence.append({"round": round_, "sha256": digest(data.decode()), "bytes": len(data)})
    return digest(encode(evidence))


def replay_tool_events(
    context: ReviewContext,
    directory: Path,
    assignment: str,
    role: str,
    attempts: int,
    expected_digest: str | None,
    repository: RepositoryTools | None = None,
) -> None:
    """Re-execute logged reads against captured blobs and compare exact results."""
    if context.repository_digest is None:
        if expected_digest is not None:
            raise ValueError("Repository tool artifacts exist without a pinned snapshot")
        return
    if expected_digest is None:
        raise ValueError("Repository tool transcript digest is missing")
    replay = repository or load_repository(context, directory / "repository")
    if replay.index.digest != context.repository_digest:
        raise ValueError("Repository tool replay uses a different captured snapshot")
    for round_ in range(attempts):
        path = attempt_directory(directory, assignment, role, round_) / "repository-tools.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        session = RepositoryTools(
            replay.index, replay.blob_dir, entries=replay.entries, context=context
        )
        for sequence, row in enumerate(rows):
            if row.get("sequence") != sequence or row.get("snapshot_digest") != replay.index.digest:
                raise ValueError("Repository tool transcript identity or order is invalid")
            result_text = session.invoke(row["name"], row["arguments"], row.get("call_id", ""))
            result_bytes = len(result_text.encode())
            if (
                result_text != encode(row.get("result"))
                or digest(result_text) != row.get("result_sha256")
                or result_bytes != row.get("result_bytes")
            ):
                raise ValueError(
                    "Repository tool result does not replay from the captured snapshot"
                )
    if (
        tool_events_digest(directory, assignment, role, attempts, repository_enabled=True)
        != expected_digest
    ):
        raise ValueError("Repository tool transcript digest does not match the manifest")
