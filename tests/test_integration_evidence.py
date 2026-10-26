"""Exercise scoped integration through real planning, SDK adapters and coverage replay.

Synthetic provider responses test the review protocol, not model understanding. Fixture
source and the task's missing-evidence/coverage rules supply the independent expectations.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_reviewer import chunking, models, work
from ai_reviewer.contracts import ChangedFile, Review, ReviewContext
from ai_reviewer.github import rebuild_patch
from ai_reviewer.repository import SnapshotEntry, SnapshotIndex, TreeSnapshot, load_repository
from ai_reviewer.repository_evidence import replay_tool_events, tool_events_digest
from ai_reviewer.serialization import digest, encode
from tests.test_github import Server, bundle
from tests.test_repository import SDKItem, git_sha


def captured_context(directory: Path) -> ReviewContext:
    """Capture three related changed files, with a divergent target tip and one rename."""
    context = bundle(Server())[0]
    context.diff_base = "c" * 40
    padding = "# unchanged padding\n" * 80
    versions = [
        (
            "src/api.py",
            None,
            "def transform(value):\n    return value\n",
            "def transform(value, strict=False):\n    return value\n",
        ),
        (
            "tests/integration/test_api.py",
            "tests/test_api.py",
            "from src.api import transform\nassert transform(1) == 1\n",
            "from src.api import transform\nassert transform(1, strict=True) == 1\n",
        ),
        (
            "tests/unit/test_other.py",
            None,
            "from src.api import transform\nassert transform(2) == 2\n",
            "from src.api import transform\nassert transform(3) == 3\n",
        ),
    ]
    context.files = [
        ChangedFile(
            filename=path,
            previous_filename=old,
            status="renamed" if old else "modified",
            patch=rebuild_patch(before + padding, after + padding),
            before=before + padding,
            after=after + padding,
            before_mode="100644",
            after_mode="100644",
        )
        for path, old, before, after in versions
    ]
    blobs = directory / "repository" / "blobs"
    blobs.mkdir(parents=True)
    snapshots = {}
    for side, commit in [
        ("head", context.head),
        ("comparison", context.diff_base),
        ("base", context.base),
    ]:
        entries = []
        for file in context.files:
            path = file.filename if side == "head" else file.previous_filename or file.filename
            text = file.after if side == "head" else file.before
            if side == "base":
                text = "# target advanced after branching\n" + text
            sha = git_sha(text)
            (blobs / sha).write_bytes(text.encode())
            entries.append(
                SnapshotEntry(
                    path=path, mode="100644", blob=sha, size=len(text.encode()), status="text"
                )
            )
        identity = {
            "commit": commit,
            "tree": commit,
            "entries": [e.model_dump() for e in sorted(entries, key=lambda e: e.path)],
        }
        snapshots[side] = TreeSnapshot(**identity, inventory_digest=digest(encode(identity)))
    body = {"repo": context.repo, "snapshots": {s: t.model_dump() for s, t in snapshots.items()}}
    index = SnapshotIndex(**body, digest=digest(encode(body)))
    context.repository_digest = index.digest
    (blobs.parent / "index.json").write_text(index.model_dump_json())
    return context


@pytest.mark.parametrize("provider", ["claude", "openai"])
@pytest.mark.parametrize("missing", [False, True])
def test_scoped_integration_retrieves_omitted_diff_and_requires_full_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str, missing: bool
) -> None:
    """Real orchestration accepts scoped completion, but missing evidence/work blocks it."""
    context = captured_context(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline" if provider == "claude" else "")
    monkeypatch.setenv("OPENAI_API_KEY", "offline" if provider == "openai" else "")
    monkeypatch.setattr(chunking, "CHUNK_BYTES", 5_000)
    target = "tests/integration/test_api.py"
    retrieved = []
    edge_packets = []

    def packet(kwargs: dict) -> dict:
        """Read the actual initial user packet from either SDK request shape."""
        messages = kwargs.get("input", kwargs.get("messages"))
        return json.loads(messages[0]["content"])

    def count(**kwargs: object) -> SimpleNamespace:
        """Force the supported capacity fallback for the small synthetic fixture."""
        return SimpleNamespace(input_tokens=300_000 if "files" in packet(kwargs) else 100)

    def generate(**kwargs: object) -> SimpleNamespace:
        """Model a targeted compatibility investigation through actual SDK tool messages."""
        data = packet(kwargs)
        messages = kwargs.get("input", kwargs.get("messages"))
        instructions = kwargs.get("instructions", kwargs.get("system"))
        names = {tool["name"] for tool in kwargs["tools"]}
        assert {"read_pr_diff", "search_repository", "read_repository_lines"} <= names
        scope = data["scope"]
        paths = {unit["filename"] for unit in scope["units"]}
        integration = "phase" in data
        needs_diff = integration and "src/api.py" in paths and target not in paths
        if integration:
            assert "complete field covers your owned sections" in instructions
            assert "Cross-chunk edge assignments must check" in instructions
            assert (
                "Not executing tests is not by itself missing static-review evidence"
                in instructions
            )
            assert len(data["overview"]["inventory"]) == 3
            coverage = data["overview"]["completed_local_scopes"]
            assert coverage["providers"] == [provider]
            assert coverage == {"providers": [provider], "count": 3}
            if scope["id"].startswith("edge-"):
                edge_packets.append(paths)
        if needs_diff and len(messages) == 1:
            arguments = {"path": "missing.py" if missing else target, "cursor": "0", "limit": 200}
            call = SDKItem(
                type="function_call" if provider == "openai" else "tool_use",
                name="read_pr_diff",
                arguments=encode(arguments),
                input=arguments,
                call_id="diff-1",
                id="diff-1",
            )
            return SimpleNamespace(
                output=[call],
                content=[call],
                output_parsed=None,
                parsed_output=None,
                status="completed",
                stop_reason="tool_use",
                usage=None,
            )
        if needs_diff:
            raw = (
                messages[-1]["output"]
                if provider == "openai"
                else messages[-1]["content"][0]["content"]
            )
            evidence = json.loads(raw)
            retrieved.append(evidence)
            if missing:
                assert evidence["diagnostic"] == "not a captured changed-file path"
            else:
                assert evidence["comparison"] == "c" * 40
                assert evidence["base"] == "b" * 40
                assert evidence["previous_filename"] == "tests/test_api.py"
                assert {
                    "text": "+assert transform(1, strict=True) == 1\n",
                    "LEFT": None,
                    "RIGHT": 2,
                } in evidence["records"]
                assert evidence["next_cursor"] == ""
        report = Review(
            complete=not (needs_diff and missing),
            findings=[],
            summary="Required caller diff unavailable"
            if needs_diff and missing
            else "Assigned changes and interactions checked; tests not executed",
        )
        return SimpleNamespace(
            output=[],
            content=[],
            output_parsed=report,
            parsed_output=report,
            status="completed",
            stop_reason="end_turn",
            usage=None,
        )

    client = SimpleNamespace(
        responses=SimpleNamespace(input_tokens=SimpleNamespace(count=count), parse=generate),
        messages=SimpleNamespace(count_tokens=count, parse=generate),
    )
    monkeypatch.setattr(models.openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(models.anthropic, "Anthropic", lambda **kwargs: client)
    if missing:
        with pytest.raises(ValueError, match="failed"):
            work.review_chunks(context, tmp_path)
        assert retrieved
        with pytest.raises(ValueError):
            work.load_reviews(context, tmp_path)
        failures = list((tmp_path / "work").rglob("failure.json"))
        assert failures
        assert any("repository_evidence_incomplete" in f.read_text() for f in failures)
        return
    work.review_chunks(context, tmp_path)
    covered = work.load_reviews(context, tmp_path)
    assert retrieved
    assert any({"src/api.py", target} <= paths for paths in edge_packets)
    assert all(review.complete for review in covered.reviews)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Removing even a successful edge assignment must prevent aggregate acceptance.
    removed = next(a["id"] for a in manifest["assignments"] if a["id"].startswith("edge-"))
    manifest["assignments"] = [a for a in manifest["assignments"] if a["id"] != removed]
    manifest["results"] = [r for r in manifest["results"] if r["assignment"] != removed]
    manifest_path.write_text(encode(manifest))
    with pytest.raises(ValueError, match="Coverage assignments are missing"):
        work.load_reviews(context, tmp_path)


def test_diff_pages_preserve_rename_coordinates_and_replay(tmp_path: Path) -> None:
    """Two-record pages reconstruct the original merge-base patch with exact anchors."""
    context = captured_context(tmp_path)
    repository = load_repository(context, tmp_path / "repository")
    path = (
        work.attempt_directory(tmp_path, "integration-0000", "openai", 0) / "repository-tools.jsonl"
    )
    session = repository.for_attempt(path)
    cursor = "0"
    pages = []
    while cursor:
        page = json.loads(
            session.invoke(
                "read_pr_diff",
                {
                    "path": "tests/integration/test_api.py",
                    "cursor": cursor,
                    "limit": 2,
                },
            )
        )
        pages.append(page)
        cursor = page["next_cursor"]
    assert len(pages) > 1
    records = [r for page in pages for r in page["records"]]
    assert "".join(r["text"] for r in records) == context.files[1].patch
    assert {"text": "-assert transform(1) == 1\n", "LEFT": 2, "RIGHT": None} in records
    assert {"text": "+assert transform(1, strict=True) == 1\n", "LEFT": None, "RIGHT": 2} in records
    assert all(p["comparison"] == "c" * 40 and p["base"] == "b" * 40 for p in pages)
    assert pages[-1]["truncated"] is False
    expected = tool_events_digest(
        tmp_path, "integration-0000", "openai", 1, repository_enabled=True
    )
    replay_tool_events(context, tmp_path, "integration-0000", "openai", 1, expected)
    # Provenance must reject an altered patch, even with unchanged repository indexes.
    context.files[1].patch = context.files[1].patch.replace("strict=True", "strict=False")
    with pytest.raises(ValueError, match="does not replay"):
        replay_tool_events(context, tmp_path, "integration-0000", "openai", 1, expected)


@pytest.mark.parametrize(
    "status,before,after",
    [
        ("removed", "gone\n", ""),
        ("added", "", "new\n"),
        ("renamed", "same\n", "same\n"),
    ],
)
def test_diff_add_delete_and_patchless_rename(
    tmp_path: Path, status: str, before: str, after: str
) -> None:
    """Absent sides and patchless metadata do not invent anchors or unavailable content."""
    context = captured_context(tmp_path)
    context.files = [
        ChangedFile(
            filename="file.py",
            previous_filename="old.py" if status == "renamed" else None,
            status=status,
            patch=rebuild_patch(before, after),
            before=before,
            after=after,
            before_mode="100644" if before else None,
            after_mode="100644" if after else None,
        )
    ]
    repository = load_repository(context, tmp_path / "repository")
    page = json.loads(
        repository.invoke("read_pr_diff", {"path": "file.py", "cursor": "0", "limit": 10})
    )
    assert page["status"] == status
    assert page["next_cursor"] == ""
    assert page["truncated"] is False
    anchors = [(r["LEFT"], r["RIGHT"]) for r in page["records"] if r["LEFT"] or r["RIGHT"]]
    assert anchors == {"removed": [(1, None)], "added": [(None, 1)], "renamed": []}[status]
    assert page["previous_filename"] == ("old.py" if status == "renamed" else None)
    assert page["before_mode"] == ("100644" if before else None)
    assert page["after_mode"] == ("100644" if after else None)


def test_diff_limits_are_explicit_and_do_not_silently_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed calls, byte pagination and an indivisible line have distinct diagnostics."""
    from ai_reviewer import repository as module

    context = captured_context(tmp_path)
    context.files[0].patch = (
        "@@ -1,2 +1,2 @@\n-def transform(value):\n"
        "+def transform(value, strict=False):\n     return value\n"
    )
    repository = load_repository(context, tmp_path / "repository")
    arguments = {"path": "src/api.py", "cursor": "0", "limit": 200}
    for change, message in [
        ({"path": "../api.py"}, "invalid repository path"),
        ({"cursor": "-1"}, "cursor must be a decimal offset"),
        ({"limit": True}, "limit must be within the diff record bound"),
        ({"limit": 201}, "limit must be within the diff record bound"),
        ({"revision": "main"}, "invalid PR diff arguments"),
    ]:
        result = json.loads(repository.invoke("read_pr_diff", {**arguments, **change}))
        assert result == {"error": message}
    result = json.loads(repository.invoke("read_pr_diff", {**arguments, "cursor": "999"}))
    assert result["diagnostic"] == "patch cursor unavailable"
    monkeypatch.setattr(module, "REPOSITORY_TOOL_VISIBLE_RESULT_BYTES", 150)
    page = json.loads(repository.invoke("read_pr_diff", arguments))
    assert page["truncated"] is True
    assert page["next_cursor"]
    assert (
        "".join(r["text"] for r in page["records"]) == "@@ -1,2 +1,2 @@\n-def transform(value):\n"
    )
    monkeypatch.setattr(module, "REPOSITORY_TOOL_VISIBLE_RESULT_BYTES", 1)
    result = json.loads(repository.invoke("read_pr_diff", arguments))
    assert result["diagnostic"] == "patch record exceeds serialized result limit"
    assert "records" not in result
