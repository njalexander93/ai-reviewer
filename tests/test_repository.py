"""Exercise immutable full-source snapshots and their bounded provider tools."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_reviewer import models
from ai_reviewer import repository as repository_module
from ai_reviewer.contracts import ChunkReview, ReviewContext
from ai_reviewer.repository import (
    RepositorySnapshotError,
    RepositoryTools,
    SnapshotEntry,
    SnapshotIndex,
    TreeSnapshot,
    load_repository,
)
from ai_reviewer.serialization import digest, encode
from ai_reviewer.work import attempt_directory, replay_tool_events, tool_events_digest
from tests.test_github import Server, bundle


def git_sha(text: str) -> str:
    """Return the content-address used by Git for one UTF-8 blob."""
    raw = text.encode()
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def make_repository(tmp_path: Path) -> tuple[ReviewContext, RepositoryTools]:
    """Build three pinned source sides, including an unchanged dependency."""
    context = bundle(Server())[0]
    blob_dir = tmp_path / "repository" / "blobs"
    blob_dir.mkdir(parents=True)
    sources = {
        "task_boundary.py": "\n" * 468 + "def edit_roles():\n    return DEFAULT_ROLE\n",
        "lib/roles.py": "DEFAULT_ROLE = 'reviewer'\n",
        "deleted.py": "obsolete = True\n",
        "old_name.py": "before = True\n",
        "new_name.py": "after = True\n",
    }
    entries = []
    for path, text in sources.items():
        sha = git_sha(text)
        (blob_dir / sha).write_bytes(text.encode())
        entries.append(
            SnapshotEntry(
                path=path, mode="100644", blob=sha, size=len(text.encode()), status="text"
            )
        )
    entries.extend(
        [
            SnapshotEntry(
                path="link.py", mode="120000", blob=git_sha("target.py"), size=9, status="symlink"
            ),
            SnapshotEntry(path="nested", mode="160000", blob=None, size=0, status="submodule"),
        ]
    )
    head_entries = [entry for entry in entries if entry.path not in {"deleted.py", "old_name.py"}]
    comparison_entries = [entry for entry in entries if entry.path != "new_name.py"]
    base_entries = comparison_entries

    def snapshot(commit: str, rows: list[SnapshotEntry]) -> TreeSnapshot:
        identity = {
            "commit": commit,
            "tree": commit[::-1],
            "entries": [row.model_dump() for row in sorted(rows, key=lambda e: e.path)],
        }
        return TreeSnapshot(**identity, inventory_digest=digest(encode(identity)))

    snapshots = {
        "head": snapshot(context.head, head_entries),
        "comparison": snapshot(context.diff_base, comparison_entries),
        "base": snapshot(context.base, base_entries),
    }
    body = {
        "repo": context.repo,
        "snapshots": {name: tree.model_dump() for name, tree in snapshots.items()},
    }
    index = SnapshotIndex(**body, digest=digest(encode(body)))
    context.repository_digest = index.digest
    repository_dir = blob_dir.parent
    (repository_dir / "index.json").write_text(index.model_dump_json())
    return context, load_repository(context, repository_dir)


def test_unchanged_dependency_search_and_edit_roles_source_are_available(tmp_path: Path) -> None:
    """Search locates a declaration omitted from changed-file navigation, then reads it."""
    context, repository = make_repository(tmp_path)
    assert "task_boundary.py" not in {file.filename for file in context.files}
    result = json.loads(
        repository.invoke(
            "search_repository",
            {
                "revision": "head",
                "path_prefix": "",
                "cursor": "0:0",
                "limit": 10,
                "query": "edit_roles",
            },
        )
    )
    assert result["matches"] == [
        {
            "path": "task_boundary.py",
            "line": 469,
            "text": "def edit_roles():",
            "line_truncated": False,
        }
    ]
    unchanged = json.loads(
        repository.invoke(
            "search_repository",
            {
                "revision": "head",
                "path_prefix": "lib",
                "cursor": "0:0",
                "limit": 10,
                "query": "DEFAULT_ROLE",
            },
        )
    )
    assert unchanged["matches"][0]["path"] == "lib/roles.py"
    section = json.loads(
        repository.invoke(
            "read_repository_lines",
            {
                "revision": "head",
                "path": "task_boundary.py",
                "start": 469,
                "end": 470,
            },
        )
    )
    assert section["text"] == "def edit_roles():\n    return DEFAULT_ROLE\n"
    assert context.repository_digest == repository.index.digest


def test_fixed_revision_views_distinguish_changed_deleted_and_renamed_paths(
    tmp_path: Path,
) -> None:
    """The three tool selectors expose only their pinned inventories."""
    _, repository = make_repository(tmp_path)
    for revision, path, diagnostic in [
        ("head", "deleted.py", "missing from snapshot"),
        ("comparison", "deleted.py", None),
        ("head", "old_name.py", "missing from snapshot"),
        ("comparison", "old_name.py", None),
        ("head", "new_name.py", None),
        ("comparison", "new_name.py", "missing from snapshot"),
        ("head", "link.py", "symlink"),
        ("head", "nested", "submodule"),
    ]:
        result = json.loads(
            repository.invoke(
                "read_repository_lines",
                {
                    "revision": revision,
                    "path": path,
                    "start": 1,
                    "end": 1,
                },
            )
        )
        if diagnostic:
            assert result["diagnostic"] == diagnostic
        else:
            assert "text" in result


def test_paths_ranges_and_pagination_are_bounded_and_replayable(tmp_path: Path) -> None:
    """Tool input cannot escape the index, and exact result evidence can be replayed."""
    context, repository = make_repository(tmp_path)
    event_path = attempt_directory(tmp_path, "whole", "claude", 0) / "repository-tools.jsonl"
    session = repository.for_attempt(event_path)
    invalid_path = json.loads(
        session.invoke(
            "read_repository_lines",
            {
                "revision": "head",
                "path": "../../etc/passwd",
                "start": 1,
                "end": 1,
            },
        )
    )
    assert invalid_path["error"] == "invalid repository path"
    invalid_range = json.loads(
        session.invoke(
            "read_repository_lines",
            {
                "revision": "head",
                "path": "task_boundary.py",
                "start": 1,
                "end": 201,
            },
        )
    )
    assert "read bound" in invalid_range["error"]
    listed = json.loads(
        session.invoke(
            "list_repository_files",
            {
                "revision": "head",
                "path_prefix": "",
                "cursor": "0",
                "limit": 2,
            },
        )
    )
    assert listed["truncated"] and listed["next_cursor"] == "2"
    expected = tool_events_digest(tmp_path, "whole", "claude", 1, repository_enabled=True)
    replay_tool_events(context, tmp_path, "whole", "claude", 1, expected)
    event_path.write_text(event_path.read_text().replace('"revision":"head"', '"revision":"base"'))
    with pytest.raises(ValueError, match="does not replay"):
        replay_tool_events(context, tmp_path, "whole", "claude", 1, expected)


@pytest.mark.parametrize(
    "budget,limit",
    [
        ("calls", 0),
        ("result_bytes", 1),
        ("context_tokens", 1),
        ("seconds", 0),
    ],
)
def test_repository_budget_exhaustion_writes_a_diagnostic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, budget: str, limit: int
) -> None:
    """Every local assignment budget failure is explicit and fails closed."""
    _, repository = make_repository(tmp_path)
    target = tmp_path / "budget" / "repository-tools.jsonl"
    session = repository.for_attempt(target)

    def invoke_list() -> str:
        return session.invoke(
            "list_repository_files",
            {
                "revision": "head",
                "path_prefix": "",
                "cursor": "0",
                "limit": 1,
            },
        )

    operation: Callable[[], object]
    if budget == "calls":
        monkeypatch.setattr(repository_module, "REPOSITORY_TOOL_CALLS", limit)
        operation = invoke_list
    elif budget == "result_bytes":
        monkeypatch.setattr(repository_module, "REPOSITORY_TOOL_TOTAL_BYTES", limit)
        operation = invoke_list
    elif budget == "context_tokens":
        monkeypatch.setattr(repository_module, "REPOSITORY_CONTEXT_TOKENS", limit)

        def count_tokens() -> None:
            session.account_tokens(limit + 1)

        operation = count_tokens
    else:
        monkeypatch.setattr(repository_module, "REPOSITORY_TOOL_SECONDS", limit)
        operation = invoke_list
    with pytest.raises(RepositorySnapshotError, match="budget exhausted"):
        operation()
    assert json.loads((target.parent / "exhaustion.json").read_text())["reason"]


class SDKItem:
    """Represent a provider block and preserve its continuation wire shape."""

    def __init__(self, **values: object) -> None:
        """Set the provider-block fields supplied by each offline scenario."""
        self.__dict__.update(values)

    def model_dump(self, **kwargs: object) -> dict[str, object]:
        """Return the exact synthetic SDK block fields."""
        return self.__dict__


@pytest.mark.parametrize("role", ["openai", "claude"])
def test_provider_tool_round_trip_is_offline_and_records_cost_and_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role: str
) -> None:
    """Both provider adapters send bounded search results back before final JSON."""
    _, repository = make_repository(tmp_path)
    counts: list[dict] = []
    generations: list[dict] = []
    tool_call = SDKItem(
        type="function_call" if role == "openai" else "tool_use",
        name="search_repository",
        arguments=json.dumps(
            {
                "revision": "head",
                "path_prefix": "",
                "cursor": "0:0",
                "limit": 10,
                "query": "edit_roles",
            }
        ),
        call_id="call-1",
        id="call-1",
        input={
            "revision": "head",
            "path_prefix": "",
            "cursor": "0:0",
            "limit": 10,
            "query": "edit_roles",
        },
    )
    report = ChunkReview(
        complete=True, summary="Inspected the declaration.", findings=[], requests=[]
    )

    def usage() -> SimpleNamespace:
        return SimpleNamespace(
            input_tokens=100,
            output_tokens=10,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )

    first = SimpleNamespace(
        output=[tool_call],
        usage=usage(),
        status="completed",
        output_parsed=None,
        content=[tool_call],
        stop_reason="tool_use",
        parsed_output=None,
    )
    final = SimpleNamespace(
        output=[SDKItem(type="message")],
        usage=usage(),
        status="completed",
        output_parsed=report,
        content=[],
        stop_reason="end_turn",
        parsed_output=report,
    )

    def count(**kwargs: object) -> SimpleNamespace:
        counts.append(kwargs)
        return SimpleNamespace(input_tokens=100)

    def generate(**kwargs: object) -> SimpleNamespace:
        generations.append(kwargs)
        return [first, final][len(generations) - 1]

    client = SimpleNamespace(
        responses=SimpleNamespace(
            input_tokens=SimpleNamespace(count=count),
            parse=generate,
        ),
        messages=SimpleNamespace(count_tokens=count, parse=generate),
    )
    monkeypatch.setattr(models.openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(models.anthropic, "Anthropic", lambda **kwargs: client)
    target = tmp_path / "model" / role
    models.call_model(role, "review payload", target, repository=repository)
    assert len(generations) == 2
    assert all(count["tools"] for count in counts)
    usage_result = json.loads((target / f"{role}.usage.json").read_text())
    assert usage_result["input"] == 200
    rows = (target / "repository-tools.jsonl").read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["result"]["matches"][0]["line"] == 469
    assert ChunkReview.model_validate_json((target / f"{role}.json").read_text()) == report
    if role == "openai":
        assert "tools" in generations[0]
        assert any(item.get("type") == "function_call_output" for item in generations[1]["input"])
    else:
        assert "tools" in generations[0]
        assert generations[1]["messages"][-1]["content"][0]["type"] == "tool_result"


def test_provider_tool_turn_exhaustion_fails_without_a_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exhaustion preserves its reason and never accepts a partial review."""
    _, repository = make_repository(tmp_path)
    monkeypatch.setattr(models, "REPOSITORY_TOOL_TURNS", 1)
    call = SDKItem(type="function_call", name="search_repository", arguments="{}", call_id="call-1")
    response = SimpleNamespace(
        output=[call],
        output_parsed=None,
        status="completed",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=1,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(
            input_tokens=SimpleNamespace(count=lambda **kwargs: SimpleNamespace(input_tokens=10)),
            parse=lambda **kwargs: response,
        )
    )
    monkeypatch.setattr(models.openai, "OpenAI", lambda **kwargs: client)
    target = tmp_path / "exhausted"
    with pytest.raises(RepositorySnapshotError, match="turns budget"):
        models.call_model("openai", "review", target, repository=repository)
    assert json.loads((target / "exhaustion.json").read_text())["reason"] == "turns"
    assert not (target / "openai.json").exists()
