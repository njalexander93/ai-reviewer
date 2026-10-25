"""Exercise immutable full-source snapshots and their bounded provider tools."""

import base64
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_reviewer import models, work
from ai_reviewer import repository as repository_module
from ai_reviewer.context import prepare
from ai_reviewer.contracts import ChunkReview, Review, ReviewContext
from ai_reviewer.diagnostics import UnresolvedEvidence, failure_details
from ai_reviewer.repository import (
    RepositorySnapshotError,
    RepositoryTools,
    SnapshotEntry,
    SnapshotIndex,
    TreeSnapshot,
    _read_repository_blob,
    capture_repository,
    load_repository,
)
from ai_reviewer.repository_evidence import replay_tool_events, tool_events_digest
from ai_reviewer.serialization import digest, encode
from ai_reviewer.work import attempt_directory
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


def test_snapshot_integrity_error_fails_instead_of_becoming_tool_result(tmp_path: Path) -> None:
    """Missing captured blobs fail the assignment instead of returning a recoverable error."""
    _, repository = make_repository(tmp_path)
    entry = repository.entries["head"]["task_boundary.py"]
    (repository.blob_dir / entry.blob).unlink()
    with pytest.raises(RepositorySnapshotError, match="missing or unsafe"):
        repository.invoke(
            "read_repository_lines",
            {"revision": "head", "path": "task_boundary.py", "start": 469, "end": 470},
        )


def test_missing_empty_transcript_is_not_accepted_as_valid_evidence(tmp_path: Path) -> None:
    """A zero-tool review still requires its explicit empty transcript artifact."""
    context, repository = make_repository(tmp_path)
    path = attempt_directory(tmp_path, "whole", "claude", 0) / "repository-tools.jsonl"
    repository.for_attempt(path)
    expected = tool_events_digest(tmp_path, "whole", "claude", 1, repository_enabled=True)
    assert path.exists()
    path.unlink()
    with pytest.raises(FileNotFoundError):
        replay_tool_events(context, tmp_path, "whole", "claude", 1, expected)


@pytest.mark.parametrize(
    "line", ["\x01" * 1000 + "\n", "x" * 20_000 + "\n"], ids=["escaped", "oversized"]
)
def test_line_reads_bound_serialized_bytes_and_report_exact_ranges(
    tmp_path: Path, line: str
) -> None:
    """Escaped source pages retain whole lines; an oversized single line is explicit."""
    _, repository = make_repository(tmp_path)
    text = line * 10
    sha = git_sha(text)
    (repository.blob_dir / sha).write_text(text)
    repository.entries["head"]["escaped.txt"] = SnapshotEntry(
        path="escaped.txt", mode="100644", blob=sha, size=len(text.encode()), status="text"
    )
    result_text = repository.invoke(
        "read_repository_lines",
        {"revision": "head", "path": "escaped.txt", "start": 1, "end": 10},
    )
    assert len(result_text.encode()) <= 16_000
    result = json.loads(result_text)
    if len(line) > 16_000:
        assert result["diagnostic"] == "line exceeds serialized result limit"
        assert result["line"] == 1
    else:
        assert result["text"] == line
        assert result["end"] == 1
        assert result["next_start"] == 2
        assert result["truncated"] is True


def test_repository_replay_rejects_legacy_requests(tmp_path: Path) -> None:
    """The shared execution/replay transition rejects changed-file follow-ups."""
    context, _ = make_repository(tmp_path)
    assignment = work.Assignment(id="whole", kind="whole", payload="{}", sections=[])
    report = ChunkReview(
        complete=False,
        summary="Need source",
        findings=[],
        requests=[{"path": "a.py", "side": "RIGHT", "start": 1, "end": 1, "reason": "Read"}],
    )
    with pytest.raises(ValueError, match="one complete report without requests"):
        work.advance(context, assignment, [], report, 600_000)


def test_tool_history_can_grow_beyond_initial_packet_cap(tmp_path: Path) -> None:
    """A valid continuation can exceed the initial packet cap within its own budget."""
    _, repository = make_repository(tmp_path)
    responses = iter(["tool", "done"])
    adapter = models.ProviderAdapter(
        initial="small packet",
        count=lambda current: 100,
        generate=lambda current: next(responses),
        usage=lambda response: None,
        calls=lambda response: (
            [
                (
                    "list_repository_files",
                    {
                        "revision": "head",
                        "path_prefix": "",
                        "cursor": "0",
                        "limit": 1,
                    },
                    "call-1",
                )
            ]
            if response == "tool"
            else []
        ),
        continue_turn=lambda current, response, results: "x" * 600_001,
        finish=lambda response: (response, True),
    )
    assert models._run_provider_turns(
        adapter,
        payload="small packet",
        session=repository,
        preflight=False,
        usage_path=tmp_path / "usage.json",
    ) == ("done", True)


def test_expired_assignment_never_starts_a_token_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An expired assignment rejects work before contacting a provider again."""
    _, repository = make_repository(tmp_path)
    monkeypatch.setattr(repository_module.time, "monotonic", lambda: repository.started + 601)
    adapter = SimpleNamespace(
        initial="payload", count=lambda current: pytest.fail("expired provider count started")
    )
    with pytest.raises(RepositorySnapshotError, match="seconds budget"):
        models._run_provider_turns(
            adapter,
            payload="payload",
            session=repository,
            preflight=False,
            usage_path=tmp_path / "usage.json",
        )


def test_incomplete_repository_review_retains_evidence_failure_category(tmp_path: Path) -> None:
    """An incomplete tool review is evidence failure, not a provider transport error."""
    context, _ = make_repository(tmp_path)
    assignment = work.Assignment(id="whole", kind="whole", payload="{}", sections=[])
    report = ChunkReview(
        complete=False, summary="Need unreadable dependency", findings=[], requests=[]
    )
    with pytest.raises(UnresolvedEvidence) as error:
        work.advance(context, assignment, [], report, 600_000)
    assert failure_details(error.value) == {
        "error": "UnresolvedEvidence",
        "category": "unresolved_evidence",
        "reason": "repository_evidence_incomplete",
        "attempts": 1,
    }


def test_github_wrapped_base64_blob_is_decoded_and_verified() -> None:
    """The blob reader accepts GitHub's line-wrapped base64 while keeping SHA checks."""
    raw = b"print('wrapped')\n"
    sha = git_sha(raw.decode())
    wrapped = base64.b64encode(raw).decode()
    wrapped = wrapped[:8] + "\n" + wrapped[8:] + "\n"

    class GitHubBlob:
        def request(self, *_: object, **__: object) -> dict[str, object]:
            return {"encoding": "base64", "size": len(raw), "content": wrapped}

    assert _read_repository_blob(GitHubBlob(), "owner/repo", sha, len(raw)) == raw


def test_blob_request_budget_keeps_complete_inventory_with_unavailable_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Capture stops fetching at its request cap while retaining all tracked paths."""
    context = bundle(Server())[0]
    monkeypatch.setattr(repository_module, "MAX_REPOSITORY_BLOB_REQUESTS", 1)
    sources = {
        context.head: b"head source\n",
        context.diff_base: b"comparison source\n",
        context.base: b"base source\n",
    }
    blobs = {git_sha(value.decode()): value for value in sources.values()}

    class GitHubTrees:
        requests_remaining = 10_000

        def request(self, _method: str, path: str, **_: object) -> dict[str, object]:
            commit = path.rsplit("/", 1)[1]
            if "/git/commits/" in path:
                return {"tree": {"sha": commit}}
            if "/git/trees/" in path:
                raw = sources[commit]
                sha = git_sha(raw.decode())
                return {
                    "truncated": False,
                    "tree": [
                        {
                            "path": f"{commit[:4]}.py",
                            "type": "blob",
                            "mode": "100644",
                            "sha": sha,
                            "size": len(raw),
                        }
                    ],
                }
            sha = path.rsplit("/", 1)[1]
            return {
                "encoding": "base64",
                "size": len(blobs[sha]),
                "content": base64.b64encode(blobs[sha]).decode(),
            }

    index = capture_repository(GitHubTrees(), context, tmp_path / "repository")
    assert (
        sum(entry.status == "text" for tree in index.snapshots.values() for entry in tree.entries)
        == 1
    )
    assert (
        sum(
            entry.status == "unavailable"
            for tree in index.snapshots.values()
            for entry in tree.entries
        )
        == 2
    )
    assert all(len(tree.entries) == 1 for tree in index.snapshots.values())


def test_insufficient_github_budget_fails_repository_capture(tmp_path: Path) -> None:
    """A requested repository review cannot silently downgrade its source access."""
    server = Server()

    class LimitedGitHub:
        requests_remaining = 7

        def request(self, *args: object, **kwargs: object) -> object:
            return server.gh.request(*args, **kwargs)

        def pages(self, *args: object, **kwargs: object) -> object:
            return server.gh.pages(*args, **kwargs)

    with pytest.raises(ValueError, match="Insufficient GitHub request budget"):
        prepare(
            LimitedGitHub(),
            "owner/repo",
            1,
            "reviewer",
            repository_directory=tmp_path / "repository",
        )
    assert not (tmp_path / "repository" / "index.json").exists()


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


@pytest.mark.parametrize("missing_usage", [None, 0, 1])
@pytest.mark.parametrize("role", ["openai", "claude"])
def test_provider_tool_round_trip_is_offline_and_records_cost_and_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role: str, missing_usage: int | None
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
    report = Review(complete=True, summary="Inspected the declaration.", findings=[])
    reasoning = SDKItem(
        type="reasoning",
        id="rs_123",
        summary=[],
        encrypted_content="opaque-encrypted-reasoning",
        status="completed",
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
        output=[reasoning, tool_call] if role == "openai" else [tool_call],
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
        index = len(generations) - 1
        response = [first, final][index]
        if missing_usage == index:
            response.usage = None
        return response

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
    if missing_usage is None:
        assert usage_result["input"] == 200
    else:
        assert usage_result is None
    rows = (target / "repository-tools.jsonl").read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["result"]["matches"][0]["line"] == 469
    assert (
        ChunkReview.model_validate_json((target / f"{role}.json").read_text()).model_dump(
            exclude={"requests"}
        )
        == report.model_dump()
    )
    if role == "openai":
        assert "tools" in generations[0]
        assert generations[0]["include"] == ["reasoning.encrypted_content"]
        assert any(
            item.get("type") == "reasoning"
            and item.get("encrypted_content") == "opaque-encrypted-reasoning"
            for item in generations[1]["input"]
        )
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


@pytest.mark.parametrize(
    "initial_count,continuation_count,turns", [(150_000, 250_000, 12), (150_001, 150_001, 1)]
)
def test_repository_token_headroom_is_reserved_before_generation(
    tmp_path: Path, initial_count: int, continuation_count: int, turns: int
) -> None:
    """Reject oversized initial input unpaid; allow twelve full-sized planned turns."""
    _, repository = make_repository(tmp_path)
    generated: list[int] = []

    def generate(current: int) -> int:
        """Record each paid turn and return its ordinal as a synthetic response."""
        generated.append(current)
        return current

    adapter = models.ProviderAdapter(
        initial=0,
        count=lambda current: initial_count if current == 0 else continuation_count,
        generate=generate,
        usage=lambda response: None,
        calls=lambda response: (
            [
                (
                    "list_repository_files",
                    {"revision": "head", "path_prefix": "", "cursor": "0", "limit": 1},
                    str(response),
                )
            ]
            if response < turns - 1
            else []
        ),
        continue_turn=lambda current, response, results: current + 1,
        finish=lambda response: (response, True),
    )
    if initial_count > 150_000:
        with pytest.raises(models.CapacityError):
            models._run_provider_turns(
                adapter,
                payload="{}",
                session=repository,
                preflight=True,
                usage_path=tmp_path / "usage.json",
            )
        assert generated == []
    else:
        result = models._run_provider_turns(
            adapter,
            payload="{}",
            session=repository,
            preflight=False,
            usage_path=tmp_path / "usage.json",
        )
        assert result == (11, True)
        assert generated == list(range(12))
        assert repository.context_tokens == 2_900_000


def test_openai_continuation_preserves_its_input_history() -> None:
    """Appending a provider response returns new state without modifying prior input."""
    history = [{"role": "user", "content": "source"}]
    response = SimpleNamespace(output=[SDKItem(type="message", content="analysis")])
    continued = models._append_openai_turn(history, response, [("call-1", "{}")])
    assert history == [{"role": "user", "content": "source"}]
    assert continued == [
        *history,
        {"type": "message", "content": "analysis"},
        {"type": "function_call_output", "call_id": "call-1", "output": "{}"},
    ]
