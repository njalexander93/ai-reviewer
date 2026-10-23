"""Exercise GitHub boundaries using REST-shaped fixtures and real HTTP request construction."""

import base64
import json

import httpx
import pytest

from ai_reviewer.contracts import Consolidation, Finding, Group, Review
from ai_reviewer.github import GitHub, diff_lines, pr_path, prepare, publish

PATCH = "@@ -1,2 +1,2 @@\n first\n-old\n+new"


class Server:
    """A scripted GitHub transport; expected verdicts come from the review requirements."""

    def __init__(self) -> None:
        """Start with one open PR whose second line changes."""
        self.pr = {
            "state": "open",
            "draft": False,
            "user": {"login": "author"},
            "head": {"sha": "a" * 40},
            "base": {"sha": "b" * 40},
            "changed_files": 1,
            "title": "Fix",
            "body": "",
            "number": 1,
        }
        self.files = [{"filename": "app.py", "status": "modified", "patch": PATCH}]
        self.reviews: list[dict] = []
        self.writes: list[tuple[str, dict]] = []
        self.content_refs: list[str] = []
        self.content = {
            "size": 10,
            "encoding": "base64",
            "content": base64.b64encode(b"first\nnew\n").decode(),
        }
        self.login = "reviewer"
        self.gh = GitHub("test-token")
        self.gh.client = httpx.Client(
            base_url="https://api.github.com", transport=httpx.MockTransport(self.handle)
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Return public API fixtures and record external write effects."""
        path = request.url.path
        if request.method == "POST":
            self.writes.append((path, json.loads(request.content)))
            return httpx.Response(200, json={})
        if path == "/user":
            data = {"login": self.login}
        elif path.endswith("/files"):
            data = self.files
        elif path.endswith("/reviews"):
            data = self.reviews
        elif path.endswith("/comments"):
            data = []
        elif "/compare/" in path:
            data = {"merge_base_commit": {"sha": "c" * 40}}
        elif "/contents/" in path:
            self.content_refs.append(request.url.params["ref"])
            data = self.content
        else:
            data = self.pr
        return httpx.Response(200, json=data)


def bundle(server: Server, blocking: bool = False) -> tuple[dict, list[Review], Consolidation]:
    """Supply an immutable reviewed revision with a real changed-line finding."""
    context = {
        "repo": "owner/repo",
        "number": 1,
        "head": "a" * 40,
        "base": "b" * 40,
        "reviewer": "reviewer",
    }
    finding = Finding(
        path="app.py",
        line=2,
        side="RIGHT",
        title="Problem",
        body="Concrete consequence.",
        blocking=blocking,
    )
    reviews = [
        Review(complete=True, summary="Reviewed", findings=[finding]),
        Review(complete=True, summary="Reviewed independently", findings=[]),
    ]
    result = Consolidation(summary="One finding", groups=[Group(source_ids=["0:0"])])
    return context, reviews, result


@pytest.mark.parametrize("blocking,event", [(False, "APPROVE"), (True, "REQUEST_CHANGES")])
def test_publish_verdict_preserves_severity(blocking: bool, event: str) -> None:
    """Blocking findings request changes; nonblocking findings permit approval."""
    server = Server()
    assert publish(server.gh, *bundle(server, blocking)) is True
    path, posted = server.writes[0]
    assert path == "/repos/owner/repo/pulls/1/reviews"
    assert posted["event"] == event
    assert posted["commit_id"] == "a" * 40
    assert posted["comments"] == [
        {
            "path": "app.py",
            "line": 2,
            "side": "RIGHT",
            "body": "**Problem**\n\nConcrete consequence.",
        }
    ]
    assert "$" not in posted["body"]


@pytest.mark.parametrize(
    "change",
    ["head", "base", "draft", "closed", "self", "identity", "incomplete", "anchor", "dropped"],
)
def test_invalid_publication_never_writes(change: str) -> None:
    """Each invalid boundary must stop the external review submission."""
    server = Server()
    context, reviews, result = bundle(server)
    if change in ("head", "base"):
        server.pr[change]["sha"] = "c" * 40
    elif change == "draft":
        server.pr["draft"] = True
    elif change == "closed":
        server.pr["state"] = "closed"
    elif change == "self":
        server.pr["user"]["login"] = "reviewer"
    elif change == "identity":
        server.login = "wrong-user"
    elif change == "incomplete":
        reviews[0].complete = False
    elif change == "anchor":
        reviews[0].findings[0].line = 999
    else:
        result.groups = []
    with pytest.raises(ValueError):
        publish(server.gh, context, reviews, result)
    assert server.writes == []


def test_rerun_does_not_duplicate_review() -> None:
    """A completed review marker from the actual reviewer makes retries idempotent."""
    server = Server()
    server.reviews = [
        {"user": {"login": "reviewer"}, "body": f"<!-- ai-reviewer:{'a' * 40}:{'b' * 40} -->"}
    ]
    assert publish(server.gh, *bundle(server)) is False
    assert server.writes == []


def test_other_user_cannot_spoof_completed_review() -> None:
    """A marker in someone else's review cannot suppress this reviewer."""
    server = Server()
    server.reviews = [
        {"user": {"login": "attacker"}, "body": f"<!-- ai-reviewer:{'a' * 40}:{'b' * 40} -->"}
    ]
    assert publish(server.gh, *bundle(server)) is True


def test_prepare_requests_reviewer_and_collects_context() -> None:
    """The requested identity and both immutable file versions reach the review packet."""
    server = Server()
    result = prepare(server.gh, "owner/repo", 1, "reviewer")
    assert result["files"][0]["before"] == "first\nnew\n"
    assert result["files"][0]["after"] == "first\nnew\n"
    assert result["head"] == "a" * 40
    assert server.content_refs == ["c" * 40, "a" * 40]
    assert server.writes == [
        ("/repos/owner/repo/pulls/1/requested_reviewers", {"reviewers": ["reviewer"]})
    ]


@pytest.mark.parametrize(
    "invalid", ["binary", "count", "bigfile", "encoding", "draft", "closed", "self", "context"]
)
def test_prepare_refuses_incomplete_context(invalid: str) -> None:
    """Unsupported context never gets a clean review or reviewer request."""
    server = Server()
    if invalid == "binary":
        del server.files[0]["patch"]
    elif invalid == "count":
        server.pr["changed_files"] = 2
    elif invalid == "bigfile":
        server.content["size"] = 150001
    elif invalid == "encoding":
        server.content["encoding"] = "none"
    elif invalid == "draft":
        server.pr["draft"] = True
    elif invalid == "closed":
        server.pr["state"] = "closed"
    elif invalid == "self":
        server.pr["user"]["login"] = "reviewer"
    else:
        server.pr["body"] = "x" * 600001
    with pytest.raises(ValueError):
        prepare(server.gh, "owner/repo", 1, "reviewer")
    assert server.writes == []


@pytest.mark.parametrize("status,empty", [("added", "before"), ("removed", "after")])
def test_added_and_removed_files_have_one_empty_version(status: str, empty: str) -> None:
    """Nonexistent file versions are not fetched from GitHub."""
    server = Server()
    server.files[0]["status"] = status
    assert prepare(server.gh, "owner/repo", 1, "reviewer")["files"][0][empty] == ""


def test_deleted_and_added_lines_have_correct_sides() -> None:
    """Unified diff line numbers map to GitHub's documented LEFT/RIGHT coordinates."""
    assert diff_lines("@@ -7,2 +9,2 @@\n same\n-old\n+new\n\\ No newline") == {
        ("LEFT", 7),
        ("RIGHT", 9),
        ("LEFT", 8),
        ("RIGHT", 10),
    }


@pytest.mark.parametrize("repo,number", [("bad/../../etc", 1), ("owner/repo", 0)])
def test_invalid_repository_identifiers_rejected(repo: str, number: int) -> None:
    """Dispatch input cannot select arbitrary API paths."""
    with pytest.raises(ValueError, match="Invalid"):
        pr_path(repo, number)


def test_http_errors_hide_response_contents() -> None:
    """Error logs must not echo sensitive response bodies."""
    gh = GitHub("secret")
    gh.client = httpx.Client(
        base_url="https://api.github.com",
        transport=httpx.MockTransport(lambda request: httpx.Response(403, text="sensitive-token")),
    )
    with pytest.raises(RuntimeError, match="HTTP 403") as error:
        gh.request("GET", "/user")
    assert "sensitive-token" not in str(error.value)


def test_pagination_collects_all_pages_and_refuses_overflow() -> None:
    """A truncated review/file list must not masquerade as complete context."""
    gh = GitHub("test")
    gh.client = httpx.Client(
        base_url="https://api.github.com",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json=[{}] * 100 if request.url.params["page"] == "1" else [{"id": 101}]
            )
        ),
    )
    assert len(gh.pages("/list")) == 101
    gh.client = httpx.Client(
        base_url="https://api.github.com",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[{}] * 100)),
    )
    with pytest.raises(ValueError, match="pagination"):
        gh.pages("/list")
