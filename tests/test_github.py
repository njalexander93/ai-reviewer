"""Exercise GitHub boundaries using REST-shaped fixtures and real HTTP request construction."""

import base64
import json

import httpx
import pytest

from ai_reviewer.contracts import Consolidation, Finding, Group, Review
from ai_reviewer.github import GitHub, diff_lines, pr_path, prepare, publish

PATCH = "@@ -1,2 +1,2 @@\n first\n-old\n+new"


class Server:
    """Serve mutable REST fixtures and record calls without accessing GitHub.

    Attributes:
        pr: Open PR metadata, mutable to simulate state and revision changes.
        files: Changed-file records containing unified patches.
        reviews: Existing review records used for duplicate detection.
        writes: Ordered pairs of POST paths and decoded request payloads.
        content_refs: Commit references requested through the contents endpoint.
        content: Base64 file response returned for either version.
        login: Identity returned by the authenticated-user endpoint.
        gh: Production adapter bound to this server's synthetic HTTP transport.
    """

    def __init__(self) -> None:
        """Start with one open PR whose second line changes."""
        # Establish a valid baseline that each test can invalidate selectively.
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
        # Replace the HTTP boundary before any production method can make a request.
        self.gh = GitHub("test-token")
        self.gh.client = httpx.Client(
            base_url="https://api.github.com", transport=httpx.MockTransport(self.handle)
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Respond to the production adapter's requests using in-memory fixtures.

        Args:
            request: HTTPX request constructed by the production GitHub adapter.

        Returns:
            HTTP 200 with the selected fixture or an empty POST acknowledgement.
            Records POST payloads and content commit references for assertions.
        """
        # Record writes separately from fixture reads.
        path = request.url.path
        if request.method == "POST":
            self.writes.append((path, json.loads(request.content)))
            return httpx.Response(200, json={})
        # Select the response by API endpoint, defaulting to PR metadata.
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
        # Use HTTPX serialization so callers exercise real response decoding.
        return httpx.Response(200, json=data)


def bundle(server: Server, blocking: bool = False) -> tuple[dict, list[Review], Consolidation]:
    """Build a complete review pair for the fixture's fixed revision identifiers.

    Args:
        server: Test server retained in the helper signature; its state is not read.
        blocking: Whether the first report's finding should require changes.

    Returns:
        Context, two independent reports and a source-preserving consolidation.
    """
    # Fix the reviewed revisions independently of later server mutations.
    context = {
        "repo": "owner/repo",
        "number": 1,
        "head": "a" * 40,
        "base": "b" * 40,
        "reviewer": "reviewer",
    }
    # Anchor one finding to the second line of the synthetic patch.
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
    # Preserve the single source in the consolidated grouping.
    result = Consolidation(summary="One finding", groups=[Group(source_ids=["0:0"])])
    return context, reviews, result


@pytest.mark.parametrize("blocking,event", [(False, "APPROVE"), (True, "REQUEST_CHANGES")])
def test_publish_verdict_preserves_severity(blocking: bool, event: str) -> None:
    """Blocking findings request changes; nonblocking findings permit approval."""
    # Prepare a synthetic PR with the selected blocking status.
    server = Server()
    # Publish once and inspect the verdict, commit and inline evidence sent to GitHub.
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
    # Start with a publishable review, then invalidate the selected boundary.
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
    # Require rejection before any review write reaches the transport.
    with pytest.raises(ValueError):
        publish(server.gh, context, reviews, result)
    assert server.writes == []


def test_rerun_does_not_duplicate_review() -> None:
    """A completed review marker from the actual reviewer makes retries idempotent."""
    # Seed a matching revision marker owned by the real reviewer.
    server = Server()
    server.reviews = [
        {"user": {"login": "reviewer"}, "body": f"<!-- ai-reviewer:{'a' * 40}:{'b' * 40} -->"}
    ]
    # Require the rerun to skip publication and leave the write log empty.
    assert publish(server.gh, *bundle(server)) is False
    assert server.writes == []


def test_other_user_cannot_spoof_completed_review() -> None:
    """A marker in someone else's review cannot suppress this reviewer."""
    # Seed the correct marker under an unrelated account.
    server = Server()
    server.reviews = [
        {"user": {"login": "attacker"}, "body": f"<!-- ai-reviewer:{'a' * 40}:{'b' * 40} -->"}
    ]
    # Require publication despite the forged completion marker.
    assert publish(server.gh, *bundle(server)) is True


def test_prepare_requests_reviewer_and_collects_context() -> None:
    """The requested identity and both immutable file versions reach the review packet."""
    # Collect a packet from the synthetic open PR.
    server = Server()
    result = prepare(server.gh, "owner/repo", 1, "reviewer")
    # Check revision-pinned contents and the requested reviewer identity.
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
    # Invalidate one eligibility or context boundary in an otherwise valid PR.
    server = Server()
    if invalid == "binary":
        del server.files[0]["patch"]
        server.content["content"] = base64.b64encode(b"binary\0data").decode()
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
    # Reject the packet without sending a reviewer request.
    with pytest.raises(ValueError):
        prepare(server.gh, "owner/repo", 1, "reviewer")
    assert server.writes == []


@pytest.mark.parametrize("status,empty", [("added", "before"), ("removed", "after")])
def test_added_and_removed_files_have_one_empty_version(status: str, empty: str) -> None:
    """Nonexistent file versions are not fetched from GitHub."""
    # Represent a file with one nonexistent version.
    server = Server()
    server.files[0]["status"] = status
    # Require preparation to leave that version empty rather than fetching it.
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
    # Return an HTTP failure containing synthetic sensitive text.
    gh = GitHub("secret")
    gh.client = httpx.Client(
        base_url="https://api.github.com",
        transport=httpx.MockTransport(lambda request: httpx.Response(403, text="sensitive-token")),
    )
    # Check the sanitized status error without leaking the response body.
    with pytest.raises(RuntimeError, match="HTTP 403") as error:
        gh.request("GET", "/user")
    assert "sensitive-token" not in str(error.value)


def test_pagination_collects_all_pages_and_refuses_overflow() -> None:
    """A truncated review/file list must not masquerade as complete context."""
    # Provide one full page followed by a final partial page.
    gh = GitHub("test")
    gh.client = httpx.Client(
        base_url="https://api.github.com",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json=[{}] * 100 if request.url.params["page"] == "1" else [{"id": 101}]
            )
        ),
    )
    # Require all entries, then reject an endpoint that never ends pagination.
    assert len(gh.pages("/list")) == 101
    # Switch to endlessly full pages to verify that overflow cannot look complete.
    gh.client = httpx.Client(
        base_url="https://api.github.com",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[{}] * 100)),
    )
    with pytest.raises(ValueError, match="pagination"):
        gh.pages("/list")


def test_omitted_text_patch_is_rebuilt_and_can_anchor_review() -> None:
    """An omitted generated-file patch still permits review of complete text content."""
    # Simulate GitHub omitting an added text file's patch while serving its bytes.
    server = Server()
    del server.files[0]["patch"]
    server.files[0]["status"] = "added"
    context = prepare(server.gh, "owner/repo", 1, "reviewer")
    assert ("RIGHT", 2) in diff_lines(context["files"][0]["patch"])
    # A new API response still omits the patch; publication uses the captured revision.
    server.files = [{"filename": "app.py", "status": "added"}]
    _, reviews, result = bundle(server, blocking=True)
    assert publish(server.gh, context, reviews, result) is True
    assert server.writes[-1][1]["event"] == "REQUEST_CHANGES"
    assert server.writes[-1][1]["comments"][0]["line"] == 2
