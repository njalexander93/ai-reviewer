"""Exercise GitHub boundaries using REST-shaped fixtures and real HTTP request construction."""

import base64
import json

import httpx
import pytest

from ai_reviewer.context import prepare
from ai_reviewer.contracts import Consolidation, Finding, Group, Review, ReviewContext
from ai_reviewer.github import (
    GitHub,
    diff_lines,
    pr_path,
    publish,
    read_version,
    rebuild_patch,
)

PATCH = "@@ -1,2 +1,2 @@\n first\n-old\n+new"


class Server:
    """Serve mutable REST fixtures and record calls without accessing GitHub.

    Attributes:
        pr: Open PR metadata, mutable to simulate state and revision changes.
        files: Changed-file records containing unified patches.
        reviews: Existing review records used for duplicate detection.
        writes: Ordered pairs of POST paths and decoded request payloads.
        content_refs: Commit references resolved to root trees.
        content: Base64 blob response returned for either version.
        mode: Git file mode supplied by tree entries.
        comments: Raw prior discussion records for context projection tests.
        linked_issues: GraphQL issue records linked to the PR.
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
            "head": {"sha": "a" * 40, "ref": "feature/save"},
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
        self.mode = "100644"
        self.comments: list[dict] = []
        self.linked_issues: list[dict] = []
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
        if path == "/graphql":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "repository": {
                            "pullRequest": {
                                "closingIssuesReferences": {
                                    "nodes": self.linked_issues,
                                    "pageInfo": {"hasNextPage": False},
                                }
                            }
                        }
                    }
                },
            )
        if request.method == "POST":
            self.writes.append((path, json.loads(request.content)))
            return httpx.Response(200, json={})
        # Select the response by API endpoint, defaulting to PR metadata.
        if path == "/user":
            data = {"login": self.login}
        elif path.endswith("/files"):
            page = int(request.url.params.get("page", "1"))
            size = int(request.url.params.get("per_page", "100"))
            data = self.files[(page - 1) * size : page * size]
        elif path.endswith("/reviews"):
            data = self.reviews
        elif path.endswith("/comments"):
            data = self.comments
        elif "/compare/" in path:
            data = {"merge_base_commit": {"sha": "c" * 40}}
        elif "/git/commits/" in path:
            ref = path.rsplit("/", 1)[1]
            self.content_refs.append(ref)
            data = {"tree": {"sha": ref}}
        elif "/git/trees/" in path:
            data = {
                "truncated": False,
                "tree": [
                    {
                        "path": "app.py",
                        "type": "blob",
                        "mode": self.mode,
                        "sha": ("e" if path.endswith("c" * 40) else "d") * 40,
                        "size": self.content["size"],
                    }
                ],
            }
        elif "/git/blobs/" in path:
            data = self.content
            if (
                path.endswith("e" * 40)
                and self.content["content"] == base64.b64encode(b"first\nnew\n").decode()
            ):
                data = {**self.content, "content": base64.b64encode(b"first\nold\n").decode()}
        else:
            data = self.pr
        # Use HTTPX serialization so callers exercise real response decoding.
        return httpx.Response(200, json=data)


def bundle(
    server: Server, blocking: bool = False
) -> tuple[ReviewContext, list[Review], Consolidation]:
    """Build a complete review pair for the fixture's fixed revision identifiers.

    Args:
        server: Test server retained in the helper signature; its state is not read.
        blocking: Whether the first report's finding should require changes.

    Returns:
        Context, two independent reports and a source-preserving consolidation.
    """
    # Fix the reviewed revisions independently of later server mutations.
    context = ReviewContext(
        **{
            "repo": "owner/repo",
            "number": 1,
            "head": "a" * 40,
            "base": "b" * 40,
            "reviewer": "reviewer",
            "diff_base": "c" * 40,
            "title": "Fix",
            "description": None,
            "files": [],
            "previous_comments": [],
        }
    )
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
            "body": (
                f"**{'Blocking' if blocking else 'Nonblocking'}: Problem**\n\nConcrete consequence."
            ),
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
    result = prepare(server.gh, "owner/repo", 1, "reviewer").model_dump()
    # Check revision-pinned contents and the requested reviewer identity.
    assert result["files"][0]["before"] == "first\nold\n"
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
        server.content["size"] = 8000001
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
    server.files[0].pop("patch", None)
    # Require preparation to leave that version empty rather than fetching it.
    assert prepare(server.gh, "owner/repo", 1, "reviewer").model_dump()["files"][0][empty] == ""


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


@pytest.mark.parametrize(
    "location",
    [
        "https://evil.example/steal",
        "http://api.github.com/repos/new/repo/issues/1",
        "https://api.github.com:8443/steal",
        "https://user@api.github.com/steal",
    ],
)
def test_redirects_cannot_expose_credentials(location: str) -> None:
    """Untrusted redirect destinations must never receive a request."""
    gh = GitHub("test-secret")

    def handle(request: httpx.Request) -> httpx.Response:
        """Only the original trusted request is permitted to reach the transport."""
        assert str(request.url) == "https://api.github.com/original"
        return httpx.Response(301, headers={"Location": location})

    gh.client = httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handle)
    )
    with pytest.raises(RuntimeError, match="trusted API host"):
        gh.request("GET", "/original")


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_redirect_loops_and_writes_are_rejected(method: str) -> None:
    """Redirect loops terminate; writes are never replayed at a new endpoint."""
    gh = GitHub("test-secret")
    gh.client = httpx.Client(
        base_url="https://api.github.com",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(301, headers={"Location": "/again"})
        ),
    )
    with pytest.raises(RuntimeError, match="redirect cannot be followed"):
        gh.request(method, "/original")


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
    assert ("RIGHT", 2) in diff_lines(context.files[0].patch)
    # A new API response still omits the patch; publication uses the captured revision.
    server.files = [{"filename": "app.py", "status": "added"}]
    _, reviews, result = bundle(server, blocking=True)
    assert publish(server.gh, context, reviews, result) is True
    assert server.writes[-1][1]["event"] == "REQUEST_CHANGES"
    assert server.writes[-1][1]["comments"][0]["line"] == 2


def test_symlink_context_contains_link_text_and_mode() -> None:
    """A symlink is reviewed as its own Git blob rather than its target's contents."""
    # The synthetic server exposes a link blob, with no Contents API fallback.
    server = Server()
    server.mode = "120000"
    server.content["content"] = base64.b64encode(b"relative/target.py").decode()
    del server.files[0]["patch"]
    server.files[0]["status"] = "added"
    context = prepare(server.gh, "owner/repo", 1, "reviewer")
    # Preserve mode and exact link bytes so the model can assess the actual change.
    assert context.files[0].after == "relative/target.py"
    assert context.files[0].after_mode == "120000"
    assert "+relative/target.py" in context.files[0].patch


def test_context_discards_repeated_api_metadata() -> None:
    """Repeated diff hunks and API metadata cannot consume the model context budget."""
    # Large irrelevant fields would breach the packet budget if copied verbatim.
    server = Server()
    server.files[0]["raw_url"] = "x" * 600001
    server.comments = [
        {
            "path": "app.py",
            "line": 2,
            "side": "RIGHT",
            "body": "Please explain this.",
            "user": {"login": "author", "avatar_url": "unused"},
            "diff_hunk": "x" * 600001,
        }
    ]
    context = prepare(server.gh, "owner/repo", 1, "reviewer")
    # Retain discussion and its source coordinates, with no opaque API passthrough.
    assert context.previous_comments[0].model_dump() == {
        "path": "app.py",
        "line": 2,
        "side": "RIGHT",
        "body": "Please explain this.",
        "author": "author",
    }
    assert "diff_hunk" not in context.model_dump_json()
    assert "raw_url" not in context.model_dump_json()


def test_git_line_boundaries_preserve_unicode_and_final_newline() -> None:
    """Unicode separators stay inside one Git line; newline-only changes remain visible."""
    # A Unicode separator followed by '+' must not become another added line.
    patch = rebuild_patch("old\n", "first\u2028+same-line\n")
    assert diff_lines(patch) == {("LEFT", 1), ("RIGHT", 1)}
    # An unterminated line and a terminated line are different Git contents.
    assert rebuild_patch("x", "x\n").endswith("@@ -1 +1 @@\n-x\n\\ No newline at end of file\n+x\n")


def test_grouped_findings_keep_distinct_locations_and_severity() -> None:
    """Grouping duplicates must not move a source to another line or hide its severity."""
    # Group observations on two valid lines with deliberately different severities.
    server = Server()
    context, reports, _ = bundle(server, blocking=True)
    reports[1].findings = [
        Finding(
            path="app.py",
            line=1,
            side="RIGHT",
            title="Observation",
            body="Other evidence",
            blocking=False,
        )
    ]
    result = Consolidation(
        summary="Related observations", groups=[Group(source_ids=["0:0", "1:0"])]
    )
    publish(server.gh, context, reports, result)
    # Each original location receives its own thread and an explicit severity label.
    comments = server.writes[-1][1]["comments"]
    assert [(c["line"], c["side"]) for c in comments] == [(2, "RIGHT"), (1, "RIGHT")]
    assert comments[0]["body"].startswith("**Blocking:")
    assert comments[1]["body"].startswith("**Nonblocking:")


def test_large_group_splits_without_losing_findings() -> None:
    """Valid maximum-sized Unicode findings never exceed the rendering byte budget."""
    # Eighty findings can exceed GitHub's body capacity when joined without bounds.
    server = Server()
    context, reports, _ = bundle(server, blocking=True)
    for reviewer, report in enumerate(reports):
        report.findings = [
            Finding(
                path="app.py",
                line=2,
                side="RIGHT",
                title=f"Issue {reviewer}:{i}",
                body="\U0001f680" * 4000,
                blocking=True,
            )
            for i in range(40)
        ]
    result = Consolidation(
        summary="Large group",
        groups=[Group(source_ids=[f"{reviewer}:{i}" for reviewer in range(2) for i in range(40)])],
    )
    publish(server.gh, context, reports, result)
    # Split only between findings, preserving every original body and valid anchor.
    comments = server.writes[-1][1]["comments"]
    assert len(comments) > 1
    assert all(0 < len(c["body"].encode()) <= 60000 for c in comments)
    assert sum(c["body"].count("\U0001f680") for c in comments) == 80 * 4000
    for reviewer in range(2):
        for i in range(40):
            assert (
                sum(c["body"].count(f"**Blocking: Issue {reviewer}:{i}**") for c in comments) == 1
            )


@pytest.mark.parametrize("invalid", ["truncated", "missing", "submodule", "directory-link"])
def test_blob_reader_rejects_incomplete_or_unsupported_trees(invalid: str) -> None:
    """Tree failures cannot substitute unrelated content for a changed path."""
    # Vary only the boundary that makes the path impossible to read correctly.
    entry = {"path": "app.py", "type": "blob", "mode": "100644", "sha": "d" * 40, "size": 1}
    if invalid == "submodule":
        entry.update(type="commit", mode="160000")
    data = {"truncated": invalid == "truncated", "tree": [] if invalid == "missing" else [entry]}
    gh = GitHub("test")
    gh.client = httpx.Client(
        base_url="https://api.github.com",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=data)),
    )
    # A blob or symlink cannot serve as an intermediate directory.
    path = "app.py/child" if invalid == "directory-link" else "app.py"
    with pytest.raises(ValueError):
        read_version(gh, "owner/repo", "root", path, {})


def test_blob_reader_walks_nested_tree_and_reuses_cache() -> None:
    """Nested files resolve through tree entries; cached entries remain sufficient later."""
    # Provide a real directory-to-blob relationship, with no Contents API responses.
    responses = {
        "root": {"tree": [{"path": "src", "type": "tree", "sha": "nested"}]},
        "nested": {
            "tree": [{"path": "app.py", "type": "blob", "mode": "100755", "sha": "blob", "size": 4}]
        },
        "blob": {"encoding": "base64", "size": 4, "content": "eCgpCg=="},
    }
    gh = GitHub("test")
    gh.client = httpx.Client(
        base_url="https://api.github.com",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=responses[request.url.path.rsplit("/", 1)[1]])
        ),
    )
    cache: dict = {}
    assert read_version(gh, "owner/repo", "root", "src/app.py", cache) == ("x()\n", "100755")
    # Removing server tree responses proves subsequent reads use the per-packet cache.
    del responses["root"], responses["nested"]
    assert read_version(gh, "owner/repo", "root", "src/app.py", cache) == ("x()\n", "100755")


def test_repetitive_blob_diff_does_not_invent_changed_lines() -> None:
    """Prepending one line leaves distant repeated lines outside the Git hunk."""
    before = "same\n" * 300
    patch = rebuild_patch(before, "new\n" + before)
    assert "@@ -1,3 +1,4 @@" in patch
    assert ("RIGHT", 200) not in diff_lines(patch)
    assert ("RIGHT", 1) in diff_lines(patch)


def test_preparation_accepts_text_blob_above_old_per_version_limit() -> None:
    """A 200 KB added text blob reaches planning rather than the old 150 KB rejection."""
    server = Server()
    content = (b"x" * 99 + b"\n") * 2000
    server.content = {
        "size": len(content),
        "encoding": "base64",
        "content": base64.b64encode(content).decode(),
    }
    server.files[0] = {"filename": "app.py", "status": "added"}
    context = prepare(server.gh, "owner/repo", 1, "reviewer")
    assert context.files[0].after.encode() == content
    assert context.files[0].before == ""


def test_preparation_accepts_more_than_eighty_changed_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The previous 80-file restriction no longer rejects a complete 125-file snapshot."""
    from ai_reviewer import context as collection

    server = Server()
    server.pr["changed_files"] = 125
    server.files = [
        {"filename": f"file-{i}.py", "status": "modified", "patch": PATCH} for i in range(125)
    ]
    monkeypatch.setattr(
        collection,
        "read_version",
        lambda gh, repo, tree, filename, cache: (
            "first\nold\n" if tree == "c" * 40 else "first\nnew\n",
            "100644",
        ),
    )
    context = prepare(server.gh, "owner/repo", 1, "reviewer")
    assert len(context.files) == 125
    assert {f.filename for f in context.files} == {f"file-{i}.py" for i in range(125)}
