"""Verify issue evidence at the HTTP boundary without live provider credentials."""

import json

import httpx
import pytest

from ai_reviewer.context import prepare
from ai_reviewer.issues import github_issues, linear_issues
from tests.test_github import Server


def test_linked_github_issue_reaches_review_context() -> None:
    """A linked issue supplies its requirements even without a textual PR reference."""
    server = Server()

    def handle(request: httpx.Request) -> httpx.Response:
        """Serve GitHub's linked-issue shape alongside the existing PR fixture."""
        if request.url.path == "/graphql":
            assert json.loads(request.content)["variables"]["number"] == 1
            return httpx.Response(
                200,
                json={
                    "data": {
                        "repository": {
                            "pullRequest": {
                                "closingIssuesReferences": {
                                    "nodes": [
                                        {
                                            "url": "https://github.com/owner/repo/issues/2",
                                            "title": "Retain history",
                                            "body": "Never delete saved edits.",
                                            "state": "OPEN",
                                        }
                                    ],
                                    "pageInfo": {"hasNextPage": False},
                                }
                            }
                        }
                    }
                },
            )
        return server.handle(request)

    server.gh.client = httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handle)
    )
    context = prepare(server.gh, "owner/repo", 1, "reviewer")
    assert [issue.model_dump() for issue in context.issues] == [
        {
            "provider": "github",
            "url": "https://github.com/owner/repo/issues/2",
            "title": "Retain history",
            "description": "Never delete saved edits.",
            "state": "open",
        }
    ]


@pytest.mark.parametrize(
    "github,linear", [(False, False), (True, False), (False, True), (True, True)]
)
def test_optional_providers_combine_without_duplicates(
    monkeypatch: pytest.MonkeyPatch, github: bool, linear: bool
) -> None:
    """All four requested combinations preserve each available issue once."""
    server = Server()
    server.pr["body"] = (
        "Related to #2, owner/repo#2 and https://github.com/owner/repo/issues/2" if github else ""
    )
    server.pr["head"]["ref"] = "author/eng-7-fix"
    client_type = httpx.Client

    def handle(request: httpx.Request) -> httpx.Response:
        """Return realistic provider shapes and reject unexpected external destinations."""
        if request.url.host == "api.linear.app":
            assert request.headers["Authorization"] == "linear-test"
            data = json.loads(request.content)
            if "teams(first:" in data["query"]:
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "teams": {"nodes": [{"key": "ENG"}], "pageInfo": {"hasNextPage": False}}
                        }
                    },
                )
            node = {
                "identifier": "ENG-7",
                "url": "https://linear.app/team/issue/ENG-7/fix",
                "title": "Retain edits",
                "description": "Keep prior edits.",
                "state": {"name": "Started"},
            }
            if "attachmentsForURL" in data["query"]:
                assert data["variables"] == {"url": "https://github.com/owner/repo/pull/1"}
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "attachmentsForURL": {
                                "nodes": [{"issue": node}],
                                "pageInfo": {"hasNextPage": False},
                            }
                        }
                    },
                )
            pytest.fail("An attached ticket must not be fetched again for its branch reference")
        assert request.url.host == "api.github.com"
        if request.url.path == "/repos/owner/repo/issues/2":
            return httpx.Response(
                200,
                json={
                    "html_url": "https://github.com/owner/repo/issues/2",
                    "title": "History",
                    "body": "Save all revisions.",
                    "state": "open",
                },
            )
        return server.handle(request)

    server.gh.client = client_type(
        base_url="https://api.github.com", transport=httpx.MockTransport(handle)
    )
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client_type(**kwargs, transport=httpx.MockTransport(handle)),
    )
    context = prepare(server.gh, "owner/repo", 1, "reviewer", "linear-test" if linear else None)
    expected = []
    if github:
        expected.append(("github", "History", "Save all revisions."))
    if linear:
        expected.append(("linear", "Retain edits", "Keep prior edits."))
    assert [(i.provider, i.title, i.description) for i in context.issues] == expected


@pytest.mark.parametrize(
    "reference", ["ENG-8", "https://linear.app/team/issue/ENG-8/fix", "author/eng-8-fix"]
)
def test_linear_reference_without_attachment(
    monkeypatch: pytest.MonkeyPatch, reference: str
) -> None:
    """Identifiers, links and lowercase branch IDs resolve without an attachment."""
    client_type = httpx.Client

    def handle(request: httpx.Request) -> httpx.Response:
        """Serve an unattached issue through the documented issue(id:) query."""
        data = json.loads(request.content)
        if "teams(first:" in data["query"]:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "teams": {"nodes": [{"key": "ENG"}], "pageInfo": {"hasNextPage": False}}
                    }
                },
            )
        if "attachmentsForURL" in data["query"]:
            return httpx.Response(
                200,
                json={
                    "data": {"attachmentsForURL": {"nodes": [], "pageInfo": {"hasNextPage": False}}}
                },
            )
        assert data["variables"] == {"id": "ENG-8"}
        return httpx.Response(
            200,
            json={
                "data": {
                    "issue": {
                        "identifier": "ENG-8",
                        "url": "https://linear.app/team/issue/ENG-8/fix",
                        "title": "Fix persistence",
                        "description": None,
                        "state": {"name": "Todo"},
                    }
                }
            },
        )

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client_type(**kwargs, transport=httpx.MockTransport(handle)),
    )
    issues = linear_issues("test", "https://github.com/owner/repo/pull/1", reference)
    assert [(i.title, i.description, i.state) for i in issues] == [
        ("Fix persistence", None, "Todo")
    ]


@pytest.mark.parametrize("provider", ["github", "linear"])
@pytest.mark.parametrize("failure", ["http", "graphql", "pagination", "oversize"])
def test_issue_failure_never_requests_review(
    monkeypatch: pytest.MonkeyPatch, provider: str, failure: str
) -> None:
    """Unavailable or oversized issue evidence fails before review side effects."""
    server = Server()
    client_type = httpx.Client

    def handle(request: httpx.Request) -> httpx.Response:
        """Inject failures only at the selected provider's issue boundary."""
        if request.url.path != "/graphql" or (request.url.host == "api.linear.app") != (
            provider == "linear"
        ):
            return server.handle(request)
        if failure == "http":
            return httpx.Response(403, text="secret-content")
        if failure == "graphql":
            return httpx.Response(200, json={"errors": [{"message": "secret-content"}], "data": {}})
        node = {
            "url": "https://github.com/owner/repo/issues/2",
            "title": "Issue",
            "body": "x" * 600001,
            "state": "open",
        }
        connection = {"nodes": [node], "pageInfo": {"hasNextPage": failure == "pagination"}}
        if provider == "linear":
            node = {
                "identifier": "ENG-1",
                "url": "https://linear.app/team/issue/ENG-1",
                "title": "Issue",
                "description": "x" * 600001,
                "state": {"name": "Todo"},
            }
            connection["nodes"] = [{"issue": node}]
            return httpx.Response(200, json={"data": {"attachmentsForURL": connection}})
        return httpx.Response(
            200,
            json={"data": {"repository": {"pullRequest": {"closingIssuesReferences": connection}}}},
        )

    server.gh.client = client_type(
        base_url="https://api.github.com", transport=httpx.MockTransport(handle)
    )
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client_type(**kwargs, transport=httpx.MockTransport(handle)),
    )
    with pytest.raises((RuntimeError, ValueError)) as error:
        prepare(server.gh, "owner/repo", 1, "reviewer", "test" if provider == "linear" else None)
    assert "secret-content" not in str(error.value)
    assert server.writes == []


def test_github_references_exclude_prs_and_foreign_hosts() -> None:
    """A PR reference is not an issue, and arbitrary URLs never receive credentials."""
    server = Server()
    server.linked_issues = [
        {
            "url": "https://github.com/owner/repo/issues/2",
            "title": "Linked",
            "body": "Requirement",
            "state": "OPEN",
        }
    ]

    def handle(request: httpx.Request) -> httpx.Response:
        """Serve only the linked issue query and the other referenced PR."""
        assert request.url.host == "api.github.com"
        if request.url.path == "/repos/other/repo/issues/3":
            return httpx.Response(200, json={"pull_request": {"url": "unused"}})
        assert request.url.path == "/graphql"
        return server.handle(request)

    server.gh.client = httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handle)
    )
    issues = github_issues(
        server.gh, "owner/repo", 1, "#1 #2 other/repo#3 https://evil.example/owner/repo/issues/9"
    )
    assert [(i.title, i.description) for i in issues] == [("Linked", "Requirement")]


@pytest.mark.parametrize(
    "text",
    [
        "Use CSS colours #333 and #123456.",
        "See https://docs.example.com/page#3 and docs.example.com/page#4.",
        "https://docs.example.com/page?next=other/repo#5",
    ],
)
def test_prose_numbers_and_url_fragments_are_not_issue_references(text: str) -> None:
    """Ordinary prose must not fetch unrelated issues or abort a review."""
    server = Server()

    def handle(request: httpx.Request) -> httpx.Response:
        """Allow the authoritative linked-issue lookup, but no guessed issue fetch."""
        assert request.url.path == "/graphql"
        return server.handle(request)

    server.gh.client = httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handle)
    )
    assert github_issues(server.gh, "owner/repo", 1, text) == []


@pytest.mark.parametrize("text", ["Fixes #2", "Related to #2", "issue #2", "owner/repo#2"])
def test_explicit_issue_references_still_resolve(text: str) -> None:
    """Tightened parsing preserves intentional issue references and normalizes state."""
    server = Server()

    def handle(request: httpx.Request) -> httpx.Response:
        """Return a real issue only for the expected repository and number."""
        if request.url.path == "/repos/owner/repo/issues/2":
            return httpx.Response(
                200,
                json={
                    "html_url": "https://github.com/owner/repo/issues/2",
                    "title": "Preserve edits",
                    "body": "Keep history",
                    "state": "open",
                },
            )
        assert request.url.path == "/graphql"
        return server.handle(request)

    server.gh.client = httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handle)
    )
    assert [(i.title, i.state) for i in github_issues(server.gh, "owner/repo", 1, text)] == [
        ("Preserve edits", "open")
    ]


def test_linear_enabled_without_tickets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unrelated hyphenated terms do not invent Linear tickets."""
    client_type = httpx.Client

    def handle(request: httpx.Request) -> httpx.Response:
        """Expose one team and no attachments; reject spurious issue queries."""
        document = json.loads(request.content)["query"]
        if "teams(first:" in document:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "teams": {"nodes": [{"key": "ENG"}], "pageInfo": {"hasNextPage": False}}
                    }
                },
            )
        assert "attachmentsForURL" in document
        return httpx.Response(
            200,
            json={"data": {"attachmentsForURL": {"nodes": [], "pageInfo": {"hasNextPage": False}}}},
        )

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client_type(**kwargs, transport=httpx.MockTransport(handle)),
    )
    assert (
        linear_issues("test", "https://github.com/owner/repo/pull/1", "Fix UTF-8 and SHA-256") == []
    )


@pytest.mark.parametrize("provider", ["github", "linear"])
def test_reference_budget_rejects_before_issue_fetches(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    """A PR with 21 explicit references cannot cause unbounded issue fetches."""
    server = Server()
    client_type = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client_type(
            **kwargs,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "data": {
                            "teams": {"nodes": [{"key": "ENG"}], "pageInfo": {"hasNextPage": False}}
                        }
                    },
                )
            ),
        ),
    )
    with pytest.raises(ValueError, match="20 references"):
        if provider == "github":
            github_issues(server.gh, "owner/repo", 1, " ".join(f"refs #{i}" for i in range(2, 23)))
        else:
            linear_issues("test", "unused", " ".join(f"ENG-{i}" for i in range(1, 22)))
