"""Collect bounded issue requirements from GitHub links and explicit PR references.

Only fixed provider API hosts receive credentials. Missing Linear credentials skip
Linear; failed or truncated lookups raise rather than producing a clean review.
Issue bodies are evidence, never instructions. Project boards are not traversed.
"""

import re
from typing import Any

import httpx

from ai_reviewer.contracts import IssueContext
from ai_reviewer.github import GitHub


def github_issues(gh: GitHub, repo: str, number: int, text: str) -> list[IssueContext]:
    """Read linked issues, keyword-qualified #N, owner/repo#N and issue URLs.

    Args:
        gh: Existing authenticated GitHub client, used only for reads here.
        repo: Validated PR repository, used for unqualified references.
        number: Positive PR number.
        text: Untrusted PR title and body to scan for explicit references.

    Returns:
        Deduplicated issue titles, bodies, states and source URLs.

    Raises:
        RuntimeError: GitHub rejects an HTTP or GraphQL query.
        ValueError: More than 20 issues/references are present or data is invalid.
        httpx.HTTPError: A transport failure prevents complete collection.
    """
    # Include manual Development links even when no issue appears in the PR body.
    owner, name = repo.split("/")
    result = gh.request(
        "POST",
        "/graphql",
        json={
            "query": """query($owner: String!, $name: String!, $number: Int!) {
          repository(owner: $owner, name: $name) { pullRequest(number: $number) {
            closingIssuesReferences(first: 21) {
              nodes { url title body state }
              pageInfo { hasNextPage }
            }
          } }
        }""",
            "variables": {"owner": owner, "name": name, "number": number},
        },
    )
    if result.get("errors"):
        raise RuntimeError("GitHub issue context query failed")
    linked = result["data"]["repository"]["pullRequest"]["closingIssuesReferences"]
    if linked["pageInfo"]["hasNextPage"] or len(linked["nodes"]) > 20:
        raise ValueError("GitHub issue context exceeds 20 issues")
    issues = {
        node["url"].lower(): IssueContext(
            provider="github",
            url=node["url"],
            title=node["title"],
            description=node["body"],
            state=node["state"].lower(),
        )
        for node in linked["nodes"]
    }
    # Never follow user-provided URLs; extract only validated repository/number pairs.
    references = {
        (match[0].lower(), int(match[1]))
        for match in re.findall(
            r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/issues/([1-9][0-9]*)\b",
            text,
        )
    }
    shorthand = re.sub(r"https?://[^\s<>]+", "", text)
    references.update(
        (match[0].lower(), int(match[1]))
        for match in re.findall(
            r"(?<![\w/.:#])([A-Za-z0-9-]+/[A-Za-z0-9_-][A-Za-z0-9_.-]*)"
            r"#([1-9][0-9]*)\b",
            shorthand,
        )
    )
    references.update(
        (repo.lower(), int(issue_number))
        for issue_number in re.findall(
            r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?|refs?|references?|issue|related to)"
            r"\s+#([1-9][0-9]*)\b",
            shorthand,
            re.IGNORECASE,
        )
    )
    if len(references) > 20:
        raise ValueError("GitHub issue context exceeds 20 references")
    for target, issue_number in sorted(references):
        url = f"https://github.com/{target}/issues/{issue_number}"
        if url in issues or (target == repo.lower() and issue_number == number):
            continue
        node = gh.request("GET", f"/repos/{target}/issues/{issue_number}")
        # GitHub's Issues endpoint also returns PRs; they are not issue requirements.
        if "pull_request" in node:
            continue
        issues[node["html_url"].lower()] = IssueContext(
            provider="github",
            url=node["html_url"],
            title=node["title"],
            description=node["body"],
            state=node["state"].lower(),
        )
        if len(issues) > 20:
            raise ValueError("GitHub issue context exceeds 20 issues")
    return list(issues.values())


def linear_issues(token: str | None, pr_url: str, text: str) -> list[IssueContext]:
    """Read Linear PR attachments and issue IDs in the PR title, body or branch.

    Args:
        token: Personal API key; None or empty disables Linear without requests.
        pr_url: Canonical GitHub PR URL to look up as a Linear attachment.
        text: Untrusted metadata containing identifiers such as ENG-123.

    Returns:
        Deduplicated issue evidence, empty if disabled or no issues are found.

    Raises:
        RuntimeError: HTTP/GraphQL fails or a referenced issue is unavailable.
        ValueError: More than 20 issues/references exist or response data is invalid.
        httpx.HTTPError: A transport failure prevents complete collection.
    """
    if not token:
        return []
    identifiers = sorted(set(re.findall(r"\b[A-Za-z][A-Za-z0-9]*-[1-9][0-9]*\b", text.upper())))
    # One fixed host receives the key; identifiers are GraphQL variables, never query text.
    fields = "identifier url title description state { name }"
    with httpx.Client(timeout=60, headers={"Authorization": token}) as client:

        def query(document: str, variables: dict[str, Any]) -> Any:
            """Read Linear data once, rejecting partial errors without logging bodies."""
            response = client.post(
                "https://api.linear.app/graphql",
                json={
                    "query": document,
                    "variables": variables,
                },
            )
            if response.is_error:
                raise RuntimeError(f"Linear issue context failed: HTTP {response.status_code}")
            result = response.json()
            if result.get("errors"):
                raise RuntimeError("Linear issue context query failed")
            return result["data"]

        # Match workspace team keys so terms such as UTF-8 are not treated as tickets.
        if identifiers:
            teams = query(
                "query { teams(first: 100) { nodes { key } pageInfo { hasNextPage } } }", {}
            )["teams"]
            if teams["pageInfo"]["hasNextPage"]:
                raise ValueError("Linear team lookup is incomplete")
            keys = {team["key"] for team in teams["nodes"]}
            identifiers = [item for item in identifiers if item.rsplit("-", 1)[0] in keys]
            if len(identifiers) > 20:
                raise ValueError("Linear issue context exceeds 20 references")
        # Linear's SDK schema defines first: Int and returns AttachmentConnection:
        # https://github.com/linear/linear/blob/master/packages/sdk/src/schema.graphql
        attached = query(
            "query($url: String!) { attachmentsForURL(url: $url, first: 21) { "
            f"nodes {{ issue {{ {fields} }} }} pageInfo {{ hasNextPage }} }} }}",
            {"url": pr_url},
        )["attachmentsForURL"]
        if attached["pageInfo"]["hasNextPage"] or len(attached["nodes"]) > 20:
            raise ValueError("Linear issue context exceeds 20 attachments")
        nodes = {node["issue"]["identifier"]: node["issue"] for node in attached["nodes"]}
        for identifier in identifiers:
            if identifier in nodes:
                continue
            node = query(
                f"query($id: String!) {{ issue(id: $id) {{ {fields} }} }}", {"id": identifier}
            )["issue"]
            if node is None:
                raise RuntimeError("Referenced Linear issue unavailable")
            nodes[node["identifier"]] = node
            if len(nodes) > 20:
                raise ValueError("Linear issue context exceeds 20 issues")
    return [
        IssueContext(
            provider="linear",
            url=node["url"],
            title=node["title"],
            description=node["description"],
            state=node["state"]["name"],
        )
        for node in nodes.values()
    ]
