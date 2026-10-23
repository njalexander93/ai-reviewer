"""Read immutable PR context and publish validated reviews through GitHub's REST API.

Repository content is never checked out or executed. All requests remain on the
fixed GitHub API host; untrusted paths are encoded, not interpolated as URLs.
"""

import base64
import re
from typing import Any
from urllib.parse import quote

import httpx

from ai_reviewer.contracts import Consolidation, Review, reconcile


class GitHub:
    """A bounded REST client with no automatic retries of write operations."""

    def __init__(self, token: str) -> None:
        """Bind credentials to GitHub only; never follow redirects with them."""
        self.client = httpx.Client(
            base_url="https://api.github.com",
            timeout=60,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Return JSON or raise a sanitized error without response bodies or tokens."""
        response = self.client.request(method, path, **kwargs)
        if response.is_error:
            raise RuntimeError(f"GitHub {method} failed: HTTP {response.status_code}")
        return response.json() if response.content else None

    def pages(self, path: str) -> list[dict[str, Any]]:
        """Read a bounded list completely; reject overflows instead of truncating."""
        items = []
        for page in range(1, 32):
            batch = self.request("GET", path, params={"per_page": 100, "page": page})
            items.extend(batch)
            if len(batch) < 100:
                return items
        raise ValueError("GitHub pagination limit exceeded")


def pr_path(repo: str, number: int) -> str:
    """Validate dispatch identifiers before building API paths."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or number < 1:
        raise ValueError("Invalid repository or PR number")
    return f"/repos/{repo}/pulls/{number}"


def diff_lines(patch: str) -> set[tuple[str, int]]:
    """Parse GitHub unified patches into valid LEFT and RIGHT inline anchors."""
    anchors: set[tuple[str, int]] = set()
    left = right = 0
    in_hunk = False
    for line in patch.splitlines():
        match = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if match:
            left, right = map(int, match.groups())
            in_hunk = True
        elif in_hunk and line.startswith("+"):
            anchors.add(("RIGHT", right))
            right += 1
        elif in_hunk and line.startswith("-"):
            anchors.add(("LEFT", left))
            left += 1
        elif in_hunk and line.startswith(" "):
            anchors.update({("LEFT", left), ("RIGHT", right)})
            left += 1
            right += 1
    return anchors


def prepare(gh: GitHub, repo: str, number: int, reviewer: str) -> dict[str, Any]:
    """Capture the diff, changed file contents and earlier review discussion.

    Raises:
        ValueError: PR is a draft, closed, self-authored by the reviewer, or too large.
    """
    path = pr_path(repo, number)
    pr = gh.request("GET", path)
    if pr["state"] != "open" or pr["draft"] or pr["user"]["login"] == reviewer:
        raise ValueError("PR is closed, draft, or authored by the reviewer account")
    files = gh.pages(f"{path}/files")
    if len(files) != pr["changed_files"] or len(files) > 80:
        raise ValueError("PR exceeds complete-context file limit")
    comparison = gh.request(
        "GET", f"/repos/{repo}/compare/{pr['base']['sha']}...{pr['head']['sha']}"
    )
    context: dict[str, Any] = {
        "repo": repo,
        "number": number,
        "head": pr["head"]["sha"],
        "base": pr["base"]["sha"],
        "diff_base": comparison["merge_base_commit"]["sha"],
        "reviewer": reviewer,
        "title": pr["title"],
        "description": pr["body"],
        "files": files,
        "previous_comments": gh.pages(f"{path}/comments"),
    }
    # Include complete changed-file versions, so findings can inspect surrounding logic.
    for file in files:
        if "patch" not in file:
            raise ValueError("Binary or unavailable diff; automatic approval prohibited")
        for side, ref, filename in [
            ("before", context["diff_base"], file.get("previous_filename", file["filename"])),
            ("after", context["head"], file["filename"]),
        ]:
            if (side == "before" and file["status"] == "added") or (
                side == "after" and file["status"] == "removed"
            ):
                file[side] = ""
                continue
            data = gh.request(
                "GET", f"/repos/{repo}/contents/{quote(filename, safe='')}", params={"ref": ref}
            )
            if data.get("encoding") != "base64" or data["size"] > 150_000:
                raise ValueError("Changed file exceeds context limit")
            file[side] = base64.b64decode(data["content"]).decode("utf-8")
    import json

    if len(json.dumps(context).encode()) > 600_000:
        raise ValueError("PR context exceeds 600 KB; split the change")
    latest = gh.request("GET", path)
    if latest["head"]["sha"] != context["head"] or latest["base"]["sha"] != context["base"]:
        raise ValueError("PR changed during context collection")
    gh.request("POST", f"{path}/requested_reviewers", json={"reviewers": [reviewer]})
    return context


def publish(
    gh: GitHub, context: dict[str, Any], reviews: list[Review], result: Consolidation
) -> bool:
    """Publish once per head/base pair, never approving incomplete or stale results.

    Returns:
        False if this revision was already published; True after a new submission.

    Raises:
        ValueError: Results, identity, diff anchors or current PR state are invalid.
    """
    groups = reconcile(reviews, result)
    repo, number = context["repo"], context["number"]
    path = pr_path(repo, number)
    login = gh.request("GET", "/user")["login"]
    if login != context["reviewer"]:
        raise ValueError("Publishing identity differs from requested reviewer")
    marker = f"<!-- ai-reviewer:{context['head']}:{context['base']} -->"
    existing = gh.pages(f"{path}/reviews")
    if any(r["user"]["login"] == login and marker in (r.get("body") or "") for r in existing):
        return False
    # Use fresh GitHub diff anchors, never trust locations carried by the model.
    files = gh.pages(f"{path}/files")
    anchors = {f["filename"]: diff_lines(f.get("patch", "")) for f in files}
    for review in reviews:
        for finding in review.findings:
            if (finding.side, finding.line) not in anchors.get(finding.path, set()):
                raise ValueError("Finding is outside the current diff")
    comments = []
    for group in groups:
        first = group[0]
        body = "\n\n---\n\n".join(f"**{f.title}**\n\n{f.body}" for f in group)
        comments.append({"path": first.path, "line": first.line, "side": first.side, "body": body})
    body = f"{marker}\n## AI review\n\n{result.summary}\n\n"
    for name, review in zip(["Claude Opus 5.5", "GPT-6.1 Sol"], reviews, strict=True):
        body += f"<details><summary>{name}</summary>\n\n{review.summary}\n\n</details>\n"
    blocking = any(f.blocking for review in reviews for f in review.findings)
    # Recheck immediately before the one write. GitHub has no conditional-review API.
    pr = gh.request("GET", path)
    if (
        pr["state"] != "open"
        or pr["draft"]
        or pr["head"]["sha"] != context["head"]
        or pr["base"]["sha"] != context["base"]
        or pr["user"]["login"] == login
    ):
        raise ValueError("Stale, closed, draft or self-authored PR; not publishing")
    gh.request(
        "POST",
        f"{path}/reviews",
        json={
            "commit_id": context["head"],
            "event": "REQUEST_CHANGES" if blocking else "APPROVE",
            "body": body,
            "comments": comments,
        },
    )
    return True
