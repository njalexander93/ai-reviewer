"""Read immutable PR context and publish validated reviews through GitHub's REST API.

Repository content is never checked out or executed. All requests remain on the
fixed GitHub API host; untrusted paths are encoded, not interpolated as URLs.
"""

import base64
import re
from difflib import unified_diff
from typing import Any
from urllib.parse import quote

import httpx

from ai_reviewer.contracts import Consolidation, Review, reconcile


class GitHub:
    """Make bounded GitHub REST requests without automatic write retries.

    Attributes:
        client: Authenticated HTTP client with a 60-second timeout and redirects
            disabled. Callers must supply GitHub-relative paths to its methods.
    """

    def __init__(self, token: str) -> None:
        """Configure the HTTP client without making a network request.

        Args:
            token: GitHub credential for the operations this job performs.
        """
        # Keep credentials on the configured host when callers use relative paths.
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
        """Send one request and decode a nonempty response as JSON.

        Args:
            method: HTTP method, including mutating methods for publication.
            path: GitHub API path relative to the configured host.
            **kwargs: HTTPX request options, such as query parameters or JSON data.

        Returns:
            Decoded JSON, or None when the response has no body.

        Raises:
            RuntimeError: GitHub returns a 4xx or 5xx status; the message omits bodies.
            httpx.HTTPError: A transport or timeout failure prevents completion.
            ValueError: A nonempty response cannot be decoded as JSON.
        """
        # Perform the request once; retrying a write could duplicate a review.
        response = self.client.request(method, path, **kwargs)
        # Reject HTTP failures without exposing GitHub response content.
        if response.is_error:
            raise RuntimeError(f"GitHub {method} failed: HTTP {response.status_code}")
        # Preserve empty responses rather than attempting to decode them.
        return response.json() if response.content else None

    def pages(self, path: str) -> list[dict[str, Any]]:
        """Collect a paginated GitHub list, with at most 31 requests.

        Args:
            path: Relative API endpoint returning a list with page/per_page support.

        Returns:
            Items in API order, once a page contains fewer than 100 entries.

        Raises:
            ValueError: All 31 pages are full, so completeness cannot be established.
            RuntimeError: GitHub rejects a request; transport errors also propagate.
        """
        # Accumulate pages until the endpoint proves there is no next full page.
        items = []
        for page in range(1, 32):
            batch = self.request("GET", path, params={"per_page": 100, "page": page})
            items.extend(batch)
            # A short page establishes that the list is complete.
            if len(batch) < 100:
                return items
        # Do not return partial evidence when the request budget is exhausted.
        raise ValueError("GitHub pagination limit exceeded")


def pr_path(repo: str, number: int) -> str:
    """Build a PR endpoint from dispatch identifiers.

    Args:
        repo: Owner/repository pair containing only letters, digits, _, . or -.
        number: Positive pull request number.

    Returns:
        Relative GitHub REST path for the pull request.

    Raises:
        ValueError: The repository pair or PR number fails validation.
    """
    # Reject malformed dispatch identifiers before interpolating an API path.
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or number < 1:
        raise ValueError("Invalid repository or PR number")
    # Construct the endpoint only from validated identifiers.
    return f"/repos/{repo}/pulls/{number}"


def diff_lines(patch: str) -> set[tuple[str, int]]:
    """Extract inline coordinates from GitHub unified-diff hunks.

    Args:
        patch: Unified patch text; content outside recognized hunks is ignored.

    Returns:
        Pairs of side and one-based line number. Context lines occur on both
        sides, deletions on LEFT and additions on RIGHT. No hunks means no anchors.
        This parser extracts coordinates; it does not validate patch completeness.
    """
    # Track old and new line counters independently across each hunk.
    anchors: set[tuple[str, int]] = set()
    left = right = 0
    in_hunk = False
    # Reset counters at hunk headers and advance the side consumed by each line.
    for line in patch.splitlines():
        match = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        # Headers establish coordinates; file headers and metadata add no anchors.
        if match:
            left, right = map(int, match.groups())
            in_hunk = True
        # Added lines exist only in the new file.
        elif in_hunk and line.startswith("+"):
            anchors.add(("RIGHT", right))
            right += 1
        # Deleted lines exist only in the old file.
        elif in_hunk and line.startswith("-"):
            anchors.add(("LEFT", left))
            left += 1
        # Unchanged context advances both files at once.
        elif in_hunk and line.startswith(" "):
            anchors.update({("LEFT", left), ("RIGHT", right)})
            left += 1
            right += 1
    # Collapse repeated coordinates across overlapping hunks.
    return anchors


def prepare(gh: GitHub, repo: str, number: int, reviewer: str) -> dict[str, Any]:
    """Capture the diff, changed file contents and earlier review discussion.

    Reads GitHub content and requests the reviewer only after collection succeeds.
    It never executes repository code or inspects unchanged dependency files.

    Args:
        gh: Client authorized to read the repository and request reviewers.
        repo: Owner/repository pair accepted by pr_path.
        number: Positive pull request number.
        reviewer: Login to request; must differ from the PR author.

    Returns:
        JSON-serializable packet with revision identifiers, PR metadata, prior
        inline comments and changed files augmented with before/after text.

    Raises:
        ValueError: Identifiers or PR state are invalid, context is incomplete or
            oversized, or head/base changes during collection. Invalid text
            decoding also fails rather than returning partial context.
        RuntimeError: GitHub rejects a request; transport errors also propagate.
    """
    # Establish eligibility before collecting potentially expensive context.
    path = pr_path(repo, number)
    pr = gh.request("GET", path)
    if pr["state"] != "open" or pr["draft"] or pr["user"]["login"] == reviewer:
        raise ValueError("PR is closed, draft, or authored by the reviewer account")
    # Require a complete, bounded file list and resolve the diff's merge base.
    files = gh.pages(f"{path}/files")
    if len(files) != pr["changed_files"] or len(files) > 80:
        raise ValueError("PR exceeds complete-context file limit")
    comparison = gh.request(
        "GET", f"/repos/{repo}/compare/{pr['base']['sha']}...{pr['head']['sha']}"
    )
    # Bind discussion and file evidence to the observed head and base revisions.
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
        # Read each existing version at its immutable commit, respecting renames.
        for side, ref, filename in [
            ("before", context["diff_base"], file.get("previous_filename", file["filename"])),
            ("after", context["head"], file["filename"]),
        ]:
            # Added and removed files have one deliberately empty version.
            if (side == "before" and file["status"] == "added") or (
                side == "after" and file["status"] == "removed"
            ):
                file[side] = ""
                continue
            # Bound file contents before decoding them into model context.
            data = gh.request(
                "GET", f"/repos/{repo}/contents/{quote(filename, safe='')}", params={"ref": ref}
            )
            if data.get("encoding") != "base64" or data["size"] > 150_000:
                raise ValueError("Changed file exceeds context limit")
            file[side] = base64.b64decode(data["content"]).decode("utf-8")
            # NUL-containing files are binary even when their bytes decode as UTF-8.
            if "\0" in file[side]:
                raise ValueError("Binary file; automatic approval prohibited")
        # GitHub can omit text patches, including generated lockfiles; rebuild from exact refs.
        if "patch" not in file:
            file["patch"] = "\n".join(
                unified_diff(file["before"].splitlines(), file["after"].splitlines(), lineterm="")
            )
    import json

    # Bound the entire packet, including description and prior discussion.
    if len(json.dumps(context).encode()) > 600_000:
        raise ValueError("PR context exceeds 600 KB; split the change")
    # Reject a collection that raced a head or base update.
    latest = gh.request("GET", path)
    if latest["head"]["sha"] != context["head"] or latest["base"]["sha"] != context["base"]:
        raise ValueError("PR changed during context collection")
    # Make the reviewer request only after the packet passes validation.
    gh.request("POST", f"{path}/requested_reviewers", json={"reviewers": [reviewer]})
    return context


def publish(
    gh: GitHub, context: dict[str, Any], reviews: list[Review], result: Consolidation
) -> bool:
    """Submit a consolidated review after checking evidence and current PR state.

    The final state check and submission are separate API calls. A concurrent
    push can still leave the review attached to an old commit; callers should
    enable stale approval dismissal in repository branch rules.

    Args:
        gh: Client authenticated as the reviewer recorded in context.
        context: Packet from prepare, including repository, PR and revision IDs.
        reviews: Exactly two complete reports, ordered Claude then OpenAI.
        result: Consolidated summary and an exact partition of source findings.

    Returns:
        False if this revision was already published; True after a new submission.

    Raises:
        ValueError: Results, identity, diff anchors or current PR state are invalid.
        RuntimeError: GitHub rejects a request; transport errors also propagate.
    """
    # Validate both reports and ensure the publishing credential owns the request.
    groups = reconcile(reviews, result)
    repo, number = context["repo"], context["number"]
    path = pr_path(repo, number)
    login = gh.request("GET", "/user")["login"]
    if login != context["reviewer"]:
        raise ValueError("Publishing identity differs from requested reviewer")
    # Avoid another submission when this account already reviewed the revision pair.
    marker = f"<!-- ai-reviewer:{context['head']}:{context['base']} -->"
    existing = gh.pages(f"{path}/reviews")
    if any(r["user"]["login"] == login and marker in (r.get("body") or "") for r in existing):
        return False
    # Use fresh GitHub diff anchors, never trust locations carried by the model.
    files = gh.pages(f"{path}/files")
    # Reuse reconstructed patches only from the trusted packet for the checked revision pair.
    captured = {f["filename"]: f["patch"] for f in context.get("files", [])}
    anchors = {
        f["filename"]: diff_lines(f.get("patch", captured.get(f["filename"], ""))) for f in files
    }
    for review in reviews:
        for finding in review.findings:
            if (finding.side, finding.line) not in anchors.get(finding.path, set()):
                raise ValueError("Finding is outside the current diff")
    # Keep original evidence and severity while combining duplicate threads.
    comments = []
    for group in groups:
        first = group[0]
        body = "\n\n---\n\n".join(f"**{f.title}**\n\n{f.body}" for f in group)
        comments.append({"path": first.path, "line": first.line, "side": first.side, "body": body})
    # Attach both independent summaries beneath the consolidated overview.
    body = f"{marker}\n## AI review\n\n{result.summary}\n\n"
    for name, review in zip(["Claude Opus 5.5", "GPT-6.1 Sol"], reviews, strict=True):
        body += f"<details><summary>{name}</summary>\n\n{review.summary}\n\n</details>\n"
    # Derive the verdict from original findings, never from Sonnet's prose.
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
    # Submit the summary, verdict and inline comments in one review operation.
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
