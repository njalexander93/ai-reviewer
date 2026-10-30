"""Read immutable PR context and publish validated reviews through GitHub's REST API.

Repository content is never checked out or executed. All requests remain on the
fixed GitHub API host; untrusted paths are encoded, not interpolated as URLs.
"""

import base64
import json
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx

from ai_reviewer.contracts import (
    Consolidation,
    CoveredReviews,
    ReviewContext,
    reconcile,
)
from ai_reviewer.limits import (
    GITHUB_BODY_BYTES,
    MAX_BLOB_BYTES,
    MAX_GITHUB_COMMENTS,
    MAX_GITHUB_PAYLOAD_BYTES,
    MAX_GITHUB_REQUESTS,
)
from ai_reviewer.patch import diff_lines
from ai_reviewer.serialization import digest


class GitHub:
    """Make bounded GitHub REST requests without automatic write retries.

    Attributes:
        requests_remaining: Per-client cap on requests, including pagination and redirects.
        client: Authenticated HTTP client with a 60-second timeout and redirects
            disabled. Callers must supply GitHub-relative paths to its methods.
    """

    def __init__(self, token: str) -> None:
        """Configure the HTTP client without making a network request.

        Args:
            token: GitHub credential for the operations this job performs.
        """
        self.requests_remaining = MAX_GITHUB_REQUESTS
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
        """Send a request, following at most five same-host redirects for reads.

        Args:
            method: HTTP method, including mutating methods for publication.
            path: GitHub API path relative to the configured host.
            **kwargs: HTTPX request options, such as query parameters or JSON data.

        Returns:
            Decoded JSON, or None when the response has no body.

        Raises:
            RuntimeError: GitHub fails, redirects a write, exceeds five redirects,
                or redirects outside its HTTPS API host. Messages omit response bodies.
            httpx.HTTPError: A transport or timeout failure prevents completion.
            ValueError: Request budget is exhausted or a nonempty response is invalid JSON.
        """
        # Follow moved read resources without forwarding credentials outside GitHub.
        for redirect in range(6):
            if self.requests_remaining <= 0:
                raise ValueError("GitHub collection request budget exceeded")
            self.requests_remaining -= 1
            response = self.client.request(method, path, **kwargs)
            if not response.is_redirect:
                break
            if method != "GET" or redirect == 5 or "location" not in response.headers:
                raise RuntimeError("GitHub redirect cannot be followed safely")
            target = response.url.join(response.headers["location"])
            if (
                target.scheme != "https"
                or target.host != "api.github.com"
                or target.port not in (None, 443)
                or target.username
                or target.password
            ):
                raise RuntimeError("GitHub redirect leaves the trusted API host")
            path = str(target)
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


def read_version(
    gh: GitHub, repo: str, tree: str, filename: str, cache: dict[str, Any]
) -> tuple[str, str]:
    """Read an immutable Git blob without dereferencing symbolic links.

    Args:
        gh: GitHub client with repository read access.
        repo: Validated owner/repository pair.
        tree: Root tree SHA for the selected commit.
        filename: Repository-relative path to walk through tree entries.
        cache: Per-preparation cache of complete tree responses, mutated on misses.

    Returns:
        Exact UTF-8 blob text and Git mode, including symlink target text for 120000.

    Raises:
        ValueError: A tree is truncated, a path is missing, the object is unsupported,
            or file bytes exceed the limit, contain NUL or are not UTF-8.
        RuntimeError: GitHub rejects a request; transport errors also propagate.
    """
    # Walk nonrecursive trees so large unrelated subtrees do not truncate the lookup.
    parts = filename.split("/")
    for index, part in enumerate(parts):
        if tree not in cache:
            cache[tree] = gh.request("GET", f"/repos/{repo}/git/trees/{tree}")
        data = cache[tree]
        if data.get("truncated"):
            raise ValueError("Truncated Git tree; context incomplete")
        entry = next((item for item in data["tree"] if item["path"] == part), None)
        if entry is None:
            raise ValueError("Changed path missing from Git tree")
        # Only directory entries may continue the path traversal.
        if index < len(parts) - 1:
            if entry["type"] != "tree":
                raise ValueError("Changed path traverses a non-directory")
            tree = entry["sha"]
            continue
        # Bound and classify the object before requesting its bytes.
        if entry["type"] != "blob" or entry["mode"] not in ("100644", "100755", "120000"):
            raise ValueError("Unsupported Git object")
        if entry["size"] > MAX_BLOB_BYTES:
            raise ValueError("Changed file exceeds context limit")
        blob = gh.request("GET", f"/repos/{repo}/git/blobs/{entry['sha']}")
        if blob.get("encoding") != "base64" or blob["size"] > MAX_BLOB_BYTES:
            raise ValueError("Unavailable or oversized Git blob")
        raw = base64.b64decode(blob["content"])
        if len(raw) > MAX_BLOB_BYTES:
            raise ValueError("Decoded blob exceeds collection budget")
        text = raw.decode("utf-8")
        if "\0" in text:
            raise ValueError("Binary file; automatic approval prohibited")
        return text, entry["mode"]
    raise ValueError("Empty changed-file path")


def rebuild_patch(before: str, after: str) -> str:
    """Reconstruct a unified patch using only LF as a physical line boundary.

    Args:
        before: Exact old blob text, including its final-newline state.
        after: Exact new blob text, including its final-newline state.

    Returns:
        Unified diff with explicit markers for unterminated lines. Unicode
        separators and carriage returns remain content rather than line boundaries.

    Raises:
        RuntimeError: Git fails to produce the diff.
        OSError: Temporary files or the Git executable are unavailable.
        subprocess.TimeoutExpired: Git exceeds the 30-second limit.
    """
    # Diff inert temporary blobs; disable external diff drivers and text conversion.
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "before").write_bytes(before.encode("utf-8"))
        (root / "after").write_bytes(after.encode("utf-8"))
        result = subprocess.run(
            [
                "git",
                "diff",
                "--no-index",
                "--no-ext-diff",
                "--no-textconv",
                "--text",
                "--no-color",
                "--diff-algorithm=myers",
                "--no-indent-heuristic",
                "--unified=3",
                "--",
                "before",
                "after",
            ],
            cwd=root,
            capture_output=True,
            timeout=30,
            check=False,
        )
        # Git uses exit 1 for a successful diff containing changes.
        if result.returncode not in (0, 1):
            raise RuntimeError("Git could not reconstruct the patch")
        return result.stdout.decode("utf-8")


def publish(
    gh: GitHub,
    context: ReviewContext,
    covered: CoveredReviews,
    result: Consolidation,
    *,
    cost_report: str = "",
) -> bool:
    """Submit a consolidated review after checking evidence and current PR state.

    The final state check and submission are separate API calls. A concurrent
    push can still leave the review attached to an old commit; callers should
    enable stale approval dismissal in repository branch rules.
    Callers must serialize publication per PR to prevent concurrent duplicate writes.

    Args:
        gh: Client authenticated as the reviewer recorded in context.
        context: Packet from prepare, including repository, PR and revision IDs.
        covered: Reports and coverage summary from the canonical artifact loader.
        result: Consolidated summary and an exact partition of source findings.
        cost_report: Optional trusted plain-text accounting to append beneath coverage.

    Returns:
        False if this request was already published; True after a new submission.

    Raises:
        ValueError: Results, identity, diff anchors or current PR state are invalid,
            or the assembled review exceeds comment-count/body/payload bounds.
        RuntimeError: GitHub rejects a request; transport errors also propagate.
    """
    # Coverage was replayed once by the canonical loader; bind it to this snapshot.
    if covered.snapshot != digest(context.model_dump_json()):
        raise ValueError("Publication snapshot differs from covered work")
    reviews = covered.reviews
    groups = reconcile(reviews, result)
    repo, number = context.repo, context.number
    path = pr_path(repo, number)
    login = gh.request("GET", "/user")["login"]
    if login != context.reviewer:
        raise ValueError("Publishing identity differs from requested reviewer")
    # Retrying one request must not suppress a fresh review of the same commits.
    marker = f"<!-- ai-reviewer:{context.head}:{context.base}:{context.request_id} -->"
    existing = gh.pages(f"{path}/reviews")
    if any(r["user"]["login"] == login and marker in (r.get("body") or "") for r in existing):
        return False
    # Use fresh GitHub diff anchors, never trust locations carried by the model.
    files = gh.pages(f"{path}/files")
    # Reuse reconstructed patches only from the trusted packet for the checked revision pair.
    captured = {f.filename: f.patch for f in context.files}
    anchors = {
        f["filename"]: diff_lines(f.get("patch", captured.get(f["filename"], ""))) for f in files
    }
    for review in reviews:
        for finding in review.findings:
            if (finding.side, finding.line) not in anchors.get(finding.path, set()):
                raise ValueError("Finding is outside the current diff")
    # Preserve source locations and severity; split only between complete findings.
    comments = []
    for group in groups:
        by_anchor: dict[tuple[str, str, int], list[str]] = {}
        for finding in group:
            label = "Blocking" if finding.blocking else "Nonblocking"
            fragment = f"**{label}: {finding.title}**\n\n{finding.body}"
            by_anchor.setdefault((finding.path, finding.side, finding.line), []).append(fragment)
        # A conservative UTF-8 byte budget stays below GitHub's reported body limits.
        for (filename, side, line), fragments in by_anchor.items():
            body = ""
            for fragment in fragments:
                combined = body + "\n\n---\n\n" + fragment if body else fragment
                if len(combined.encode()) > GITHUB_BODY_BYTES:
                    comments.append({"path": filename, "side": side, "line": line, "body": body})
                    body = fragment
                else:
                    body = combined
            comments.append({"path": filename, "side": side, "line": line, "body": body})
    # Attach the validated completion summary beneath the consolidated overview.
    body = f"{marker}\n## AI review\n\n{result.summary}\n\n"
    for assessment in covered.independent_summaries:
        label = {"claude": "Claude Opus", "openai": "OpenAI"}[assessment.provider]
        body += f"<details><summary>{label}</summary>\n\n{assessment.summary}\n\n</details>\n"
    body += covered.summary
    if cost_report:
        body += (
            "\n\n<details><summary>API costs</summary>\n\n"
            f"```text\n{cost_report}\n```\n\n</details>"
        )
    # Reject oversized publication atomically; never publish a partial approval.
    if (
        len(body.encode()) > GITHUB_BODY_BYTES
        or len(comments) > MAX_GITHUB_COMMENTS
        or len(json.dumps(comments, ensure_ascii=False).encode()) > MAX_GITHUB_PAYLOAD_BYTES
    ):
        raise ValueError("Publication exceeds bounded payload; manual review required")
    # Derive the verdict from original findings, never from the consolidator's prose.
    blocking = any(f.blocking for review in reviews for f in review.findings)
    # Recheck immediately before the one write. GitHub has no conditional-review API.
    pr = gh.request("GET", path)
    if (
        pr["state"] != "open"
        or pr["draft"]
        or pr["head"]["sha"] != context.head
        or pr["base"]["sha"] != context.base
        or pr["user"]["login"] == login
    ):
        raise ValueError("Stale, closed, draft or self-authored PR; not publishing")
    # Submit the summary, verdict and inline comments in one review operation.
    gh.request(
        "POST",
        f"{path}/reviews",
        json={
            "commit_id": context.head,
            "event": "REQUEST_CHANGES" if blocking else "APPROVE",
            "body": body,
            "comments": comments,
        },
    )
    return True
