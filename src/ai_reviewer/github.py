"""Read immutable PR context and publish validated reviews through GitHub's REST API.

Repository content is never checked out or executed. All requests remain on the
fixed GitHub API host; untrusted paths are encoded, not interpolated as URLs.
"""

import base64
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx

from ai_reviewer.contracts import (
    ChangedFile,
    Consolidation,
    PriorComment,
    Review,
    ReviewContext,
    reconcile,
)
from ai_reviewer.issues import github_issues, linear_issues


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
    for line in patch.split("\n"):
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
        if entry["size"] > 150_000:
            raise ValueError("Changed file exceeds context limit")
        blob = gh.request("GET", f"/repos/{repo}/git/blobs/{entry['sha']}")
        if blob.get("encoding") != "base64" or blob["size"] > 150_000:
            raise ValueError("Unavailable or oversized Git blob")
        text = base64.b64decode(blob["content"]).decode("utf-8")
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


def validate_anchors(context: ReviewContext, reviews: list[Review]) -> None:
    """Reject invalid model locations before purchasing consolidation.

    Args:
        context: Immutable changed-file packet supplied to both reviewers.
        reviews: Independent reports with locations to validate.

    Raises:
        ValueError: A finding is outside the captured diff. Reports remain on disk
            for diagnosis; this fails the run rather than silently dropping findings.
    """
    anchors = {file.filename: diff_lines(file.patch) for file in context.files}
    for review in reviews:
        for finding in review.findings:
            if (finding.side, finding.line) not in anchors.get(finding.path, set()):
                raise ValueError("Finding is outside the captured diff")


def prepare(
    gh: GitHub, repo: str, number: int, reviewer: str, linear_token: str | None = None
) -> ReviewContext:
    """Capture the diff, changed file contents and earlier review discussion.

    Reads GitHub content and requests the reviewer only after collection succeeds.
    It never executes repository code or inspects unchanged dependency files.

    Args:
        gh: Client authorized to read the repository and request reviewers.
        repo: Owner/repository pair accepted by pr_path.
        number: Positive pull request number.
        reviewer: Login to request; must differ from the PR author.
        linear_token: Optional Linear personal API key for issue requirements.

    Returns:
        Validated context model containing revisions, PR metadata, discussion and
        changed-file evidence. Serialize it with model_dump_json for job handoff.

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
    # Resolve immutable roots once; tree reads are shared across changed paths.
    head, base = pr["head"]["sha"], pr["base"]["sha"]
    diff_base = comparison["merge_base_commit"]["sha"]
    roots = {
        ref: gh.request("GET", f"/repos/{repo}/git/commits/{ref}")["tree"]["sha"]
        for ref in (diff_base, head)
    }
    cache: dict[str, Any] = {}
    evidence = []
    # Read actual Git objects, retaining mode changes and symlink text.
    for file in files:
        before, before_mode = (
            ("", None)
            if file["status"] == "added"
            else read_version(
                gh, repo, roots[diff_base], file.get("previous_filename", file["filename"]), cache
            )
        )
        after, after_mode = (
            ("", None)
            if file["status"] == "removed"
            else read_version(gh, repo, roots[head], file["filename"], cache)
        )
        evidence.append(
            ChangedFile(
                filename=file["filename"],
                previous_filename=file.get("previous_filename"),
                status=file["status"],
                patch=file["patch"] if "patch" in file else rebuild_patch(before, after),
                before=before,
                after=after,
                before_mode=before_mode,
                after_mode=after_mode,
            )
        )
    # Project discussion into a bounded, explicit contract without repeated diff hunks.
    comments = [
        PriorComment(
            path=c["path"],
            line=c.get("line"),
            side=c.get("side"),
            body=c["body"],
            author=(c.get("user") or {}).get("login"),
        )
        for c in gh.pages(f"{path}/comments")
    ]
    # Share one issue snapshot with both reviewers, under the existing packet budget.
    reference_text = f"{pr['title']}\n{pr['body'] or ''}"
    issues = github_issues(gh, repo, number, reference_text)
    issues.extend(
        linear_issues(
            linear_token,
            f"https://github.com/{repo}/pull/{number}",
            f"{reference_text}\n{pr['head'].get('ref', '')}",
        )
    )
    context = ReviewContext(
        repo=repo,
        number=number,
        head=head,
        base=base,
        diff_base=diff_base,
        reviewer=reviewer,
        title=pr["title"],
        description=pr["body"],
        files=evidence,
        previous_comments=comments,
        issues=issues,
    )
    # Budget only the evidence that will actually reach the providers.
    if len(context.model_dump_json().encode()) > 600_000:
        raise ValueError("PR context exceeds 600 KB; split the change")
    # Reject a collection that raced a head or base update.
    latest = gh.request("GET", path)
    if latest["head"]["sha"] != context.head or latest["base"]["sha"] != context.base:
        raise ValueError("PR changed during context collection")
    # Make the reviewer request only after the packet passes validation.
    gh.request("POST", f"{path}/requested_reviewers", json={"reviewers": [reviewer]})
    return context


def publish(
    gh: GitHub, context: ReviewContext, reviews: list[Review], result: Consolidation
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
    repo, number = context.repo, context.number
    path = pr_path(repo, number)
    login = gh.request("GET", "/user")["login"]
    if login != context.reviewer:
        raise ValueError("Publishing identity differs from requested reviewer")
    # Avoid another submission when this account already reviewed the revision pair.
    marker = f"<!-- ai-reviewer:{context.head}:{context.base} -->"
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
                if len(combined.encode()) > 60_000:
                    comments.append({"path": filename, "side": side, "line": line, "body": body})
                    body = fragment
                else:
                    body = combined
            comments.append({"path": filename, "side": side, "line": line, "body": body})
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
