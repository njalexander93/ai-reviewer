"""Assemble a review snapshot from GitHub evidence and optional Linear requirements.

Preparation owns eligibility, collection, packet limits and the final revision
check. It requests a reviewer only after all provider context is validated.
"""

import re
from pathlib import Path
from typing import Any

from ai_reviewer.chunking import plan, validate_patch
from ai_reviewer.contracts import ChangedFile, PriorComment, ReviewContext
from ai_reviewer.github import GitHub, pr_path, read_version, rebuild_patch
from ai_reviewer.issues import github_issues, linear_issues
from ai_reviewer.limits import MAX_FILES, MAX_SNAPSHOT_BYTES, REPOSITORY_RESERVED_REQUESTS
from ai_reviewer.repository import capture_repository


def prepare(
    gh: GitHub,
    repo: str,
    number: int,
    reviewer: str,
    linear_token: str | None = None,
    repository_directory: Path | None = None,
    *,
    expected_head: str | None = None,
    expected_base: str | None = None,
) -> ReviewContext:
    """Capture changed files, review discussion and optional issue requirements.

    Reads GitHub content and requests the reviewer only after collection succeeds.
    It never executes repository code; optional snapshots include unchanged tracked files.

    Args:
        gh: Client authorized to read the repository and request reviewers.
        repo: Owner/repository pair accepted by pr_path.
        number: Positive pull request number.
        reviewer: Login to request; must differ from the PR author.
        linear_token: Optional Linear personal API key for issue requirements.
        repository_directory: Optional artifact directory for the complete immutable
            comparison/head/base tracked-source index and content-addressed text blobs.
        expected_head: Optional full head SHA the caller admitted; a different PR head
            stops preparation before any further provider request.
        expected_base: Optional full base SHA the caller admitted, checked likewise.

    Returns:
        Validated context containing revisions, PR metadata, discussion, changed
        files and issue evidence. Serialize with model_dump_json for job handoff.

    Raises:
        ValueError: Identifiers, expected revisions or PR state are invalid, the PR
            differs from a supplied expected revision, context is incomplete or
            oversized, or head/base changes during collection. Invalid text
            decoding also fails rather than returning partial context.
        RuntimeError: Either issue provider rejects a query, a referenced issue is
            unavailable, or a GitHub redirect is unsafe. Transport failures propagate.
    """
    # Establish eligibility before collecting potentially expensive context.
    path = pr_path(repo, number)
    for expected in (expected_head, expected_base):
        if expected is not None and not re.fullmatch(r"[0-9a-f]{40}", expected):
            raise ValueError("Expected revision must be a full lowercase commit SHA")
    pr = gh.request("GET", path)
    # Bind the review to the revision the caller admitted, before any further reads.
    if (expected_head is not None and pr["head"]["sha"] != expected_head) or (
        expected_base is not None and pr["base"]["sha"] != expected_base
    ):
        raise ValueError("PR head or base differs from the admitted revision; not reviewing")
    if pr["state"] != "open" or pr["draft"] or pr["user"]["login"] == reviewer:
        raise ValueError("PR is closed, draft, or authored by the reviewer account")
    # Require a complete, bounded file list and resolve the diff's merge base.
    files = gh.pages(f"{path}/files")
    if len(files) != pr["changed_files"] or len(files) > MAX_FILES:
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
    collected_bytes = 0
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
        collected_bytes += len(before.encode()) + len(after.encode())
        if collected_bytes > MAX_SNAPSHOT_BYTES:
            raise ValueError("Snapshot exceeds collection budget")
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
        validate_patch(evidence[-1])
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
    # Share one issue snapshot across every local, follow-up and integration review.
    reference_text = f"{pr['title']}\n{pr['body'] or ''}"
    issues = github_issues(gh, repo, number, reference_text)
    issues.extend(
        linear_issues(
            linear_token,
            f"https://github.com/{repo}/pull/{number}",
            f"{reference_text}\n{pr['head']['ref']}",
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
    # Capture complete tracked source separately; prompts receive only its digest.
    if repository_directory is not None:
        if gh.requests_remaining < REPOSITORY_RESERVED_REQUESTS:
            raise ValueError("Insufficient GitHub request budget for repository snapshots")
        context.repository_digest = capture_repository(gh, context, repository_directory).digest
    # Budget only the evidence that will actually reach the providers.
    plan(context)
    # Reject a collection that raced a head or base update.
    latest = gh.request("GET", path)
    if latest["head"]["sha"] != context.head or latest["base"]["sha"] != context.base:
        raise ValueError("PR changed during context collection")
    # Make the reviewer request only after the packet passes validation.
    gh.request("POST", f"{path}/requested_reviewers", json={"reviewers": [reviewer]})
    return context
