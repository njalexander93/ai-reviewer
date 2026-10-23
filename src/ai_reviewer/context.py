"""Assemble a review snapshot from GitHub evidence and optional Linear requirements.

Preparation owns eligibility, collection, packet limits and the final revision
check. It requests a reviewer only after all provider context is validated.
"""

from typing import Any

from ai_reviewer.contracts import ChangedFile, PriorComment, ReviewContext
from ai_reviewer.github import GitHub, pr_path, read_version, rebuild_patch
from ai_reviewer.issues import github_issues, linear_issues


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
