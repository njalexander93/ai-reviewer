"""Execute and validate bounded chunk, follow-up and integration work in one job.

Every provider attempt has an isolated artifact directory. The manifest records
planned work before calls start; downstream stages replay the plan and source
requests against the captured snapshot before accepting any completed report.
"""

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field

from ai_reviewer.chunking import (
    MAX_WORK,
    PACKET_BYTES,
    Chunk,
    Plan,
    Unit,
    digest,
    encode,
    lines,
    plan,
)
from ai_reviewer.contracts import (
    ChunkReview,
    CompletedReview,
    Consolidation,
    Contract,
    CoveredReviews,
    FinalConsolidation,
    Group,
    Review,
    ReviewContext,
    SourceRequest,
    reconcile,
    source_findings,
    validate_anchors,
)
from ai_reviewer.models import CapacityError, Role, call_model


class Assignment(Contract):
    """Bind one local/integration scope to both required provider reviews."""

    id: str
    payload: str
    sections: list[str]


class AssignmentRecord(Contract):
    """Persist assignment identity and evidence digest without duplicating its payload."""

    id: str
    payload_digest: str
    sections: list[str]


def assignment_record(assignment: Assignment) -> AssignmentRecord:
    """Bind manifest ownership to the payload retained in each attempt directory."""
    return AssignmentRecord(
        id=assignment.id, payload_digest=digest(assignment.payload), sections=assignment.sections
    )


class Completion(Contract):
    """Record one provider's terminal state and number of persisted attempts."""

    assignment: str
    role: Literal["claude", "openai"]
    status: Literal["pending", "complete", "failed"] = "pending"
    attempts: int = Field(default=0, ge=0, le=3)


class Manifest(Contract):
    """Persist exact section ownership and required provider work, including failures."""

    plan_digest: str
    force_chunks: bool = False
    assignments: list[AssignmentRecord]
    results: list[Completion]


def local_assignments(context: ReviewContext, planned: Plan) -> list[Assignment]:
    """Require one pair of independent reviews for every planned chunk."""
    if planned.whole is not None:
        return [
            Assignment(
                id="whole", payload=planned.whole, sections=[f.filename for f in context.files]
            )
        ]
    return [
        Assignment(id=c.id, payload=c.payload(planned.metadata), sections=[u.id for u in c.units])
        for c in planned.chunks
    ]


def integration_assignments(
    context: ReviewContext, planned: Plan, local: dict[tuple[str, str], Review]
) -> list[Assignment]:
    """Require actual-source integration across every visible cross-chunk relation.

    Directory peers, source/test pairs and direct references are conservative
    candidates, not a complete dependency graph. Every chunk also receives an
    integration pass to identify missing relationships via source requests.
    """
    if planned.whole is not None:
        return []
    assignments = []
    links = {(r.left, r.right): r.evidence for r in planned.relations}

    def reference_source(scope: list[Unit]) -> list[dict[str, Any]]:
        """Retain actual imports for same-chunk and cross-chunk relationships alike."""
        paths = {u.filename for u in scope}
        requests = [
            request
            for (a, b), evidence in links.items()
            if a in paths and b in paths
            for request in evidence
        ]
        unique = {request.model_dump_json(): request for request in requests}
        return source_sections(context, list(unique.values()))

    # Each chunk rechecks integration using actual local source and both reports.
    for i, chunk in enumerate(planned.chunks):
        payload = json.loads(chunk.payload(planned.metadata))
        payload["phase"] = "integration: assess PR-wide compatibility; request missing source"
        payload["reports"] = [local[(chunk.id, role)].model_dump() for role in ("claude", "openai")]
        payload["reference_source"] = reference_source(chunk.units)
        assignments.append(
            Assignment(
                id=f"integration-{i:04d}",
                payload=encode(payload),
                sections=[u.id for u in chunk.units],
            )
        )
    # Directory and split-file neighbors establish bounded integration scopes.
    # Explicit source/test and textual references add non-adjacent relationships.
    ownership = {u.id: i for i, c in enumerate(planned.chunks) for u in c.units}
    all_units = sorted(
        (u for c in planned.chunks for u in c.units), key=lambda u: (u.filename, u.patch_start)
    )
    by_file: dict[str, list[Unit]] = defaultdict(list)
    for unit in all_units:
        by_file[unit.filename].append(unit)
    edges: dict[tuple[str, str], tuple[Unit, Unit]] = {}

    def add_edge(a: Unit, b: Unit) -> None:
        """Keep distinct cross-owner pairs bounded before packet construction."""
        if ownership[a.id] != ownership[b.id]:
            edges[(a.id, b.id)] = (a, b)
            if len(edges) > 20_000:
                raise ValueError("Integration relationship budget exceeded")

    for a, b in zip(all_units, all_units[1:]):
        if PurePosixPath(a.filename).parent == PurePosixPath(b.filename).parent:
            add_edge(a, b)
    for left, right in links:
        for a in by_file[left]:
            for b in by_file[right]:
                add_edge(a, b)
    # Pack relationship pairs together, preserving both actual source endpoints.
    pending: list[Unit] = []

    def packet(scope: list[Unit]) -> Assignment:
        """Attach reports only from owners of the actual source under integration."""
        pair = Chunk(id=f"edge-{len(assignments):04d}", units=scope)
        payload = json.loads(pair.payload(planned.metadata))
        payload["phase"] = "integration: check cross-chunk relationships using actual source"
        payload["reference_source"] = reference_source(scope)
        owners = sorted({ownership[u.id] for u in scope})
        payload["reports"] = [
            local[(planned.chunks[k].id, role)].model_dump()
            for k in owners
            for role in ("claude", "openai")
        ]
        return Assignment(id=pair.id, payload=encode(payload), sections=[u.id for u in scope])

    order = {unit.id: i for i, unit in enumerate(all_units)}
    for a, b in sorted(edges.values(), key=lambda edge: (order[edge[0].id], order[edge[1].id])):
        trial = pending + [u for u in (a, b) if u not in pending]
        if pending and len(packet(trial).payload.encode()) > PACKET_BYTES:
            assignments.append(packet(pending))
            pending = []
            trial = [a, b]
        pending = trial
        if len(assignments) + len(planned.chunks) >= MAX_WORK:
            raise ValueError("Integration work budget exceeded; review incomplete")
    if pending:
        assignments.append(packet(pending))
    # Large accumulated reports are paged with the same actual source on every page.
    expanded = [page for assignment in assignments for page in integration_pages(assignment)]
    if len(expanded) + len(planned.chunks) > MAX_WORK:
        raise ValueError("Integration work budget exceeded; review incomplete")
    return expanded


def integration_pages(assignment: Assignment) -> list[Assignment]:
    """Partition report evidence while repeating actual source and preserving all findings.

    Each page is required work for both providers. Source alone or an indivisible
    report field that cannot fit fails explicitly instead of losing evidence.
    """
    if len(assignment.payload.encode()) <= PACKET_BYTES:
        return [assignment]
    base = json.loads(assignment.payload)
    reports = base.pop("reports")
    atoms = []
    for index, report in enumerate(reports):
        atoms.append(
            {"report_index": index, "summary": report["summary"], "complete": report["complete"]}
        )
        atoms.extend({"report_index": index, "finding": finding} for finding in report["findings"])
    pages = []
    pending: list[dict[str, Any]] = []
    for atom in atoms:
        trial = encode({**base, "reports": [*pending, atom]})
        if len(trial.encode()) > PACKET_BYTES and pending:
            pages.append(encode({**base, "reports": pending}))
            pending = []
        pending.append(atom)
        if len(encode({**base, "reports": pending}).encode()) > PACKET_BYTES:
            raise ValueError("Indivisible integration evidence exceeds packet budget")
    if pending:
        pages.append(encode({**base, "reports": pending}))
    return [
        Assignment(
            id=f"{assignment.id}-page-{index:04d}", payload=payload, sections=assignment.sections
        )
        for index, payload in enumerate(pages)
    ]


def source_sections(context: ReviewContext, requests: list[SourceRequest]) -> list[dict[str, Any]]:
    """Resolve validated line ranges from captured blobs, with no live repository access."""
    files = {f.filename: f for f in context.files}
    sections = []
    for request in requests:
        if request.path not in files:
            raise ValueError("Required source is outside the captured snapshot")
        file = files[request.path]
        source = lines(file.before if request.side == "LEFT" else file.after)
        if request.end < request.start or request.end > len(source):
            raise ValueError("Required source range is unavailable")
        sections.append(
            {**request.model_dump(), "text": "".join(source[request.start - 1 : request.end])}
        )
    return sections


def followup_payload(
    context: ReviewContext, assignment: Assignment, history: list[ChunkReview]
) -> str:
    """Keep original work and all requested evidence available across bounded rounds."""
    payload = json.loads(assignment.payload)
    payload["prior_attempts"] = [r.model_dump() for r in history]
    payload["requested_source"] = source_sections(context, [q for r in history for q in r.requests])
    serialized = encode(payload)
    if len(serialized.encode()) > PACKET_BYTES:
        raise ValueError("Required follow-up context exceeds packet budget")
    return serialized


def attempt_directory(directory: Path, assignment: str, role: str, round_: int) -> Path:
    """Use only internally generated identifiers for isolated attempt artifacts."""
    return directory / "work" / assignment / role / str(round_)


def run_assignment(
    context: ReviewContext, assignment: Assignment, role: Role, directory: Path
) -> int:
    """Run at most three calls; unresolved or unrequested missing context fails closed."""
    history: list[ChunkReview] = []
    payload = assignment.payload
    for round_ in range(3):
        target = attempt_directory(directory, assignment.id, role, round_)
        target.mkdir(parents=True, exist_ok=True)
        (target / "payload.json").write_text(payload)
        call_model(role, payload, target, chunked=True)
        report = ChunkReview.model_validate_json((target / f"{role}.json").read_text())
        validate_anchors(context, [report])
        history.append(report)
        if not report.requests:
            if not report.complete:
                raise ValueError("Required context unresolved")
            return len(history)
        if report.complete:
            raise ValueError("Complete report cannot have required source requests")
        payload = followup_payload(context, assignment, history)
    raise ValueError("Required follow-up work exhausted")


def save_manifest(directory: Path, manifest: Manifest) -> None:
    """Write coordinator-owned state atomically, never from worker threads."""
    temporary = directory / "manifest.tmp"
    temporary.write_text(manifest.model_dump_json())
    temporary.replace(directory / "manifest.json")


def run_batch(
    context: ReviewContext, manifest: Manifest, assignments: list[Assignment], directory: Path
) -> None:
    """Schedule two workers, recording all terminal results even if one fails."""
    if len(manifest.assignments) + len(assignments) > MAX_WORK:
        raise ValueError("Total review work budget exceeded")
    manifest.assignments.extend(assignment_record(a) for a in assignments)
    completions = [
        Completion(assignment=a.id, role=role) for a in assignments for role in ("claude", "openai")
    ]
    manifest.results.extend(completions)
    save_manifest(directory, manifest)
    failed = False
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(run_assignment, context, assignment, completion.role, directory): completion
            for assignment in assignments
            for completion in completions
            if completion.assignment == assignment.id
        }
        for future in as_completed(futures):
            completion = futures[future]
            try:
                completion.attempts = future.result()
                completion.status = "complete"
            except Exception:
                completion.status = "failed"
                failed = True
            save_manifest(directory, manifest)
    if failed:
        raise ValueError("Required provider work failed")


def read_batch(
    context: ReviewContext, assignments: list[Assignment], manifest: Manifest, directory: Path
) -> dict[tuple[str, str], Review]:
    """Replay attempt inputs and require complete results with no unresolved requests."""
    reports: dict[tuple[str, str], Review] = {}
    for assignment in assignments:
        for role in ("claude", "openai"):
            matches = [
                r for r in manifest.results if r.assignment == assignment.id and r.role == role
            ]
            if len(matches) != 1 or matches[0].status != "complete" or not matches[0].attempts:
                raise ValueError("Missing, duplicate or failed worker result")
            completion = matches[0]
            history: list[ChunkReview] = []
            payload = assignment.payload
            for round_ in range(completion.attempts):
                target = attempt_directory(directory, assignment.id, role, round_)
                if (target / "payload.json").read_text() != payload:
                    raise ValueError("Attempt evidence differs from required work")
                report = ChunkReview.model_validate_json((target / f"{role}.json").read_text())
                validate_anchors(context, [report])
                history.append(report)
                terminal = round_ == completion.attempts - 1
                if terminal:
                    if not report.complete or report.requests:
                        raise ValueError("Unresolved required work")
                elif report.complete or not report.requests:
                    raise ValueError("Unexpected follow-up attempt")
                else:
                    payload = followup_payload(context, assignment, history)
            reports[(assignment.id, role)] = CompletedReview(
                complete=True,
                summary=history[-1].summary,
                findings=[f for r in history for f in r.findings],
            )
    return reports


def review_chunks(context: ReviewContext, directory: Path) -> None:
    """Run one manifest-backed review flow, using the whole packet when capacity permits."""
    # Reject reused workspaces instead of mixing attempts and their cost accounting.
    if (directory / "manifest.json").exists() or (directory / "work").exists():
        raise ValueError("Review run requires a fresh artifact directory")
    planned = plan(context)
    force_chunks = False
    if planned.whole is not None:
        try:
            # Count both provider inputs before either provider can generate a report.
            for role in ("claude", "openai"):
                call_model(
                    role, planned.whole, directory / "preflight", chunked=True, preflight=True
                )
        except CapacityError:
            force_chunks = True
            planned = plan(context, force_chunks=True)
    manifest = Manifest(
        plan_digest=digest(planned.model_dump_json()),
        force_chunks=force_chunks,
        assignments=[],
        results=[],
    )
    local = local_assignments(context, planned)
    run_batch(context, manifest, local, directory)
    reports = read_batch(context, local, manifest, directory)
    integration = integration_assignments(context, planned, reports)
    run_batch(context, manifest, integration, directory)
    read_batch(context, integration, manifest, directory)


def load_reviews(context: ReviewContext, directory: Path) -> CoveredReviews:
    """Recompute exact required work before consolidation and again before publication."""
    manifest = Manifest.model_validate_json((directory / "manifest.json").read_text())
    planned = plan(context, force_chunks=manifest.force_chunks)
    if manifest.plan_digest != digest(planned.model_dump_json()):
        raise ValueError("Coverage plan does not match the captured snapshot")
    local = local_assignments(context, planned)
    reports = read_batch(context, local, manifest, directory)
    integration = integration_assignments(context, planned, reports)
    expected = local + integration
    keys = [(a.id, role) for a in expected for role in ("claude", "openai")]
    actual = [(r.assignment, r.role) for r in manifest.results]
    if manifest.assignments != [assignment_record(a) for a in expected] or sorted(keys) != sorted(
        actual
    ):
        raise ValueError("Coverage assignments are missing, duplicated or unexpected")
    reports.update(read_batch(context, integration, manifest, directory))
    return CoveredReviews(
        reviews=[reports[key] for key in keys],
        snapshot=planned.snapshot,
        summary=f"{len(local)} local scopes; {len(expected)} local/integration assignments "
        f"completed by both providers. Snapshot `{planned.snapshot}`. "
        "Exact section ownership and follow-up evidence are in the workflow coverage manifest. "
        "Completion does not establish model understanding.",
    )


def consolidate_chunks(context: ReviewContext, directory: Path) -> None:
    """Consolidate bounded source batches with stable global IDs and no dropped findings.

    Cross-batch duplicates remain separate. Sonnet never replaces original text.
    Summary batches remain explicitly separate so aggregation cannot erase dissent.
    """
    if any((directory / name).exists() for name in ("consolidation", "consolidation-summaries")):
        raise ValueError("Consolidation requires fresh artifacts; earlier usage must be retained")
    covered = load_reviews(context, directory)
    reports = covered.reviews
    sources = source_findings(reports, len(reports))
    items = list(sources.items())
    groups: list[Group] = []
    summaries = [r.summary for r in reports]
    # Bound each consolidation call by bytes and source count, retaining global IDs.
    batches: list[dict[str, Any]] = []
    batch: dict[str, Any] = {}
    for key, finding in items:
        trial = {**batch, key: finding.model_dump()}
        if len(trial) > 40 or len(encode({"sources": trial}).encode()) > 200_000:
            batches.append(batch)
            batch = {}
        batch[key] = finding.model_dump()
    if batch or not batches:
        batches.append(batch)
    if len(batches) > 128:
        raise ValueError("Consolidation work budget exceeded")
    for index, sources_batch in enumerate(batches):
        target = directory / "consolidation" / str(index)
        payload = encode({"sources": sources_batch})
        call_model("consolidate", payload, target)
        result = Consolidation.model_validate_json((target / "consolidate.json").read_text())
        flattened = [key for group in result.groups for key in group.source_ids]
        if sorted(flattened) != sorted(sources_batch):
            raise ValueError("Consolidation lost or duplicated original findings")
        groups.extend(result.groups)
        summaries.append(result.summary)
    # Summarize bounded pages of original assessments and grouping results. No finding
    # text is rewritten; explicit disagreements must survive every summary round.
    summary_calls = 0
    while len(summaries) > 1:
        reduced = []
        for start in range(0, len(summaries), 8):
            target = directory / "consolidation-summaries" / str(summary_calls)
            summary_calls += 1
            if summary_calls > 128:
                raise ValueError("Summary work budget exceeded")
            call_model(
                "consolidate",
                encode({"sources": {}, "reviews": summaries[start : start + 8]}),
                target,
            )
            result = Consolidation.model_validate_json((target / "consolidate.json").read_text())
            if result.groups:
                raise ValueError("Summary pass invented findings")
            reduced.append(result.summary)
        summaries = reduced
    result = FinalConsolidation(summary=summaries[0], groups=groups)
    reconcile(reports, result, len(reports))
    (directory / "consolidate.json").write_text(result.model_dump_json())
    (directory / "coverage.txt").write_text(covered.summary)
