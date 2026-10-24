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
    Chunk,
    PacketSizer,
    Plan,
    Unit,
    plan,
)
from ai_reviewer.contracts import (
    AnchorIndex,
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
    compile_anchors,
    reconcile,
    source_findings,
    validate_anchors,
)
from ai_reviewer.diagnostics import report_failure
from ai_reviewer.limits import (
    CAPACITY_PACKET_BYTES,
    CONSOLIDATION_BYTES,
    MAX_ATTEMPTS,
    MAX_CONSOLIDATION_CALLS,
    MAX_FINDINGS,
    MAX_RELATIONS,
    MAX_SUMMARY_CALLS,
    MAX_WORK,
    PACKET_BYTES,
    SUMMARY_PAGE_SIZE,
)
from ai_reviewer.models import AssignmentKind, CapacityError, Role, call_model
from ai_reviewer.patch import lines
from ai_reviewer.serialization import digest, encode


class Assignment(Contract):
    """Bind one local/integration scope to both required provider reviews."""

    id: str
    kind: AssignmentKind
    packet_bytes: int = PACKET_BYTES
    payload: str
    sections: list[str]


class AssignmentRecord(Contract):
    """Persist assignment identity and evidence digest without duplicating its payload."""

    id: str
    kind: AssignmentKind
    packet_bytes: int = PACKET_BYTES
    payload_digest: str
    sections: list[str]


def assignment_record(assignment: Assignment) -> AssignmentRecord:
    """Bind manifest ownership to the payload retained in each attempt directory."""
    return AssignmentRecord(
        id=assignment.id,
        kind=assignment.kind,
        packet_bytes=assignment.packet_bytes,
        payload_digest=digest(assignment.payload),
        sections=assignment.sections,
    )


class Completion(Contract):
    """Record one provider's terminal state and number of persisted attempts."""

    assignment: str
    role: Literal["claude", "openai"]
    status: Literal["pending", "complete", "failed"] = "pending"
    attempts: int = Field(default=0, ge=0, le=MAX_ATTEMPTS)


class Manifest(Contract):
    """Persist exact section ownership and required provider work, including failures."""

    plan_digest: str
    packet_bytes: int = PACKET_BYTES
    assignments: list[AssignmentRecord]
    results: list[Completion]


def local_assignments(context: ReviewContext, planned: Plan) -> list[Assignment]:
    """Require one pair of independent reviews for every planned chunk."""
    if planned.whole is not None:
        return [
            Assignment(
                id="whole",
                kind="whole",
                payload=planned.whole,
                sections=[f.filename for f in context.files],
            )
        ]
    return [
        Assignment(
            id=c.id,
            kind="local",
            packet_bytes=planned.packet_bytes,
            payload=c.payload(planned.metadata),
            sections=[u.id for u in c.units],
        )
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

    requests = {q.model_dump_json(): q for evidence in links.values() for q in evidence}
    reference_data = dict(
        zip(requests, source_sections(context, list(requests.values())), strict=True)
    )
    adjacency: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for (left, right), evidence in links.items():
        keys = {q.model_dump_json() for q in evidence}
        adjacency[left][right] = keys
        adjacency[right][left] = keys

    def reference_keys(paths: set[str], added: set[str]) -> set[str]:
        """Look up only relationships incident to newly included paths."""
        return {
            key
            for path in added
            for neighbor, keys in adjacency[path].items()
            if neighbor in paths
            for key in keys
        }

    def reference_source(scope: list[Unit]) -> list[dict[str, Any]]:
        """Reuse resolved immutable evidence, ordered by its canonical request key."""
        paths = {u.filename for u in scope}
        return [reference_data[key] for key in sorted(reference_keys(paths, paths))]

    # Each chunk rechecks integration using actual local source and both reports.
    for i, chunk in enumerate(planned.chunks):
        payload = json.loads(chunk.payload(planned.metadata))
        payload["phase"] = "integration: assess PR-wide compatibility; request missing source"
        payload["reports"] = [local[(chunk.id, role)].model_dump() for role in ("claude", "openai")]
        payload["reference_source"] = reference_source(chunk.units)
        assignments.append(
            Assignment(
                id=f"integration-{i:04d}",
                kind="integration",
                packet_bytes=planned.packet_bytes,
                payload=encode(payload),
                sections=[u.id for u in chunk.units],
            )
        )
    ownership, all_units, edges = integration_edges(planned)
    # Pack relationship pairs together, preserving both actual source endpoints.
    pending: list[Unit] = []
    paths: set[str] = set()
    phase = "integration: check cross-chunk relationships using actual source"
    sizer = PacketSizer(
        planned.metadata,
        all_units,
        {
            "reference_source": reference_data,
            "reports": {
                f"{i}:{role}": local[(chunk.id, role)].model_dump()
                for i, chunk in enumerate(planned.chunks)
                for role in ("claude", "openai")
            },
        },
    )
    budget = sizer.budget(
        f"edge-{len(assignments):04d}", {"phase": phase, "reference_source": [], "reports": []}
    )

    def additions(scope: list[Unit]) -> dict[str, set[str]]:
        """Count newly introduced source, comments, references and owner report pairs."""
        added_paths = {u.filename for u in scope}
        return {
            **sizer.additions(scope),
            "reference_source": reference_keys(paths | added_paths, added_paths - paths),
            "reports": {
                f"{ownership[u.id]}:{role}" for u in scope for role in ("claude", "openai")
            },
        }

    def packet(scope: list[Unit]) -> Assignment:
        """Attach reports only from owners of the actual source under integration."""
        pair = Chunk(id=f"edge-{len(assignments):04d}", units=scope)
        payload = json.loads(pair.payload(planned.metadata))
        payload["phase"] = phase
        payload["reference_source"] = reference_source(scope)
        owners = sorted({ownership[u.id] for u in scope})
        payload["reports"] = [
            local[(planned.chunks[k].id, role)].model_dump()
            for k in owners
            for role in ("claude", "openai")
        ]
        return Assignment(
            id=pair.id,
            kind="integration",
            packet_bytes=planned.packet_bytes,
            payload=encode(payload),
            sections=[u.id for u in scope],
        )

    for a, b in edges:
        extra = [u for u in (a, b) if u.id not in budget.selected["units"]]
        delta = additions(extra)
        if pending and budget.size + budget.growth(delta) > planned.packet_bytes:
            assignments.append(packet(pending))
            pending = []
            paths = set()
            budget = sizer.budget(
                f"edge-{len(assignments):04d}",
                {"phase": phase, "reference_source": [], "reports": []},
            )
            extra = [a, b]
            delta = additions(extra)
        budget.include(delta)
        pending.extend(extra)
        paths.update(u.filename for u in extra)
        if len(assignments) + len(planned.chunks) >= MAX_WORK:
            raise ValueError("Integration work budget exceeded; review incomplete")
    if pending:
        assignments.append(packet(pending))
    # Large accumulated reports are paged with the same actual source on every page.
    expanded = [page for assignment in assignments for page in integration_pages(assignment)]
    if len(expanded) + len(planned.chunks) > MAX_WORK:
        raise ValueError("Integration work budget exceeded; review incomplete")
    return expanded


def integration_edges(planned: Plan) -> tuple[dict[str, int], list[Unit], list[tuple[Unit, Unit]]]:
    """Enumerate bounded cross-owner neighbors and captured file relationships."""
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
            if len(edges) > MAX_RELATIONS:
                raise ValueError("Integration relationship budget exceeded")

    for a, b in zip(all_units, all_units[1:]):
        if PurePosixPath(a.filename).parent == PurePosixPath(b.filename).parent:
            add_edge(a, b)
    for relation in planned.relations:
        left, right = relation.left, relation.right
        for a in by_file[left]:
            for b in by_file[right]:
                add_edge(a, b)
    order = {unit.id: i for i, unit in enumerate(all_units)}
    return (
        ownership,
        all_units,
        sorted(edges.values(), key=lambda edge: (order[edge[0].id], order[edge[1].id])),
    )


def integration_pages(assignment: Assignment) -> list[Assignment]:
    """Partition report evidence while repeating actual source and preserving all findings.

    Each page is required work for both providers. Source alone or an indivisible
    report field that cannot fit fails explicitly instead of losing evidence.
    """
    if len(assignment.payload.encode()) <= assignment.packet_bytes:
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
        if len(trial.encode()) > assignment.packet_bytes and pending:
            pages.append(encode({**base, "reports": pending}))
            pending = []
        pending.append(atom)
        if len(encode({**base, "reports": pending}).encode()) > assignment.packet_bytes:
            raise ValueError("Indivisible integration evidence exceeds packet budget")
    if pending:
        pages.append(encode({**base, "reports": pending}))
    return [
        Assignment(
            id=f"{assignment.id}-page-{index:04d}",
            kind=assignment.kind,
            packet_bytes=assignment.packet_bytes,
            payload=payload,
            sections=assignment.sections,
        )
        for index, payload in enumerate(pages)
    ]


def source_sections(context: ReviewContext, requests: list[SourceRequest]) -> list[dict[str, Any]]:
    """Resolve validated line ranges from captured blobs, with no live repository access."""
    files = {f.filename: f for f in context.files}
    sections = []
    sources: dict[tuple[str, str], list[str]] = {}
    for request in requests:
        if request.path not in files:
            raise ValueError("Required source is outside the captured snapshot")
        file = files[request.path]
        key = (request.path, request.side)
        if key not in sources:
            sources[key] = lines(file.before if request.side == "LEFT" else file.after)
        source = sources[key]
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
    if len(serialized.encode()) > assignment.packet_bytes:
        raise ValueError("Required follow-up context exceeds packet budget")
    return serialized


def attempt_directory(directory: Path, assignment: str, role: str, round_: int) -> Path:
    """Use only internally generated identifiers for isolated attempt artifacts."""
    return directory / "work" / assignment / role / str(round_)


def run_assignment(
    context: ReviewContext,
    assignment: Assignment,
    role: Role,
    directory: Path,
    anchors: AnchorIndex,
) -> int:
    """Run at most three calls; unresolved or unrequested missing context fails closed."""
    history: list[ChunkReview] = []
    payload = assignment.payload
    for round_ in range(MAX_ATTEMPTS):
        target = attempt_directory(directory, assignment.id, role, round_)
        target.mkdir(parents=True, exist_ok=True)
        (target / "payload.json").write_text(payload)
        print(
            f"Review attempt: assignment={assignment.id} role={role} attempt={round_}", flush=True
        )
        call_model(role, payload, target, kind=assignment.kind)
        report = ChunkReview.model_validate_json((target / f"{role}.json").read_text())
        validate_anchors(anchors, [report])
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
    context: ReviewContext,
    manifest: Manifest,
    assignments: list[Assignment],
    directory: Path,
    anchors: AnchorIndex,
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
            pool.submit(
                run_assignment, context, assignment, completion.role, directory, anchors
            ): completion
            for assignment in assignments
            for completion in completions
            if completion.assignment == assignment.id
        }
        for future in as_completed(futures):
            completion = futures[future]
            try:
                completion.attempts = future.result()
                completion.status = "complete"
            except Exception as error:
                report_failure(
                    error, assignment=completion.assignment, role=completion.role, phase="worker"
                )
                completion.status = "failed"
                failed = True
            save_manifest(directory, manifest)
    if failed:
        raise ValueError("Required provider work failed")


def read_batch(
    context: ReviewContext,
    assignments: list[Assignment],
    manifest: Manifest,
    directory: Path,
    anchors: AnchorIndex,
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
                validate_anchors(anchors, [report])
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
    if planned.whole is not None:
        try:
            # Count both provider inputs before either provider can generate a report.
            for role in ("claude", "openai"):
                call_model(
                    role, planned.whole, directory / "preflight", kind="whole", preflight=True
                )
        except CapacityError:
            planned = plan(context, packet_bytes=CAPACITY_PACKET_BYTES)
    manifest = Manifest(
        plan_digest=digest(planned.model_dump_json()),
        packet_bytes=planned.packet_bytes,
        assignments=[],
        results=[],
    )
    anchors = compile_anchors(context)
    local = local_assignments(context, planned)
    run_batch(context, manifest, local, directory, anchors)
    reports = read_batch(context, local, manifest, directory, anchors)
    integration = integration_assignments(context, planned, reports)
    run_batch(context, manifest, integration, directory, anchors)
    read_batch(context, integration, manifest, directory, anchors)


def load_reviews(context: ReviewContext, directory: Path) -> CoveredReviews:
    """Recompute exact required work before consolidation and again before publication."""
    manifest = Manifest.model_validate_json((directory / "manifest.json").read_text())
    planned = plan(context, packet_bytes=manifest.packet_bytes)
    if manifest.plan_digest != digest(planned.model_dump_json()):
        raise ValueError("Coverage plan does not match the captured snapshot")
    anchors = compile_anchors(context)
    local = local_assignments(context, planned)
    reports = read_batch(context, local, manifest, directory, anchors)
    integration = integration_assignments(context, planned, reports)
    expected = local + integration
    keys = [(a.id, role) for a in expected for role in ("claude", "openai")]
    actual = [(r.assignment, r.role) for r in manifest.results]
    if manifest.assignments != [assignment_record(a) for a in expected] or sorted(keys) != sorted(
        actual
    ):
        raise ValueError("Coverage assignments are missing, duplicated or unexpected")
    reports.update(read_batch(context, integration, manifest, directory, anchors))
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
    sources = source_findings(reports)
    items = list(sources.items())
    groups: list[Group] = []
    summaries = [r.summary for r in reports]
    # Bound each consolidation call by bytes and source count, retaining global IDs.
    batches: list[dict[str, Any]] = []
    batch: dict[str, Any] = {}
    for key, finding in items:
        trial = {**batch, key: finding.model_dump()}
        if (
            len(trial) > MAX_FINDINGS
            or len(encode({"sources": trial}).encode()) > CONSOLIDATION_BYTES
        ):
            batches.append(batch)
            batch = {}
        batch[key] = finding.model_dump()
    if batch or not batches:
        batches.append(batch)
    if len(batches) > MAX_CONSOLIDATION_CALLS:
        raise ValueError("Consolidation work budget exceeded")
    combined = encode({"sources": batches[0], "reviews": summaries})
    include_summaries = len(batches) == 1 and len(combined.encode()) <= CONSOLIDATION_BYTES
    if include_summaries:
        summaries = []
    for index, sources_batch in enumerate(batches):
        target = directory / "consolidation" / str(index)
        payload = combined if include_summaries else encode({"sources": sources_batch})
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
        for start in range(0, len(summaries), SUMMARY_PAGE_SIZE):
            target = directory / "consolidation-summaries" / str(summary_calls)
            summary_calls += 1
            if summary_calls > MAX_SUMMARY_CALLS:
                raise ValueError("Summary work budget exceeded")
            call_model(
                "consolidate",
                encode({"sources": {}, "reviews": summaries[start : start + SUMMARY_PAGE_SIZE]}),
                target,
            )
            result = Consolidation.model_validate_json((target / "consolidate.json").read_text())
            if result.groups:
                raise ValueError("Summary pass invented findings")
            reduced.append(result.summary)
        summaries = reduced
    result = FinalConsolidation(summary=summaries[0], groups=groups)
    reconcile(reports, result)
    (directory / "consolidate.json").write_text(result.model_dump_json())
    (directory / "coverage.txt").write_text(covered.summary)
