"""Execute and validate bounded chunk, follow-up and integration work in one job.

Every provider attempt has an isolated artifact directory. The manifest records
the enabled providers and planned work before generation starts. Downstream stages
replay the plan and source requests against the captured snapshot before accepting
any completed report.
Pure integration planning also lives here; only review execution calls models.
No repository code runs. Failures preserve evidence but prevent partial approval.
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
    omit_navigation,
    plan,
)
from ai_reviewer.contracts import (
    AnchorIndex,
    ChunkReview,
    CompletedReview,
    Contract,
    CoveredReviews,
    IndependentSummary,
    Provider,
    Providers,
    Review,
    ReviewContext,
    SourceRequest,
    compile_anchors,
    validate_anchors,
)
from ai_reviewer.diagnostics import FollowupExhausted, IncompleteWithoutRequests, report_failure
from ai_reviewer.limits import (
    CAPACITY_PACKET_BYTES,
    MAX_ATTEMPTS,
    MAX_RELATIONS,
    MAX_WORK,
)
from ai_reviewer.models import (
    AssignmentKind,
    CapacityError,
    Role,
    call_model,
    configured_providers,
)
from ai_reviewer.patch import lines
from ai_reviewer.serialization import digest, encode


class Assignment(Contract):
    """Bind one whole, local or integration scope to the enabled reviewers.

    Attributes:
        id: Internally generated identifier used in artifact paths.
        kind: Prompt scope; independent of payload shape or size.
        payload: Initial serialized JSON evidence, retained unchanged for replay.
        sections: Owned filenames for whole work, otherwise source-unit IDs.
    """

    id: str
    kind: AssignmentKind
    payload: str
    sections: list[str]


class AssignmentRecord(Contract):
    """Persist assignment identity without duplicating its payload.

    Attributes:
        id: Assignment identifier, matching its attempt directories.
        kind: Required prompt scope.
        payload_digest: SHA-256 of the initial payload's UTF-8 bytes.
        sections: Exact ownership copied from the planned assignment.
    """

    id: str
    kind: AssignmentKind
    payload_digest: str
    sections: list[str]


def assignment_record(assignment: Assignment) -> AssignmentRecord:
    """Return identity, ownership and a UTF-8 payload digest for assignment.

    The returned record shares the section values but stores no source payload.
    No files are read or written.
    """
    return AssignmentRecord(
        id=assignment.id,
        kind=assignment.kind,
        payload_digest=digest(assignment.payload),
        sections=assignment.sections,
    )


class Completion(Contract):
    """Track coordinator-owned progress for one assignment/provider pair.

    Attributes:
        assignment: Planned assignment ID.
        role: Independent provider required to complete that assignment.
        status: Pending before execution; complete or failed after its future settles.
        attempts: Validated report count for success or incomplete responses, at most
            MAX_ATTEMPTS. Zero on other failures means unknown, not no provider calls.
    """

    assignment: str
    role: Provider
    status: Literal["pending", "complete", "failed"] = "pending"
    attempts: int = Field(default=0, ge=0, le=MAX_ATTEMPTS)


class Manifest(Contract):
    """Persist required work and results for deterministic coverage replay.

    Attributes:
        providers: Required reviewers resolved once before preflight, never from artifacts.
        plan_digest: Hash binding the complete plan and enabled provider set.
        packet_bytes: Selected UTF-8 ceiling needed to reconstruct that plan.
        assignments: Ordered, digest-only records appended before each batch starts.
        results: One mutable completion per assignment/provider, owned by the coordinator.
    """

    providers: Providers
    plan_digest: str
    packet_bytes: int
    assignments: list[AssignmentRecord]
    results: list[Completion]


def plan_digest(planned: Plan, providers: Providers) -> str:
    """Hash the exact plan and canonical required reviewers using the replay wire format.

    Both inputs remain unchanged. The digest binds snapshot, packet policy and provider
    selection; execution and replay must use this same serialization.
    """
    return digest(encode({"plan": planned.model_dump(), "providers": providers}))


def local_assignments(context: ReviewContext, planned: Plan) -> list[Assignment]:
    """Create initial work from a plan for the same immutable context.

    Args:
        context: Captured snapshot used to name whole-packet ownership.
        planned: Plan produced from that context; not mutated here.

    Returns:
        One whole assignment with byte-identical input, or ordered local assignments.
        Each assignment requires every enabled provider; this function performs no I/O.
    """
    # Preserve small-input bytes while using the same downstream work protocol.
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
            payload=c.payload(planned.metadata),
            sections=[u.id for u in c.units],
        )
        for c in planned.chunks
    ]


def report_atoms(assignment: str, provider: str, report: Review) -> list[dict[str, Any]]:
    """Return identity-bearing summary and finding records without modifying report.

    Args:
        assignment: Source owner ID retained on every record.
        provider: Independent reviewer label retained on every record.
        report: Completed assessment whose findings must all survive paging.

    Returns:
        Summary/complete record first, followed by individual findings in original order.
    """
    identity = {"assignment": assignment, "provider": provider}
    return [
        {**identity, "summary": report.summary, "complete": report.complete},
        *({**identity, "finding": finding.model_dump()} for finding in report.findings),
    ]


def integration_assignments(
    context: ReviewContext,
    planned: Plan,
    local: dict[tuple[str, str], Review],
    providers: Providers,
) -> list[Assignment]:
    """Plan actual-source checks across visible relationships and every local scope.

    Directory peers, source/test names and literal references are heuristics, not a
    complete dependency graph. No files, providers or GitHub endpoints are accessed.

    Args:
        context: Immutable snapshot supplying captured reference source.
        planned: Matching local plan, metadata, relationships and packet ceiling.
        local: Completed reports keyed by every (chunk ID, provider) pair.
        providers: Canonical required reviewer set, matching the run manifest.

    Returns:
        Deterministically ordered per-chunk, cross-chunk and required page assignments.
        Whole-packet plans return no additional work. Inputs remain unchanged.

    Raises:
        KeyError: A required local report is absent.
        ValueError: Source evidence is unavailable, an indivisible packet cannot fit,
            or relationship/assignment bounds are exceeded. Evidence is never dropped.
    """
    if planned.whole is not None:
        return []
    # Resolve report identities and captured reference ranges once for all scopes.
    report_data = {
        (chunk.id, role): report_atoms(chunk.id, role, local[(chunk.id, role)])
        for chunk in planned.chunks
        for role in providers
    }
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
        """Return evidence keys for relationships from added paths into paths.

        Both arguments contain current filenames; added identifies the newly admitted
        endpoints. The captured adjacency index is read without fetching source.
        """
        return {
            key
            for path in added
            for neighbor, keys in adjacency[path].items()
            if neighbor in paths
            for key in keys
        }

    def reference_source(scope: list[Unit]) -> list[dict[str, Any]]:
        """Return cached evidence for scope's internal relationships in stable key order.

        Scope contains planned units; the returned dictionaries are shared read-only
        values from reference_data, not additional source reads.
        """
        paths = {u.filename for u in scope}
        return [reference_data[key] for key in sorted(reference_keys(paths, paths))]

    # Each chunk rechecks integration using actual local source and all enabled reports.
    for i, chunk in enumerate(planned.chunks):
        payload = json.loads(chunk.payload(planned.metadata))
        payload["phase"] = "integration: assess PR-wide compatibility; request missing source"
        payload["reports"] = [atom for role in providers for atom in report_data[(chunk.id, role)]]
        payload["reference_source"] = reference_source(chunk.units)
        assignments.append(
            Assignment(
                id=f"integration-{i:04d}",
                kind="integration",
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
                f"{i}:{role}:{j}": atom
                for i, chunk in enumerate(planned.chunks)
                for role in providers
                for j, atom in enumerate(report_data[(chunk.id, role)])
            },
        },
    )
    budget = sizer.budget(
        f"edge-{len(assignments):04d}", {"phase": phase, "reference_source": [], "reports": []}
    )

    def additions(scope: list[Unit]) -> dict[str, set[str]]:
        """Return array membership additions for scope relative to current pending paths.

        Keys identify source, indexed comments, reference ranges and owner report atoms.
        The sizer deduplicates these memberships; this helper does not mutate its budget.
        """
        added_paths = {u.filename for u in scope}
        return {
            **sizer.additions(scope),
            "reference_source": reference_keys(paths | added_paths, added_paths - paths),
            "reports": {
                f"{ownership[u.id]}:{role}:{j}"
                for u in scope
                for role in providers
                for j in range(len(report_data[(planned.chunks[ownership[u.id]].id, role)]))
            },
        }

    def packet(scope: list[Unit]) -> Assignment:
        """Serialize scope with reports from its owners and captured reference evidence.

        The edge ID follows the current assignments count. All scope units become owned
        sections in the returned assignment; packet size is checked by the caller/pager.
        """
        pair = Chunk(id=f"edge-{len(assignments):04d}", units=scope)
        payload = json.loads(pair.payload(planned.metadata))
        payload["phase"] = phase
        payload["reference_source"] = reference_source(scope)
        owners = sorted({ownership[u.id] for u in scope})
        payload["reports"] = [
            atom
            for k in owners
            for role in providers
            for atom in report_data[(planned.chunks[k].id, role)]
        ]
        return Assignment(
            id=pair.id,
            kind="integration",
            payload=encode(payload),
            sections=[u.id for u in scope],
        )

    # Admit whole endpoint pairs; flush before adding a pair that exceeds the ceiling.
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
    expanded = [
        page
        for assignment in assignments
        for page in integration_pages(assignment, planned.packet_bytes)
    ]
    if len(expanded) + len(planned.chunks) > MAX_WORK:
        raise ValueError("Integration work budget exceeded; review incomplete")
    return expanded


def integration_edges(planned: Plan) -> tuple[dict[str, int], list[Unit], list[tuple[Unit, Unit]]]:
    """Enumerate deterministic cross-owner neighbor and file-reference pairs.

    Args:
        planned: Chunked plan with unique unit IDs and captured file relationships.

    Returns:
        Unit-to-owner index, units sorted by filename/patch offset, and ordered pairs.
        Directory adjacency includes split-file neighbors; same-owner pairs are omitted.

    Raises:
        ValueError: Distinct cross-owner pairs exceed MAX_RELATIONS.
    """
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
        """Add the directed pair to the closure's edge map only across distinct owners.

        Repeated unit-ID pairs replace the same entry. Raise ValueError immediately
        when distinct pairs exceed MAX_RELATIONS; a and b must belong to the plan.
        """
        if ownership[a.id] != ownership[b.id]:
            edges[(a.id, b.id)] = (a, b)
            if len(edges) > MAX_RELATIONS:
                raise ValueError("Integration relationship budget exceeded")

    # Include local neighbors first, then explicit references across file boundaries.
    for a, b in zip(all_units, all_units[1:]):
        if PurePosixPath(a.filename).parent == PurePosixPath(b.filename).parent:
            add_edge(a, b)
    for relation in planned.relations:
        left, right = relation.left, relation.right
        for a in by_file[left]:
            for b in by_file[right]:
                add_edge(a, b)
    # Stabilize output independently of relationship-map insertion order.
    order = {unit.id: i for i, unit in enumerate(all_units)}
    return (
        ownership,
        all_units,
        sorted(edges.values(), key=lambda edge: (order[edge[0].id], order[edge[1].id])),
    )


def integration_pages(assignment: Assignment, packet_bytes: int) -> list[Assignment]:
    """Page report atoms while repeating actual source and preserving all findings.

    Args:
        assignment: Integration JSON containing a reports array; left unchanged.
        packet_bytes: Maximum serialized UTF-8 bytes per returned payload.

    Returns:
        The original assignment if it fits. Otherwise advisory navigation is omitted
        before report paging; every page retains identical required source and owned
        sections and requires every enabled provider. The input assignment remains unchanged.

    Raises:
        ValueError: Invalid JSON or source plus one indivisible atom exceeds the ceiling.

    Candidate checks currently reserialize the base packet; paging is not incremental.
    """
    # Keep fitting packets identical; only oversized reports require page identities.
    if len(assignment.payload.encode()) <= packet_bytes:
        return [assignment]
    base = json.loads(assignment.payload)
    # Navigation must yield to source and reports before creating more required work.
    if "overview" in base:
        base["overview"] = omit_navigation(base["overview"])
    serialized = encode(base)
    if len(serialized.encode()) <= packet_bytes:
        return [assignment.model_copy(update={"payload": serialized})]
    atoms = base.pop("reports")
    pages = []
    pending: list[dict[str, Any]] = []
    # Repeat source on every page and never split a summary or finding record.
    for atom in atoms:
        trial = encode({**base, "reports": [*pending, atom]})
        if len(trial.encode()) > packet_bytes and pending:
            pages.append(encode({**base, "reports": pending}))
            pending = []
        pending.append(atom)
        if len(encode({**base, "reports": pending}).encode()) > packet_bytes:
            raise ValueError("Indivisible integration evidence exceeds packet budget")
    if pending:
        pages.append(encode({**base, "reports": pending}))
    # Retain ownership while making every page independently required work.
    return [
        Assignment(
            id=f"{assignment.id}-page-{index:04d}",
            kind=assignment.kind,
            payload=payload,
            sections=assignment.sections,
        )
        for index, payload in enumerate(pages)
    ]


def source_sections(context: ReviewContext, requests: list[SourceRequest]) -> list[dict[str, Any]]:
    """Resolve inclusive Git line ranges from captured versions without live reads.

    Args:
        context: Snapshot indexed by current changed-file paths, including renames.
        requests: One-based inclusive ranges; LEFT selects before, RIGHT selects after.

    Returns:
        Request dictionaries in input order with exact LF-preserving text added.
        Repeated requests remain repeated; no input is mutated.

    Raises:
        ValueError: A path is outside the snapshot, end precedes start, or end is unavailable.
    """
    # Index only captured versions; requests cannot widen the repository boundary.
    files = {f.filename: f for f in context.files}
    sections = []
    sources: dict[tuple[str, str], list[str]] = {}
    # Validate each range before slicing and cache LF records for repeated requests.
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
    context: ReviewContext, assignment: Assignment, history: list[ChunkReview], packet_bytes: int
) -> str:
    """Combine original work, all prior reports and all requested captured source.

    Args:
        context: Immutable source snapshot for resolving requests.
        assignment: Original serialized input, never replaced with a partial packet.
        history: Reports in attempt order, including the newly incomplete report.
        packet_bytes: Maximum serialized UTF-8 bytes for the combined packet.

    Returns:
        JSON for the next provider call without mutating history or assignment.
        Advisory navigation is omitted if needed; required evidence is never removed.

    Raises:
        ValueError: Original JSON is invalid, a requested range is unavailable, or
            the combined packet exceeds the ceiling.
    """
    # Rebuild from original evidence so later attempts cannot forget prior requirements.
    payload = json.loads(assignment.payload)
    payload["prior_attempts"] = [r.model_dump() for r in history]
    payload["requested_source"] = source_sections(context, [q for r in history for q in r.requests])
    # Prefer all requested evidence over advisory navigation on a full continuation.
    serialized = encode(payload)
    if len(serialized.encode()) > packet_bytes and "overview" in payload:
        payload["overview"] = omit_navigation(payload["overview"])
        serialized = encode(payload)
    # Reject the entire continuation if required evidence still cannot fit.
    if len(serialized.encode()) > packet_bytes:
        raise ValueError("Required follow-up context exceeds packet budget")
    return serialized


def attempt_directory(directory: Path, assignment: str, role: str, round_: int) -> Path:
    """Return directory/work/assignment/role/round_ without creating it.

    Directory is the run root; assignment and role must be trusted internal path
    components. Round_ is a zero-based attempt index. This helper does not sanitize IDs.
    """
    return directory / "work" / assignment / role / str(round_)


def advance(
    context: ReviewContext,
    assignment: Assignment,
    history: list[ChunkReview],
    report_json: str,
    anchors: AnchorIndex,
    packet_bytes: int,
) -> tuple[ChunkReview, str | None]:
    """Validate one report and determine the next protocol step without I/O.

    Args:
        context: Immutable snapshot supplying requested source.
        assignment: Original work whose payload remains in every continuation.
        history: Prior validated reports in order, excluding report_json; not mutated.
        report_json: Current provider response or recorded artifact to parse.
        anchors: Read-only diff coordinates compiled from the same snapshot.
        packet_bytes: Maximum UTF-8 size of a continuation payload.

    Returns:
        Parsed report and next JSON payload, or None for the payload when complete.

    Raises:
        FollowupExhausted: The final allowed report remains incomplete, with or without requests.
        IncompleteWithoutRequests: An earlier incomplete report omits source requests.
        ValueError: Schema or anchors are invalid, completion conflicts with requests,
            requested source is unavailable, or the continuation exceeds the packet ceiling.
    """
    # Parse and anchor-check before interpreting a provider completion claim.
    report = ChunkReview.model_validate_json(report_json)
    validate_anchors(anchors, [report])
    # Complete work must have no outstanding required source requests.
    if report.complete:
        if report.requests:
            raise ValueError("Complete report cannot have required source requests")
        return report, None
    # Incomplete work must request available evidence within the remaining attempt bound.
    if len(history) + 1 >= MAX_ATTEMPTS:
        raise FollowupExhausted(
            len(history) + 1, report.requests, {file.filename for file in context.files}
        )
    if not report.requests:
        raise IncompleteWithoutRequests(len(history) + 1)
    return report, followup_payload(context, assignment, [*history, report], packet_bytes)


def run_assignment(
    context: ReviewContext,
    assignment: Assignment,
    role: Role,
    directory: Path,
    anchors: AnchorIndex,
    packet_bytes: int,
) -> int:
    """Run one provider through bounded attempts, preserving input and usage evidence.

    Args:
        context: Immutable snapshot for anchors and source requests.
        assignment: Required work with an internally generated path-safe ID.
        role: Independent provider to call.
        directory: Fresh run root; each attempt gets its own subdirectory.
        anchors: Index compiled from context and shared read-only among workers.
        packet_bytes: Plan ceiling for follow-up JSON, in UTF-8 bytes.

    Returns:
        Number of attempts through the first complete report.

    Raises:
        ValueError: Protocol, capacity or response validation fails, including exhaustion.
        OSError: Attempt artifacts cannot be written or read.
        Exception: Provider errors propagate; call_model retains available usage.

    Writes payloads/reports/usage, prints progress, and may make paid provider calls.
    """
    history: list[ChunkReview] = []
    payload = assignment.payload
    # The shared transition enforces the attempt bound for execution and replay.
    # Persist exact input before each paid call.
    while True:
        round_ = len(history)
        target = attempt_directory(directory, assignment.id, role, round_)
        target.mkdir(parents=True, exist_ok=True)
        (target / "payload.json").write_text(payload)
        print(
            f"Review attempt: assignment={assignment.id} role={role} attempt={round_}", flush=True
        )
        call_model(role, payload, target, kind=assignment.kind)
        report, next_payload = advance(
            context,
            assignment,
            history,
            (target / f"{role}.json").read_text(),
            anchors,
            packet_bytes,
        )
        # Keep prior findings available and stop only on a validated terminal report.
        history.append(report)
        if next_payload is None:
            return len(history)
        payload = next_payload


def save_manifest(directory: Path, manifest: Manifest) -> None:
    """Replace directory/manifest.json atomically with coordinator-owned state.

    Directory must already exist. Manifest is serialized to manifest.tmp first;
    workers must never call this concurrently. OSError propagates on either write
    or replacement failure; a failed replacement may leave the temporary file.
    """
    # Readers see the previous complete manifest until replacement succeeds.
    temporary = directory / "manifest.tmp"
    temporary.write_text(manifest.model_dump_json())
    temporary.replace(directory / "manifest.json")


def run_batch(
    context: ReviewContext,
    manifest: Manifest,
    assignments: list[Assignment],
    directory: Path,
    anchors: AnchorIndex,
    packet_bytes: int,
) -> None:
    """Run enabled providers with two workers and persist every terminal result.

    Args:
        context: Immutable snapshot shared by all workers.
        manifest: Coordinator-owned state; extended and updated in place.
        assignments: New work in deterministic order, with unique internal IDs.
        directory: Existing run root for manifest and isolated attempt artifacts.
        anchors: Read-only coordinates compiled from context.
        packet_bytes: Selected plan ceiling passed to every attempt.

    Raises:
        ValueError: Total work exceeds MAX_WORK, or any worker fails after draining.
        OSError: Manifest persistence fails; already-started workers still finish.

    Work is recorded before submission. Worker errors are logged and converted to
    failed completions with best-effort failure.json diagnostics. Exhaustion preserves
    its validated report count; queued work is not cancelled after a failure. All in-flight
    calls finish so available usage is retained before the batch can fail.
    """
    # Record the full required batch before any provider can start spending.
    if len(manifest.assignments) + len(assignments) > MAX_WORK:
        raise ValueError("Total review work budget exceeded")
    manifest.assignments.extend(assignment_record(a) for a in assignments)
    completions = [
        Completion(assignment=a.id, role=role) for a in assignments for role in manifest.providers
    ]
    manifest.results.extend(completions)
    save_manifest(directory, manifest)
    failed = False
    # Isolated attempt directories let two workers run without sharing artifact writes.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(
                run_assignment,
                context,
                assignment,
                completion.role,
                directory,
                anchors,
                packet_bytes,
            ): completion
            for assignment in assignments
            for completion in completions
            if completion.assignment == assignment.id
        }
        # Only the coordinator mutates completion state; drain failures to retain usage.
        for future in as_completed(futures):
            completion = futures[future]
            try:
                completion.attempts = future.result()
                completion.status = "complete"
            except Exception as error:
                if isinstance(error, (FollowupExhausted, IncompleteWithoutRequests)):
                    completion.attempts = error.attempts
                report_failure(
                    error,
                    attempt_directory(directory, completion.assignment, completion.role, 0).parent
                    / "failure.json",
                    assignment=completion.assignment,
                    role=completion.role,
                    phase="worker",
                )
                completion.status = "failed"
                failed = True
            save_manifest(directory, manifest)
    # Partial success must never allow downstream approval.
    if failed:
        raise ValueError("Required provider work failed")


def read_batch(
    context: ReviewContext,
    assignments: list[Assignment],
    manifest: Manifest,
    directory: Path,
    anchors: AnchorIndex,
    packet_bytes: int,
) -> dict[tuple[str, str], Review]:
    """Replay recorded attempts and return validated reports keyed by work identity.

    Args:
        context: Original immutable snapshot; never fetched again here.
        assignments: Required work and exact initial payloads in plan order.
        manifest: Recorded completions; read without mutation.
        directory: Existing run artifacts, including each payload and provider report.
        anchors: Read-only snapshot coordinates used by execution too.
        packet_bytes: Plan ceiling for reconstructing continuation inputs.

    Returns:
        Reports keyed by (assignment ID, provider), retaining all attempt findings
        and the last summary. Performs file reads only, with no provider calls.

    Raises:
        ValueError: Results are missing/duplicated/failed, schema or protocol is invalid,
            payloads differ, or recorded attempt boundaries disagree with completion.
        OSError: Required evidence cannot be read.
    """
    # Require one successful completion for each planned assignment/provider identity.
    reports: dict[tuple[str, str], Review] = {}
    for assignment in assignments:
        for role in manifest.providers:
            matches = [
                r for r in manifest.results if r.assignment == assignment.id and r.role == role
            ]
            if len(matches) != 1 or matches[0].status != "complete" or not matches[0].attempts:
                raise ValueError("Missing, duplicate or failed worker result")
            completion = matches[0]
            history: list[ChunkReview] = []
            payload = assignment.payload
            # Reconstruct each input through the same transition used during execution.
            for round_ in range(completion.attempts):
                target = attempt_directory(directory, assignment.id, role, round_)
                if (target / "payload.json").read_text() != payload:
                    raise ValueError("Attempt evidence differs from required work")
                report, next_payload = advance(
                    context,
                    assignment,
                    history,
                    (target / f"{role}.json").read_text(),
                    anchors,
                    packet_bytes,
                )
                history.append(report)
                terminal = round_ == completion.attempts - 1
                if terminal != (next_payload is None):
                    raise ValueError("Recorded attempts disagree with required follow-up protocol")
                if next_payload is not None:
                    payload = next_payload
            # Preserve every attempt finding, using only the final assessment as summary.
            reports[(assignment.id, role)] = CompletedReview(
                complete=True,
                summary=history[-1].summary,
                findings=[f for r in history for f in r.findings],
            )
    return reports


def preflight(context: ReviewContext, planned: Plan, directory: Path, providers: Providers) -> None:
    """Count enabled providers on every local scope before purchasing any review.

    Args:
        providers: Required reviewer set resolved before any provider calls.
        context: Snapshot that produced planned.
        planned: Candidate plan whose initial inputs will be counted serially.
        directory: Run root; provider setup uses preflight/assignment-ID directories.

    Raises:
        CapacityError: Any input exceeds provider-counted token or byte policy.
        Exception: Filesystem and provider failures propagate; no generation occurs.

    call_model may write diagnostic artifacts on failure, including capacity rejection.
    """
    # Count every enabled provider format before any initial review is generated.
    for assignment in local_assignments(context, planned):
        for role in providers:
            call_model(
                role,
                assignment.payload,
                directory / "preflight" / assignment.id,
                kind=assignment.kind,
                preflight=True,
            )


def review_chunks(context: ReviewContext, directory: Path) -> None:
    """Execute one fresh manifest-backed review, including required integration work.

    Args:
        context: Captured snapshot; source remains immutable throughout the run.
        directory: Run root without a manifest or work directory; preflight may create it.

    Raises:
        ValueError: Artifacts already exist, planning/capacity fails, or required work
            cannot complete. Capacity overflow selects one smaller plan before retrying.
        Exception: Unhandled provider or filesystem errors propagate to the CLI.

    Writes the manifest and all attempt artifacts. Preflight, enabled local providers,
    local replay, integration execution and integration replay run in that order.
    """
    # Reject reused workspaces instead of mixing attempts and their cost accounting.
    if (directory / "manifest.json").exists() or (directory / "work").exists():
        raise ValueError("Review run requires a fresh artifact directory")
    # Resolve credentials once; failed configured providers never become disabled.
    providers = configured_providers()
    # Select a capacity policy before committing work to a manifest.
    planned = plan(context)
    try:
        preflight(context, planned, directory, providers)
    except CapacityError:
        planned = plan(context, packet_bytes=CAPACITY_PACKET_BYTES)
        preflight(context, planned, directory, providers)
    # Bind required work to the chosen plan and compile shared coordinates once.
    manifest = Manifest(
        providers=providers,
        plan_digest=plan_digest(planned, providers),
        packet_bytes=planned.packet_bytes,
        assignments=[],
        results=[],
    )
    anchors = compile_anchors(context)
    # Validate local evidence before using its reports to plan integration.
    local = local_assignments(context, planned)
    run_batch(context, manifest, local, directory, anchors, planned.packet_bytes)
    reports = read_batch(context, local, manifest, directory, anchors, planned.packet_bytes)
    integration = integration_assignments(context, planned, reports, manifest.providers)
    run_batch(context, manifest, integration, directory, anchors, planned.packet_bytes)
    read_batch(context, integration, manifest, directory, anchors, planned.packet_bytes)


def load_reviews(context: ReviewContext, directory: Path) -> CoveredReviews:
    """Validate exact snapshot coverage before consolidation or publication.

    Args:
        context: Captured snapshot that must match the manifest's reconstructed plan.
        directory: Run root containing the manifest and immutable attempt artifacts.

    Returns:
        Trusted internal reports in deterministic assignment/provider order, explicitly
        identified whole-review summaries, snapshot hash and human-readable coverage.

    Raises:
        ValueError: Plan, ownership, completion keys, artifacts or protocol disagree.
        OSError: A required artifact is absent or unreadable.

    Replans and reads evidence without provider calls or writes. The return type is
    application data, not an unforgeable capability for arbitrary Python callers.
    """
    # Rebuild the plan from persisted policy and require its exact snapshot binding.
    manifest = Manifest.model_validate_json((directory / "manifest.json").read_text())
    planned = plan(context, packet_bytes=manifest.packet_bytes)
    if manifest.plan_digest != plan_digest(planned, manifest.providers):
        raise ValueError("Coverage plan does not match the captured snapshot")
    anchors = compile_anchors(context)
    local = local_assignments(context, planned)
    reports = read_batch(context, local, manifest, directory, anchors, planned.packet_bytes)
    integration = integration_assignments(context, planned, reports, manifest.providers)
    # Compare exact ownership and provider keys, not a report-count approximation.
    expected = local + integration
    keys = [(a.id, role) for a in expected for role in manifest.providers]
    actual = [(r.assignment, r.role) for r in manifest.results]
    if manifest.assignments != [assignment_record(a) for a in expected] or sorted(keys) != sorted(
        actual
    ):
        raise ValueError("Coverage assignments are missing, duplicated or unexpected")
    reports.update(
        read_batch(context, integration, manifest, directory, anchors, planned.packet_bytes)
    )
    # Flatten only after replay succeeds; carry explicit identity for presentation.
    return CoveredReviews(
        providers=manifest.providers,
        reviews=[reports[key] for key in keys],
        snapshot=planned.snapshot,
        independent_summaries=[
            IndependentSummary(
                assignment=assignment.id,
                provider=role,
                summary=reports[(assignment.id, role)].summary,
            )
            for assignment in local
            if assignment.kind == "whole"
            for role in manifest.providers
        ],
        summary=f"{len(local)} local scopes; {len(expected)} local/integration assignments "
        f"completed by {', '.join(manifest.providers)}. Snapshot `{planned.snapshot}`. "
        "Exact section ownership and follow-up evidence are in the workflow coverage manifest. "
        "Completion does not establish model understanding.",
    )
