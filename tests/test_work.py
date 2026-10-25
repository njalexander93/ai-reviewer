"""Exercise bounded orchestration and publication with synthetic provider responses.

Expected behavior comes from the agreed two-provider coverage and no-partial-
approval rules. Fakes replace paid calls; planning, files and validation stay real.
"""

import json
from pathlib import Path

import pytest

from ai_reviewer import consolidation, work
from ai_reviewer.chunking import plan
from ai_reviewer.contracts import (
    ChangedFile,
    ChunkReview,
    Consolidation,
    FinalConsolidation,
    Group,
    ReviewContext,
    reconcile,
)
from ai_reviewer.github import publish
from tests.test_github import Server, bundle


@pytest.fixture(autouse=True)
def configured_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep existing dual-provider fixtures independent of developer credentials."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-openai")


def large_context() -> ReviewContext:
    """Return an isolated synthetic snapshot that exceeds the whole-packet ceiling.

    Two added source/test files each carry 2,000 LF-terminated lines and full patches.
    No repository contents or provider responses are read; callers may mutate the fixture.
    """
    context, _, _ = bundle(Server())
    context.files = [
        ChangedFile(
            filename=name,
            previous_filename=None,
            status="added",
            patch="@@ -0,0 +1,2000 @@\n" + ("+" + "x" * 100 + "\n") * 2000,
            before="",
            after=("x" * 100 + "\n") * 2000,
            before_mode=None,
            after_mode="100644",
        )
        for name in ("src/app.py", "tests/test_app.py")
    ]
    return context


def fake_model(
    role: str,
    payload: str,
    directory: Path,
    *,
    kind: str = "whole",
    preflight: bool = False,
    repository: object | None = None,
) -> None:
    """Write synthetic reports at the provider boundary without network calls.

    Args:
        role: Reviewer writes an empty complete report; consolidate groups each source ID.
        payload: JSON input whose sources determine the synthetic consolidation groups.
        directory: Artifact location, created if needed.
        kind: Accepted for call_model compatibility; it does not change the fake's behavior.
        preflight: Return without files or generation when true.
        repository: Optional read-only snapshot accepted for tool-enabled runs.

    Writes role.json and a null usage artifact. Invalid JSON and filesystem errors
    propagate. These application fixtures do not establish real provider behavior.
    """
    if preflight:
        return
    directory.mkdir(parents=True, exist_ok=True)
    data = json.loads(payload)
    report = (
        Consolidation(
            summary="Assessments agree.",
            groups=[Group(source_ids=[key]) for key in data.get("sources", {})],
        )
        if role in ("haiku", "luna")
        else ChunkReview(
            complete=True, summary="Reviewed actual supplied source.", findings=[], requests=[]
        )
    )
    (directory / f"{role}.json").write_text(report.model_dump_json())
    (directory / f"{role}.usage.json").write_text("null")


def test_large_pipeline_requires_both_providers_and_actual_integration_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """All sections reach both providers and cross-chunk pairs carry actual source."""
    context = large_context()
    monkeypatch.setattr(work, "call_model", fake_model)
    monkeypatch.setattr(consolidation, "call_model", fake_model)
    work.review_chunks(context, tmp_path)
    covered = work.load_reviews(context, tmp_path)
    reports = covered.reviews
    manifest = work.Manifest.model_validate_json((tmp_path / "manifest.json").read_text())
    assert len(reports) == 2 * len(manifest.assignments)
    assert all(r.status == "complete" for r in manifest.results)
    edges = [a for a in manifest.assignments if a.id.startswith("edge-")]
    assert edges
    for edge in edges:
        packet = json.loads(
            (work.attempt_directory(tmp_path, edge.id, "claude", 0) / "payload.json").read_text()
        )
        assert packet["reports"]
        assert all(unit["after"] for unit in packet["scope"]["units"])
    consolidation.consolidate_chunks(context, tmp_path)
    result = FinalConsolidation.model_validate_json((tmp_path / "consolidate.json").read_text())
    usage_before = {str(p): p.read_text() for p in tmp_path.rglob("*.usage.json")}
    with pytest.raises(ValueError, match="fresh artifacts"):
        consolidation.consolidate_chunks(context, tmp_path)
    assert {str(p): p.read_text() for p in tmp_path.rglob("*.usage.json")} == usage_before
    server = Server()
    server.files = [{"filename": f.filename, "patch": f.patch} for f in context.files]
    assert publish(server.gh, context, covered, result)
    assert len(server.writes) == 1
    assert server.writes[0][1]["event"] == "APPROVE"
    assert "Completion does not establish model understanding" in server.writes[0][1]["body"]
    # The existing stale revision protection also applies to chunked publication.
    server.pr["head"]["sha"] = "f" * 40
    with pytest.raises(ValueError, match="Stale"):
        publish(server.gh, context, covered, result)
    assert len(server.writes) == 1


@pytest.mark.parametrize(
    "damage", ["missing", "duplicate", "failed", "payload", "snapshot", "plan"]
)
def test_corrupt_coverage_cannot_approve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, damage: str
) -> None:
    """Deleting, inventing or changing completed work fails the coverage boundary."""
    context = large_context()
    monkeypatch.setattr(work, "call_model", fake_model)
    work.review_chunks(context, tmp_path)
    # Alter one previously valid coverage invariant, then require replay to fail.
    path = tmp_path / "manifest.json"
    manifest = work.Manifest.model_validate_json(path.read_text())
    if damage == "missing":
        manifest.results.pop()
    elif damage == "duplicate":
        manifest.results.append(manifest.results[0])
    elif damage == "failed":
        manifest.results[0].status = "failed"
    elif damage == "plan":
        manifest.plan_digest = "tampered"
    elif damage == "snapshot":
        context.head = "f" * 40
    else:
        next((tmp_path / "work").rglob("payload.json")).write_text("{}")
    path.write_text(manifest.model_dump_json())
    with pytest.raises(ValueError):
        work.load_reviews(context, tmp_path)


@pytest.mark.parametrize("unresolved", [False, True])
def test_cross_chunk_source_requests_are_resolved_or_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, unresolved: bool
) -> None:
    """Required other-file source reaches the follow-up; exhaustion prohibits approval."""
    context = large_context()
    assignment = work.local_assignments(context, plan(context))[0]
    seen = []

    def model(role: str, payload: str, directory: Path, **kwargs: object) -> None:
        """Write a required source request, then complete only when unresolved is false.

        Role selects the artifact filename; payload is recorded in seen and its requested
        text is checked on continuation. Directory must exist. Extra call options are
        ignored. The written JSON is synthetic, not a recorded provider response.
        """
        data = json.loads(payload)
        seen.append(data)
        report = ChunkReview(
            complete=False,
            summary="Need the other component",
            findings=[],
            requests=[
                {
                    "path": "tests/test_app.py",
                    "side": "RIGHT",
                    "start": 100,
                    "end": 101,
                    "reason": "Check changed caller",
                }
            ],
        )
        if len(seen) > 1 and not unresolved:
            assert data["requested_source"][0]["text"] == ("x" * 100 + "\n") * 2
            report = ChunkReview(complete=True, summary="Caller checked", findings=[], requests=[])
        (directory / f"{role}.json").write_text(report.model_dump_json())

    monkeypatch.setattr(work, "call_model", model)
    # Exercise successful continuation or exhaustion through the real attempt loop.
    if unresolved:
        with pytest.raises(ValueError, match="exhausted"):
            work.run_assignment(
                context, assignment, "claude", tmp_path, work.compile_anchors(context), 600_000
            )
        assert len(seen) == 3
    else:
        assert (
            work.run_assignment(
                context, assignment, "claude", tmp_path, work.compile_anchors(context), 600_000
            )
            == 2
        )
        assert seen[1]["prior_attempts"][0]["requests"]


def test_failed_provider_leaves_manifest_incomplete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One successful provider cannot mask the other provider's failure."""

    def model(role: str, payload: str, directory: Path, **kwargs: object) -> None:
        """Accept preflight, then raise TimeoutError for OpenAI generation.

        Role/payload/directory follow fake_model; kwargs supplies the preflight flag.
        Claude generation delegates to fake_model with local framing and writes evidence.
        """
        if kwargs.get("preflight"):
            return
        if role == "openai":
            raise TimeoutError()
        fake_model(role, payload, directory, kind="local")

    # Run the real batch with one failing generation provider and inspect persisted states.
    monkeypatch.setattr(work, "call_model", model)
    with pytest.raises(ValueError, match="failed"):
        work.review_chunks(large_context(), tmp_path)
    manifest = work.Manifest.model_validate_json((tmp_path / "manifest.json").read_text())
    assert {r.status for r in manifest.results if r.role == "claude"} == {"complete"}
    assert {r.status for r in manifest.results if r.role == "openai"} == {"failed"}
    with pytest.raises(ValueError):
        work.load_reviews(large_context(), tmp_path)


def test_many_reports_preserve_global_source_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """More than two reports and 80 findings retain distinct global IDs and severity."""
    from ai_reviewer.contracts import CompletedReview, Finding

    context = large_context()
    monkeypatch.setattr(work, "call_model", fake_model)
    monkeypatch.setattr(consolidation, "call_model", fake_model)
    work.review_chunks(context, tmp_path)
    covered = work.load_reviews(context, tmp_path)
    # Independent fixture: 12 reports with 10 distinct original observations each.
    reports = [
        CompletedReview(
            complete=True,
            summary=f"Report {i}",
            findings=[
                Finding(
                    path="src/app.py",
                    line=j + 1,
                    side="RIGHT",
                    title=f"Original {i}:{j}",
                    body="Original evidence",
                    blocking=i % 2 == 0,
                )
                for j in range(10)
            ],
        )
        for i in range(12)
    ]
    monkeypatch.setattr(
        consolidation, "load_reviews", lambda *args: covered.model_copy(update={"reviews": reports})
    )
    consolidation.consolidate_chunks(context, tmp_path)
    result = FinalConsolidation.model_validate_json((tmp_path / "consolidate.json").read_text())
    ids = [source for group in result.groups for source in group.source_ids]
    assert len(ids) == len(set(ids)) == 120
    assert "0:0" in ids and "11:9" in ids
    grouped = reconcile(reports, result)
    assert sum(f.blocking for group in grouped for f in group) == 60
    assert {f.title for group in grouped for f in group} == {
        f"Original {i}:{j}" for i in range(12) for j in range(10)
    }


def test_known_cross_chunk_defect_matches_whole_packet_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A synthetic signature/caller defect stays observable in actual integration source.

    This compares evidence availability with a deterministic recognizer. It does
    not measure model judgment or establish model-quality equivalence.
    """
    from ai_reviewer import chunking
    from ai_reviewer.contracts import Finding
    from ai_reviewer.github import rebuild_patch

    context, _, _ = bundle(Server())
    padding = "# padding\n" * 160
    texts = {
        "parse.py": "def parse(value, strict):\n    return value\n" + padding,
        "test_parse.py": 'result = parse("x")\n' + padding,
    }
    context.files = [
        ChangedFile(
            filename=name,
            previous_filename=None,
            status="added",
            patch=rebuild_patch("", source),
            before="",
            after=source,
            before_mode=None,
            after_mode="100644",
        )
        for name, source in texts.items()
    ]
    assert plan(context).whole is not None
    expected = Finding(
        path="test_parse.py",
        line=1,
        side="RIGHT",
        title="Required argument missing",
        body="parse now requires strict, but this caller supplies only value.",
        blocking=True,
    )

    def recognize(payload: str) -> list:
        """Return the planted finding only when payload contains both incompatible snippets.

        Payload is whole-context or scoped JSON. The returned list contains the fixture's
        original expected object, not a model inference; only supplied after text is read.
        """
        data = json.loads(payload)
        files = data.get("files", data.get("scope", {}).get("units", []))
        actual = "\n".join(f["after"] for f in files)
        return (
            [expected] if "def parse(value, strict):" in actual and 'parse("x")' in actual else []
        )

    whole_findings = recognize(context.model_dump_json())
    assert whole_findings == [expected]
    # Force partitioning only for this small comparison fixture.
    monkeypatch.setattr(chunking, "CHUNK_BYTES", 5000)
    assert len(plan(context, packet_bytes=200_000).chunks) == 2

    def model(role: str, payload: str, directory: Path, **kwargs: object) -> None:
        """Write recognizer findings or force chunking at the whole-packet preflight.

        Role selects the output filename; payload supplies captured source; directory
        must exist. kwargs carries kind/preflight. Whole preflight raises CapacityError,
        local preflight returns, and generation writes a synthetic complete ChunkReview.
        """
        if kwargs.get("preflight"):
            if kwargs.get("kind") == "whole":
                raise work.CapacityError("Synthetic forced partition")
            return
        report = ChunkReview(
            complete=True,
            summary="Synthetic evidence comparison",
            findings=recognize(payload),
            requests=[],
        )
        (directory / f"{role}.json").write_text(report.model_dump_json())

    monkeypatch.setattr(work, "call_model", model)
    work.review_chunks(context, tmp_path)
    reports = work.load_reviews(context, tmp_path).reviews
    found = [f for r in reports for f in r.findings]
    assert found == [expected, expected]


def test_accumulated_findings_are_paged_with_actual_integration_source() -> None:
    """Three legal 40-finding attempts per provider cannot overflow integration input."""
    from ai_reviewer.contracts import CompletedReview, Finding

    context = large_context()
    planned = plan(context)
    reports = [
        CompletedReview(
            complete=True,
            summary="Retained original assessment",
            findings=[
                Finding(
                    path="src/app.py",
                    line=1,
                    side="RIGHT",
                    title=f"Report {i} finding {j}",
                    body="x" * 1200,
                    blocking=True,
                )
                for j in range(120)
            ],
        )
        for i in range(2 * len(planned.chunks))
    ]
    # Force report paging, then require actual source and every distinct finding on its pages.
    assignments = work.integration_assignments(
        context,
        planned,
        {
            (c.id, role): reports[i * 2 + j]
            for i, c in enumerate(planned.chunks)
            for j, role in enumerate(("claude", "openai"))
        },
        ("claude", "openai"),
    )
    pages = [a for a in assignments if a.id.startswith("integration-0000")]
    assert len(pages) > 1
    assert all(len(a.payload.encode()) <= 600_000 for a in assignments)
    findings = []
    for page in pages:
        packet = json.loads(page.payload)
        assert packet["scope"]["units"] == [u.model_dump() for u in planned.chunks[0].units]
        assert all(r["assignment"] == planned.chunks[0].id for r in packet["reports"])
        assert {r["provider"] for r in packet["reports"]} <= {"claude", "openai"}
        findings.extend(r["finding"] for r in packet["reports"] if "finding" in r)
    assert len(findings) == 240
    assert {f["title"] for f in findings} == {
        f"Report {i} finding {j}" for i in range(2) for j in range(120)
    }


def test_reordered_reports_keep_assignment_identity() -> None:
    """Changing mapping insertion order cannot attach another chunk's reports."""
    from ai_reviewer.contracts import Review

    context = large_context()
    planned = plan(context)
    reports = {
        (c.id, role): Review(complete=True, summary=f"{c.id}/{role}", findings=[])
        for c in planned.chunks
        for role in ("claude", "openai")
    }
    # Reorder storage only; scope/report identity must remain unchanged.
    reversed_reports = dict(reversed(list(reports.items())))
    actual = work.integration_assignments(context, planned, reversed_reports, ("claude", "openai"))
    for i, chunk in enumerate(planned.chunks):
        packet = json.loads(next(a.payload for a in actual if a.id == f"integration-{i:04d}"))
        assert [r["summary"] for r in packet["reports"]] == [
            f"{chunk.id}/claude",
            f"{chunk.id}/openai",
        ]
    assert actual == work.integration_assignments(context, planned, reports, ("claude", "openai"))


def test_manifest_retains_hashes_and_rejects_tampered_assignment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The manifest stays small while payload edits still prevent consolidation."""
    context = large_context()
    monkeypatch.setattr(work, "call_model", fake_model)
    work.review_chunks(context, tmp_path)
    path = tmp_path / "manifest.json"
    data = json.loads(path.read_text())
    assert path.stat().st_size < 50_000
    assert all("payload" not in a and len(a["payload_digest"]) == 64 for a in data["assignments"])
    data["assignments"][0]["payload_digest"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Coverage assignments"):
        work.load_reviews(context, tmp_path)


@pytest.mark.parametrize("rejecting_role", ["claude", "openai"])
def test_token_overflow_replans_before_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rejecting_role: str
) -> None:
    """Either tokenizer can force chunks without spending on the rejected whole packet."""
    context = large_context()
    context.files = context.files[:1]
    original = context.model_dump_json()
    assert plan(context).whole == original
    generated = []

    def model(
        role: str, payload: str, directory: Path, *, kind: str = "whole", preflight: bool = False
    ) -> None:
        """Reject only the selected role's whole preflight, then delegate to fake_model.

        Role/payload/directory/kind/preflight follow fake_model. Generated input strings
        are retained in generated. CapacityError is independent of byte length here;
        all other preflights return without artifacts.
        """
        if preflight:
            if role == rejecting_role and kind == "whole":
                raise work.CapacityError("Provider input capacity budget exceeded")
            return
        generated.append(payload)
        fake_model(role, payload, directory, kind=kind)

    monkeypatch.setattr(work, "call_model", model)
    work.review_chunks(context, tmp_path)
    covered = work.load_reviews(context, tmp_path)
    assert covered.reviews
    assert original not in generated
    assert all("scope" in json.loads(payload) for payload in generated)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["packet_bytes"] == 200_000
    # Replay cannot silently switch back to the byte-only whole assignment.
    manifest["packet_bytes"] = 600_000
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Coverage plan"):
        work.load_reviews(context, tmp_path)


def test_stage_compiles_anchors_once_for_all_workers_and_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Review and loader each compile one index, independent of assignment count."""
    context = large_context()
    compile_original = work.compile_anchors
    compiled = []

    def compile_once(context: ReviewContext) -> work.AnchorIndex:
        """Compile context with the original implementation and record the returned index.

        Returns the same read-only index, without replacing validation or changing context.
        Only the test's compiled list is mutated.
        """
        anchors = compile_original(context)
        compiled.append(anchors)
        return anchors

    # Count actual index construction during review/replay and verify its read-only boundary.
    monkeypatch.setattr(work, "compile_anchors", compile_once)
    monkeypatch.setattr(work, "call_model", fake_model)
    work.review_chunks(context, tmp_path)
    assert len(compiled) == 1
    work.load_reviews(context, tmp_path)
    assert len(compiled) == 2
    with pytest.raises(TypeError):
        compiled[0]["invented.py"] = frozenset()


def test_integration_budget_matches_finished_packets_without_trial_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fit accounting agrees with actual source/report bytes and serializes only final scopes."""
    from ai_reviewer.chunking import Chunk, PacketSizer
    from ai_reviewer.contracts import Review

    context = large_context()
    context.files[0].after = "tests/test_app.py\n" + context.files[0].after
    planned = plan(context)
    reports = {
        (c.id, role): Review(complete=True, summary='é"\n' + c.id, findings=[])
        for c in planned.chunks
        for role in ("claude", "openai")
    }
    original_budget, original_payload = PacketSizer.budget, Chunk.payload
    budgets = {}
    serialized = []

    def budget(self: PacketSizer, id: str, extra: dict | None = None) -> object:
        """Call the original PacketSizer.budget and retain its live counter by scope id.

        Id/extra follow that method's contract. Returns its actual mutable ArrayBudget;
        the test's budgets map observes later admission without reimplementing accounting.
        """
        result = original_budget(self, id, extra)
        budgets[id] = result
        return result

    def payload(self: Chunk, metadata: object) -> str:
        """Record this chunk's ID, then return its original payload serialization.

        Metadata follows Chunk.payload's contract. Only the test's serialized list changes;
        source and byte construction remain the production implementation.
        """
        serialized.append(self.id)
        return original_payload(self, metadata)

    # Observe final serialization and compare its bytes to the retained incremental budget.
    monkeypatch.setattr(PacketSizer, "budget", budget)
    monkeypatch.setattr(Chunk, "payload", payload)
    assignments = work.integration_assignments(context, planned, reports, ("claude", "openai"))
    edges = [a for a in assignments if a.id.startswith("edge-")]
    assert edges
    assert any(json.loads(a.payload)["reference_source"] for a in edges)
    for assignment in edges:
        assert budgets[assignment.id].size == len(assignment.payload.encode())
        assert len(assignment.payload.encode()) <= 600_000
    assert len(serialized) == len(planned.chunks) + len(edges)


@pytest.mark.parametrize("file_count", [1, 2])
def test_capacity_fallback_bounds_integration_and_followup_payloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, file_count: int
) -> None:
    """A dense-input counter rejects every oversized call, including later integration."""
    from ai_reviewer.contracts import SourceRequest

    context = large_context()
    context.files = context.files[:file_count]
    assert (plan(context).whole is None) == (file_count == 2)
    generated = []
    requested = set()

    def model(
        role: str, payload: str, directory: Path, *, kind: str = "whole", preflight: bool = False
    ) -> None:
        """Reject oversized input, then request source once per local provider assignment.

        Role/payload/directory/kind/preflight follow fake_model. CapacityError is raised
        above 200,000 UTF-8 bytes, even on preflight. Accepted preflight writes nothing.
        Generation records input, writes one synthetic request for local work, and delegates
        later reports to fake_model. The requested set tracks each assignment/provider pair.
        """
        if len(payload.encode()) > 200_000:
            raise work.CapacityError("Dense synthetic input exceeds capacity")
        if preflight:
            return
        generated.append((kind, payload))
        key = (directory.parent, role)
        if kind == "local" and key not in requested:
            requested.add(key)
            report = ChunkReview(
                complete=False,
                summary="Need source",
                findings=[],
                requests=[
                    SourceRequest(
                        path=context.files[0].filename,
                        side="RIGHT",
                        start=1,
                        end=2,
                        reason="Check caller",
                    )
                ],
            )
            (directory / f"{role}.json").write_text(report.model_dump_json())
        else:
            fake_model(role, payload, directory, kind=kind)

    # Run fallback and real artifact replay under the synthetic byte-sensitive provider gate.
    monkeypatch.setattr(work, "call_model", model)
    work.review_chunks(context, tmp_path)
    assert work.load_reviews(context, tmp_path).reviews
    assert any(kind == "integration" for kind, _ in generated)
    assert any("requested_source" in json.loads(payload) for _, payload in generated)
    assert all(len(payload.encode()) <= 200_000 for _, payload in generated)


def test_small_review_consolidates_once_with_original_assessments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A small complete review includes both independent summaries in its one Haiku call."""
    context, _, _ = bundle(Server())
    calls = []

    def model(role: str, payload: str, directory: Path, **kwargs: object) -> None:
        """Record decoded consolidation inputs, then delegate every call to fake_model.

        Role/payload/directory and kwargs follow fake_model's contract. The calls list
        changes only for consolidation; artifact creation and preflight behavior remain shared.
        """
        if role in ("haiku", "luna"):
            calls.append(json.loads(payload))
        fake_model(role, payload, directory, **kwargs)

    # Run the real small-review flow and inspect the single consolidation request.
    monkeypatch.setattr(work, "call_model", model)
    monkeypatch.setattr(consolidation, "call_model", model)
    work.review_chunks(context, tmp_path)
    consolidation.consolidate_chunks(context, tmp_path)
    assert len(calls) == 1
    assert calls[0]["reviews"] == ["Reviewed actual supplied source."] * 2
