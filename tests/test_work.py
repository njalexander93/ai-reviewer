"""Exercise bounded orchestration and publication with synthetic provider responses.

Expected behavior comes from the agreed two-provider coverage and no-partial-
approval rules. Fakes replace paid calls; planning, files and validation stay real.
"""

import json
from pathlib import Path

import pytest

from ai_reviewer import work
from ai_reviewer.chunking import plan
from ai_reviewer.contracts import (
    ChangedFile,
    ChunkReview,
    Consolidation,
    FinalConsolidation,
    Group,
    ReviewContext,
)
from ai_reviewer.github import publish
from tests.test_github import Server, bundle


def large_context() -> ReviewContext:
    """Provide two changed components whose combined packet exceeds 600 KB."""
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
    role: str, payload: str, directory: Path, *, kind: str = "whole", preflight: bool = False
) -> None:
    """Return complete source-preserving responses without paid calls or GitHub writes."""
    if preflight:
        return
    directory.mkdir(parents=True, exist_ok=True)
    data = json.loads(payload)
    report = (
        Consolidation(
            summary="Assessments agree.",
            groups=[Group(source_ids=[key]) for key in data.get("sources", {})],
        )
        if role == "consolidate"
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
    work.consolidate_chunks(context, tmp_path)
    result = FinalConsolidation.model_validate_json((tmp_path / "consolidate.json").read_text())
    usage_before = {str(p): p.read_text() for p in tmp_path.rglob("*.usage.json")}
    with pytest.raises(ValueError, match="fresh artifacts"):
        work.consolidate_chunks(context, tmp_path)
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
        """Request a known line, then inspect the exact captured bytes on continuation."""
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
    if unresolved:
        with pytest.raises(ValueError, match="exhausted"):
            work.run_assignment(
                context, assignment, "claude", tmp_path, work.compile_anchors(context)
            )
        assert len(seen) == 3
    else:
        assert (
            work.run_assignment(
                context, assignment, "claude", tmp_path, work.compile_anchors(context)
            )
            == 2
        )
        assert seen[1]["prior_attempts"][0]["requests"]


def test_failed_provider_leaves_manifest_incomplete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One successful provider cannot mask the other provider's failure."""

    def model(role: str, payload: str, directory: Path, **kwargs: object) -> None:
        """Fail OpenAI while allowing all scheduled Claude evidence to finish."""
        if role == "openai":
            raise TimeoutError()
        fake_model(role, payload, directory, kind="local")

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
        work, "load_reviews", lambda *args: covered.model_copy(update={"reviews": reports})
    )
    work.consolidate_chunks(context, tmp_path)
    result = FinalConsolidation.model_validate_json((tmp_path / "consolidate.json").read_text())
    ids = [source for group in result.groups for source in group.source_ids]
    assert len(ids) == len(set(ids)) == 120
    assert "0:0" in ids and "11:9" in ids
    grouped = work.reconcile(reports, result, len(reports))
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
        """Recognize only the deliberately planted incompatibility in actual source."""
        data = json.loads(payload)
        files = data.get("files", data.get("scope", {}).get("units", []))
        actual = "\n".join(f["after"] for f in files)
        return (
            [expected] if "def parse(value, strict):" in actual and 'parse("x")' in actual else []
        )

    whole_findings = recognize(context.model_dump_json())
    assert whole_findings == [expected]
    # Force partitioning only for this small comparison fixture.
    monkeypatch.setattr(chunking, "PACKET_BYTES", 500)
    monkeypatch.setattr(chunking, "CHUNK_BYTES", 5000)
    assert len(plan(context).chunks) == 2

    def model(role: str, payload: str, directory: Path, **kwargs: object) -> None:
        """Return the recognizer's evidence-backed finding in the application schema."""
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
    assignments = work.integration_assignments(
        context,
        planned,
        {
            (c.id, role): reports[i * 2 + j]
            for i, c in enumerate(planned.chunks)
            for j, role in enumerate(("claude", "openai"))
        },
    )
    pages = [a for a in assignments if a.id.startswith("integration-0000")]
    assert len(pages) > 1
    assert all(len(a.payload.encode()) <= 600_000 for a in assignments)
    findings = []
    for page in pages:
        packet = json.loads(page.payload)
        assert packet["scope"]["units"] == [u.model_dump() for u in planned.chunks[0].units]
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
    reversed_reports = dict(reversed(list(reports.items())))
    actual = work.integration_assignments(context, planned, reversed_reports)
    for i, chunk in enumerate(planned.chunks):
        packet = json.loads(next(a.payload for a in actual if a.id == f"integration-{i:04d}"))
        assert [r["summary"] for r in packet["reports"]] == [
            f"{chunk.id}/claude",
            f"{chunk.id}/openai",
        ]
    assert actual == work.integration_assignments(context, planned, reports)


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
        """Model the capacity response independently of byte size; record generated inputs."""
        if preflight:
            if role == rejecting_role:
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
    assert manifest["force_chunks"] is True
    # Replay cannot silently switch back to the byte-only whole assignment.
    manifest["force_chunks"] = False
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
        """Count the expensive snapshot boundary, without bypassing its implementation."""
        anchors = compile_original(context)
        compiled.append(anchors)
        return anchors

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
        """Retain the counters used for each final packet, without replacing their logic."""
        result = original_budget(self, id, extra)
        budgets[id] = result
        return result

    def payload(self: Chunk, metadata: object) -> str:
        """Observe full source serialization at the public packet boundary."""
        serialized.append(self.id)
        return original_payload(self, metadata)

    monkeypatch.setattr(PacketSizer, "budget", budget)
    monkeypatch.setattr(Chunk, "payload", payload)
    assignments = work.integration_assignments(context, planned, reports)
    edges = [a for a in assignments if a.id.startswith("edge-")]
    assert edges
    assert any(json.loads(a.payload)["reference_source"] for a in edges)
    for assignment in edges:
        assert budgets[assignment.id].size == len(assignment.payload.encode())
        assert len(assignment.payload.encode()) <= 600_000
    assert len(serialized) == len(planned.chunks) + len(edges)
