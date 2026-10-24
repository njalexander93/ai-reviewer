"""Verify advisory-location rejection, artifact retention and publication offline.

Synthetic provider reports exercise the approved rejection policy, including failures
that must preserve diagnostics without allowing approval. No paid APIs are called.
"""

import json
from pathlib import Path

import pytest

from ai_reviewer import consolidation, work
from ai_reviewer.chunking import plan
from ai_reviewer.contracts import ChangedFile, ChunkReview, FinalConsolidation, reconcile
from ai_reviewer.github import publish
from tests.test_github import Server, bundle
from tests.test_work import configured_keys as configured_keys
from tests.test_work import fake_model, large_context


@pytest.mark.parametrize("blocking", [False, True])
def test_nonblocking_invalid_location_is_artifact_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, blocking: bool
) -> None:
    """Reject an advisory anchor without losing valid findings or publishing its prose.

    The approved regression contract permits dropping only invalid nonblockers.
    Synthetic responses exercise both providers, replay, consolidation and publication.
    """
    from ai_reviewer.contracts import Finding

    server = Server()
    context, _, _ = bundle(server)
    context.files = [
        ChangedFile(
            filename="app.py",
            previous_filename=None,
            status="modified",
            patch="@@ -1,2 +1,2 @@\n first\n-old\n+new\n",
            before="first\nold\n",
            after="first\nnew\n",
            before_mode="100644",
            after_mode="100644",
        )
    ]
    accepted = Finding(
        path="app.py",
        side="RIGHT",
        line=2,
        title="Accepted finding",
        body="Retain this original evidence.",
        blocking=blocking,
    )
    rejected = accepted.model_copy(
        update={
            "line": 592,
            "title": "Rejected finding",
            "body": "Rejected evidence",
            "blocking": False,
        }
    )
    raw = ChunkReview(
        complete=True,
        summary="Rejected finding: Rejected evidence",
        findings=[rejected, accepted],
        requests=[],
    ).model_dump_json()
    calls = []

    def model(role: str, payload: str, directory: Path, **kwargs: object) -> None:
        """Record only synthetic generation calls and preserve exact response bytes."""
        if kwargs.get("preflight"):
            return
        calls.append(role)
        (directory / f"{role}.json").write_text(raw)
        (directory / f"{role}.usage.json").write_text('{"input_tokens": 123}')

    monkeypatch.setattr(work, "call_model", model)
    monkeypatch.setattr(consolidation, "call_model", fake_model)
    work.review_chunks(context, tmp_path)
    assert sorted(calls) == ["claude", "openai"]
    for role in ("claude", "openai"):
        target = work.attempt_directory(tmp_path, "whole", role, 0)
        assert (target / f"{role}.json").read_text() == raw
        assert (target / f"{role}.usage.json").read_text() == '{"input_tokens": 123}'
        assert json.loads((target / "rejected-findings.json").read_text()) == [
            {
                "finding_index": 0,
                "finding": rejected.model_dump(),
                "reason": "Finding is outside the captured diff",
            }
        ]
    covered = work.load_reviews(context, tmp_path)
    assert [r.findings for r in covered.reviews] == [[accepted], [accepted]]
    consolidation.consolidate_chunks(context, tmp_path)
    result = FinalConsolidation.model_validate_json((tmp_path / "consolidate.json").read_text())
    assert reconcile(covered.reviews, result) == [[accepted], [accepted]]
    assert publish(server.gh, context, covered, result)
    published = server.writes[0][1]
    assert published["event"] == ("REQUEST_CHANGES" if blocking else "APPROVE")
    assert "Accepted finding" in json.dumps(published)
    assert "Rejected" not in json.dumps(published)
    assert sorted(calls) == ["claude", "openai"]


@pytest.mark.parametrize(
    "damage",
    ["blocking", "schema", "incomplete", "request", "exhausted", "unavailable", "oversized"],
)
@pytest.mark.parametrize("role", ["claude", "openai"])
def test_rejected_advisory_does_not_waive_report_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, damage: str, role: str
) -> None:
    """Invalid blockers, malformed findings and incomplete protocols still fail closed."""
    context = large_context()
    assignment = work.local_assignments(context, plan(context))[0]
    finding = {
        "path": "src/app.py",
        "side": "RIGHT",
        "line": 2001,
        "title": "Outside diff",
        "body": "Must not waive blockers",
        "blocking": False,
    }
    data = {"complete": True, "summary": "Review", "findings": [finding], "requests": []}
    if damage == "blocking":
        data["findings"].append({**finding, "blocking": True})
    elif damage == "schema":
        finding["blocking"] = "false"
    elif damage == "incomplete":
        data["complete"] = False
    else:
        data["requests"] = [
            {
                "path": "src/app.py",
                "side": "RIGHT",
                "start": 1,
                "end": 1,
                "reason": "Needed",
            }
        ]
    if damage in ("exhausted", "unavailable", "oversized"):
        data["complete"] = False
    if damage == "unavailable":
        data["requests"][0]["path"] = "missing.py"
    packet_bytes = 100 if damage == "oversized" else 600_000
    raw = json.dumps(data)

    def model(role: str, payload: str, directory: Path, **kwargs: object) -> None:
        """Write invalid report evidence without invoking a provider."""
        (directory / f"{role}.json").write_text(raw)

    monkeypatch.setattr(work, "call_model", model)
    with pytest.raises(ValueError):
        work.run_assignment(
            context, assignment, role, tmp_path, work.compile_anchors(context), packet_bytes
        )
    target = work.attempt_directory(tmp_path, assignment.id, role, 0)
    assert (target / f"{role}.json").read_text() == raw
    if damage not in ("blocking", "schema"):
        attempts = 3 if damage == "exhausted" else 1
        for attempt in range(attempts):
            target = work.attempt_directory(tmp_path, assignment.id, role, attempt)
            assert json.loads((target / "rejected-findings.json").read_text()) == [
                {
                    "finding_index": 0,
                    "finding": finding,
                    "reason": "Finding is outside the captured diff",
                }
            ]
    # Forged successful completion cannot bypass the same replay validation.
    attempts = 3 if damage == "exhausted" else 1
    manifest = work.Manifest(
        providers=(role,),
        plan_digest="unused by read_batch",
        packet_bytes=packet_bytes,
        assignments=[work.assignment_record(assignment)],
        results=[
            work.Completion(
                assignment=assignment.id,
                role=role,
                status="complete",
                attempts=attempts,
            )
        ],
    )
    with pytest.raises(ValueError):
        work.read_batch(
            context, [assignment], manifest, tmp_path, work.compile_anchors(context), packet_bytes
        )


def test_filtered_followup_replays_without_trusting_rejection_sidecars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Continuation retains accepted evidence and replays from raw provider responses."""
    context = large_context()
    planned = plan(context)
    assignment = work.local_assignments(context, planned)[0]
    accepted = {
        "path": "src/app.py",
        "side": "RIGHT",
        "line": 1,
        "title": "Accepted",
        "body": "Original evidence",
        "blocking": True,
    }
    rejected = {**accepted, "line": 2001, "title": "Rejected", "blocking": False}
    raw = json.dumps(
        {
            "complete": False,
            "summary": "Rejected observation",
            "findings": [accepted, rejected],
            "requests": [
                {
                    "path": "src/app.py",
                    "side": "RIGHT",
                    "start": 1,
                    "end": 1,
                    "reason": "Verify caller",
                }
            ],
        }
    )
    seen = []

    def model(role: str, payload: str, directory: Path, **kwargs: object) -> None:
        """Request real captured source, then complete without repeating findings."""
        seen.append(json.loads(payload))
        response = (
            raw
            if len(seen) == 1
            else json.dumps(
                {
                    "complete": True,
                    "summary": "Caller verified",
                    "findings": [],
                    "requests": [],
                }
            )
        )
        (directory / f"{role}.json").write_text(response)

    monkeypatch.setattr(work, "call_model", model)
    anchors = work.compile_anchors(context)
    assert work.run_assignment(context, assignment, "claude", tmp_path, anchors, 600_000) == 2
    assert seen[1]["prior_attempts"][0]["findings"] == [accepted]
    assert "Rejected observation" not in json.dumps(seen[1])
    # Sidecars are diagnostic only; tampering must not control accepted findings.
    target = work.attempt_directory(tmp_path, assignment.id, "claude", 0)
    (target / "rejected-findings.json").write_text("not valid JSON")
    manifest = work.Manifest(
        providers=("claude",),
        plan_digest=work.plan_digest(planned, ("claude",)),
        packet_bytes=600_000,
        assignments=[work.assignment_record(assignment)],
        results=[
            work.Completion(
                assignment=assignment.id,
                role="claude",
                status="complete",
                attempts=2,
            )
        ],
    )
    reports = work.read_batch(context, [assignment], manifest, tmp_path, anchors, 600_000)
    assert [f.model_dump() for f in reports[(assignment.id, "claude")].findings] == [accepted]
    assert reports[(assignment.id, "claude")].summary == "Caller verified"
    assert len(seen) == 2
