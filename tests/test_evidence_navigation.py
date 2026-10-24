"""Check bounded source discovery, follow-up diagnostics and exact artifact replay.

Synthetic provider responses drive the real packet/manifest path without paid calls.
The large snapshot fixture is shared with the orchestration suite.
"""

import json
from pathlib import Path

import pytest

from ai_reviewer import work
from ai_reviewer.chunking import plan
from ai_reviewer.contracts import ChangedFile, ChunkReview
from tests.test_github import Server, bundle
from tests.test_work import large_context


@pytest.mark.parametrize(
    "state", ["complete_with_request", "incomplete_without_request", "exhausted"]
)
def test_attempt_transition_rejects_invalid_completion(state: str) -> None:
    """Completion requires no pending requests; incomplete work needs bounded follow-up."""
    from ai_reviewer.contracts import SourceRequest

    context, _, _ = bundle(Server())
    assignment = work.local_assignments(context, plan(context))[0]
    report = ChunkReview(
        complete=state == "complete_with_request",
        summary="Need evidence",
        findings=[],
        requests=[]
        if state == "incomplete_without_request"
        else [SourceRequest(path="app.py", side="RIGHT", start=1, end=1, reason="Check caller")],
    )
    # Supply invalid states directly so rejection does not depend on a model response.
    history = [report, report] if state == "exhausted" else []
    with pytest.raises(ValueError):
        work.advance(
            context,
            assignment,
            history,
            report,
            600_000,
        )


def test_integration_discovers_handler_before_generation() -> None:
    """Expose the reported line-434 handler and resolve it from the exact snapshot."""
    from ai_reviewer.contracts import SourceRequest
    from ai_reviewer.github import rebuild_patch
    from ai_reviewer.limits import CAPACITY_PACKET_BYTES

    # Small synthetic analogue of Nplet #20: handle sits deep in captured source.
    context, _, _ = bundle(Server())
    before = "# padding\n" * 433 + "def handle(event):\n    return event\n" + "# tail\n" * 420
    after = before.replace(
        "    return event\n", '    event.pop("repository", None)\n    return event\n'
    )
    context.files = [
        ChangedFile(
            filename="src/nplet/adapters/codex.py",
            previous_filename="src/old_codex.py",
            status="renamed",
            patch=rebuild_patch(before, after),
            before=before,
            after=after,
            before_mode="100644",
            after_mode="100644",
        )
    ]
    planned = plan(context, packet_bytes=CAPACITY_PACKET_BYTES)
    local = {
        (a.id, role): ChunkReview(complete=True, summary="Checked", findings=[], requests=[])
        for a in work.local_assignments(context, planned)
        for role in ("claude", "openai")
    }
    assignment = work.integration_assignments(context, planned, local, ("claude", "openai"))[0]
    inventory = json.loads(assignment.payload)["overview"]["inventory"][0]
    assert inventory["filename"] == "src/nplet/adapters/codex.py"
    assert inventory["declarations"]["RIGHT"] == {
        "status": "indexed",
        "entries": [{"name": "handle", "start": 434, "end": 436}],
    }
    assert inventory["declarations"]["LEFT"] == {
        "status": "indexed",
        "entries": [{"name": "handle", "start": 434, "end": 435}],
    }
    # Consume the advertised coordinates as a reviewer would, including renamed LEFT.
    requests = [
        SourceRequest(
            path=inventory["filename"],
            side=side,
            start=inventory["declarations"][side]["entries"][0]["start"],
            end=inventory["declarations"][side]["entries"][0]["end"],
            reason="Inspect handle",
        )
        for side in ("LEFT", "RIGHT")
    ]
    sections = work.source_sections(context, requests)
    assert sections[0]["text"] == "def handle(event):\n    return event\n"
    assert (
        sections[1]["text"]
        == 'def handle(event):\n    event.pop("repository", None)\n    return event\n'
    )


@pytest.mark.parametrize("empty_final", [False, True])
def test_exhaustion_records_attempts_and_safe_outstanding_requests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    empty_final: bool,
) -> None:
    """Three incomplete reports remain failed with actionable, source-free diagnostics."""
    context, _, _ = bundle(Server())
    context.files = [
        ChangedFile(
            filename="app.py",
            previous_filename=None,
            status="added",
            patch="@@ -0,0 +1 @@\n+pass\n",
            before="",
            after="pass\n",
            before_mode=None,
            after_mode="100644",
        )
    ]
    planned = plan(context)
    assignments = work.local_assignments(context, planned)
    manifest = work.Manifest(
        providers=("claude", "openai"),
        plan_digest="test",
        packet_bytes=planned.packet_bytes,
        assignments=[],
        results=[],
    )

    def incomplete(role: str, payload: str, directory: Path, **kwargs: object) -> None:
        """Write contract-valid synthetic requests; the coordinator must reject exhaustion."""
        report = ChunkReview(
            complete=False,
            summary="secret-summary",
            findings=[],
            requests=[
                {
                    "path": context.files[0].filename,
                    "side": "RIGHT",
                    "start": 1,
                    "end": 1,
                    "reason": "secret-reason",
                }
            ],
        )
        if empty_final and directory.name == "2":
            report.requests = []
        (directory / f"{role}.json").write_text(report.model_dump_json())

    monkeypatch.setattr(work, "call_model", incomplete)
    with pytest.raises(ValueError, match="Required provider work failed"):
        work.run_batch(
            context,
            manifest,
            assignments,
            tmp_path,
            work.compile_anchors(context),
            planned.packet_bytes,
        )
    saved = work.Manifest.model_validate_json((tmp_path / "manifest.json").read_text())
    assert [(r.status, r.attempts) for r in saved.results] == [("failed", 3), ("failed", 3)]
    details = json.loads((tmp_path / "work" / "whole" / "claude" / "failure.json").read_text())
    assert details == {
        "assignment": "whole",
        "role": "claude",
        "phase": "worker",
        "error": "FollowupExhausted",
        "attempts": 3,
        "completed_followups": 2,
        "category": "unresolved_evidence",
        "reason": "followup_budget_exhausted",
        "outstanding_request_count": 0 if empty_final else 1,
        "outstanding_requests": []
        if empty_final
        else [{"path": context.files[0].filename, "side": "RIGHT", "start": 1, "end": 1}],
    }
    assert "secret" not in json.dumps(details) + capsys.readouterr().out
    with pytest.raises(ValueError, match="failed worker result"):
        work.read_batch(
            context,
            assignments,
            saved,
            tmp_path,
            work.compile_anchors(context),
            planned.packet_bytes,
        )


@pytest.mark.parametrize("phase", ["integration", "followup"])
def test_navigation_cannot_displace_required_packet_evidence(phase: str) -> None:
    """An advisory index must yield when otherwise fitting evidence needs the space."""
    from ai_reviewer.serialization import encode

    context, _, _ = bundle(Server())
    context.files = [
        ChangedFile(
            filename="app.py",
            previous_filename=None,
            status="added",
            patch="@@ -0,0 +1 @@\n+pass\n",
            before="",
            after="pass\n",
            before_mode=None,
            after_mode="100644",
        )
    ]
    required = {
        "scope": {"units": [{"after": "source" * 1000}]},
        "reports": [{"summary": "report" * 100}],
        "overview": {
            "inventory": [{"filename": "app.py", "after_lines": 1}],
            "declaration_navigation": "complete",
        },
    }
    packet = json.loads(encode(required))
    packet["overview"]["inventory"][0]["declarations"] = {
        "RIGHT": {"status": "indexed", "entries": [{"name": "x" * 12000, "start": 1, "end": 1}]}
    }
    assignment = work.Assignment(
        id="integration-0000", kind="integration", payload=encode(packet), sections=["app.py"]
    )
    original = assignment.payload
    if phase == "integration":
        pages = work.integration_pages(assignment, 10000)
        assert len(pages) == 1
        output = pages[0].payload
    else:
        report = ChunkReview(
            complete=False,
            summary="Need source",
            findings=[],
            requests=[
                {"path": "app.py", "side": "RIGHT", "start": 1, "end": 1, "reason": "Inspect"}
            ],
        )
        output = work.followup_payload(context, assignment, [report], 10000)
    result = json.loads(output)
    assert len(output.encode()) <= 10000
    assert result["scope"] == required["scope"]
    assert result["reports"] == required["reports"]
    assert result["overview"] == {
        "inventory": [{"filename": "app.py", "after_lines": 1}],
        "declaration_navigation": "truncated",
    }
    assert assignment.payload == original
    if phase == "followup":
        assert result["requested_source"][0]["text"] == "pass\n"
        assert result["prior_attempts"][0]["summary"] == "Need source"
    # Required source alone remains a hard failure, even after navigation is dropped.
    with pytest.raises(ValueError, match="exceeds packet budget"):
        if phase == "integration":
            work.integration_pages(assignment, 100)
        else:
            work.followup_payload(context, assignment, [report], 100)


@pytest.mark.parametrize("prior_attempts", [0, 1])
def test_early_incomplete_without_requests_is_protocol_failure(prior_attempts: int) -> None:
    """An early missing request is malformed, not evidence that the budget was spent."""
    from ai_reviewer.diagnostics import IncompleteWithoutRequests, failure_details

    context, _, _ = bundle(Server())
    assignment = work.local_assignments(context, plan(context))[0]
    report = ChunkReview(complete=False, summary="private text", findings=[], requests=[])
    with pytest.raises(IncompleteWithoutRequests) as caught:
        work.advance(
            context,
            assignment,
            [report] * prior_attempts,
            report,
            100_000,
        )
    assert failure_details(caught.value) == {
        "error": "IncompleteWithoutRequests",
        "category": "unresolved_evidence",
        "reason": "missing_source_requests",
        "attempts": prior_attempts + 1,
    }


def test_markdown_section_request_delivers_contract_and_replays(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An offline reviewer can locate line 549 and replay its bounded source request."""
    context = large_context()
    source = "\n" * 535 + "### Task-free parent operations\n" + "\n" * 12
    source += "External file edits contract\n\n### Next section\n"
    context.files.append(
        ChangedFile(
            filename="docs/host-hooks.md",
            previous_filename=None,
            status="added",
            patch="@@ -0,0 +1,551 @@\n" + "".join("+" + line for line in source.splitlines(True)),
            before="",
            after=source,
            before_mode=None,
            after_mode="100644",
        )
    )
    planned = plan(context)
    assignment = work.local_assignments(context, planned)[0]
    manifest = work.Manifest(
        providers=("claude", "openai"),
        plan_digest="test",
        packet_bytes=planned.packet_bytes,
        assignments=[],
        results=[],
    )

    def request_section(role: str, payload: str, directory: Path, **kwargs: object) -> None:
        """Use real packet navigation with schema-valid synthetic provider responses."""
        packet = json.loads(payload)
        if directory.name == "0":
            item = next(
                item
                for item in packet["overview"]["inventory"]
                if item["filename"] == "docs/host-hooks.md"
            )
            section = next(
                entry
                for entry in item["sections"]["RIGHT"]["entries"]
                if entry["name"] == "Task-free parent operations"
            )
            report = ChunkReview(
                complete=False,
                summary="Need contract",
                findings=[],
                requests=[
                    {
                        "path": item["filename"],
                        "side": "RIGHT",
                        "start": section["start"],
                        "end": section["end"],
                        "reason": "Read the external edit contract",
                    }
                ],
            )
        else:
            assert packet["requested_source"] == [
                {
                    "path": "docs/host-hooks.md",
                    "side": "RIGHT",
                    "start": 536,
                    "end": 550,
                    "reason": "Read the external edit contract",
                    "text": "### Task-free parent operations\n"
                    + "\n" * 12
                    + "External file edits contract\n\n",
                }
            ]
            report = ChunkReview(complete=True, summary="Contract read", findings=[], requests=[])
        (directory / f"{role}.json").write_text(report.model_dump_json())

    monkeypatch.setattr(work, "call_model", request_section)
    work.run_batch(
        context,
        manifest,
        [assignment],
        tmp_path,
        work.compile_anchors(context),
        planned.packet_bytes,
    )
    assert [(r.status, r.attempts) for r in manifest.results] == [("complete", 2)] * 2
    replayed = work.read_batch(
        context,
        [assignment],
        manifest,
        tmp_path,
        work.compile_anchors(context),
        planned.packet_bytes,
    )
    assert len(replayed) == 2
