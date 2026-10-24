"""Exercise the stage artifact handoff through the real CLI and REST adapter."""

import json
import sys
from pathlib import Path
from threading import Barrier

import pytest

from ai_reviewer import cli, work
from ai_reviewer.contracts import ChunkReview, Consolidation
from tests.test_github import Server, bundle


def test_pipeline_posts_one_clean_review_and_logs_costs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Independent reports reach publication through files, without prices in feedback."""
    # Bind the CLI to a synthetic repository and isolated artifact directory.
    server = Server()
    server.linked_issues = [
        {
            "url": "https://github.com/owner/repo/issues/2",
            "title": "Keep history",
            "body": "Retain prior edits.",
            "state": "OPEN",
        }
    ]
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    monkeypatch.setattr(cli, "GitHub", lambda token: server.gh)
    for key, value in {
        "GH_TOKEN": "test",
        "AI_REVIEWER_TOKEN": "test",
        "GITHUB_REPOSITORY": "owner/repo",
        "PR_NUMBER": "1",
    }.items():
        monkeypatch.setenv(key, value)
    payloads: dict[str, dict] = {}

    def model(
        role: str, payload: str, directory: Path, *, kind: str = "whole", preflight: bool = False
    ) -> None:
        """Write deterministic reports at the paid-provider boundary.

        Args:
            role: Reviewer or consolidator selected by the CLI stage.
            payload: JSON context or review pair, retained for assertions.
            directory: Isolated stage directory receiving the synthetic report.
            kind: Assignment scope framing.
            preflight: Count-only phase; no report is generated.
        """
        if preflight:
            return
        directory.mkdir(parents=True, exist_ok=True)
        # Retain the actual stage input before selecting its response contract.
        payloads[role] = json.loads(payload)
        report = (
            Consolidation(summary="Both reviewers found no blockers.", groups=[])
            if role == "consolidate"
            else ChunkReview(complete=True, summary=f"{role} completed", findings=[], requests=[])
        )
        # Match the artifact interface consumed by the next real CLI stage.
        (directory / f"{role}.json").write_text(report.model_dump_json())

    # Run the real stage dispatcher with only external boundaries replaced.
    monkeypatch.setattr(work, "call_model", model)
    for stage in ["prepare", "review", "consolidate", "publish", "costs"]:
        monkeypatch.setattr(sys, "argv", ["review", stage, "--directory", str(tmp_path)])
        cli.main()
    # Verify identical reviewer input, the published verdict and log-only accounting.
    assert payloads["claude"] == payloads["openai"]
    assert payloads["claude"]["head"] == "a" * 40
    assert payloads["claude"]["issues"][0]["description"] == "Retain prior edits."
    assert payloads["consolidate"]["reviews"][:2] == [
        "claude completed",
        "openai completed",
    ]
    assert server.writes[0][1] == {"reviewers": ["reviewer"]}
    posted = server.writes[1][1]
    assert posted["event"] == "APPROVE"
    assert posted["comments"] == []
    assert "$" not in posted["body"]
    assert "Estimated API total:" in capsys.readouterr().out


@pytest.mark.parametrize("stage", ["consolidate", "publish"])
def test_incomplete_report_blocks_later_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage: str,
    capsys: pytest.CaptureFixture,
) -> None:
    """Insufficient context cannot turn into a clean review in either downstream stage."""
    from tests.test_work import fake_model

    context, _, _ = bundle(Server())
    (tmp_path / "context.json").write_text(context.model_dump_json())
    monkeypatch.setattr(work, "call_model", fake_model)
    work.review_chunks(context, tmp_path)
    target = work.attempt_directory(tmp_path, "whole", "openai", 0) / "openai.json"
    target.write_text(
        ChunkReview(
            complete=False, summary="secret-content", findings=[], requests=[]
        ).model_dump_json()
    )
    monkeypatch.setattr(sys, "argv", ["review", stage, "--directory", str(tmp_path)])
    # Require a failed stage without exposing report text or creating consolidation.
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert "secret-content" not in capsys.readouterr().out
    assert not (tmp_path / "consolidate.json").exists()


@pytest.mark.parametrize("failed_role", [None, "claude", "openai"])
def test_parallel_review_waits_for_both_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failed_role: str | None
) -> None:
    """Both calls start concurrently and finish recording usage even if one fails."""
    # Neither fake call can finish unless the other starts, exposing serial execution.
    started = Barrier(2, timeout=5)
    context, _, _ = bundle(Server())
    (tmp_path / "context.json").write_text(context.model_dump_json())

    def model(
        role: str, payload: str, directory: Path, *, kind: str = "whole", preflight: bool = False
    ) -> None:
        """Synchronize provider calls and record completion before a selected failure.

        Args:
            role: Provider under test; selects its independent usage artifact.
            payload: Context text shared by both concurrent calls.
            directory: Isolated artifact directory.
            kind: Assignment scope framing.
            preflight: Count-only phase; no report is generated.

        Raises:
            RuntimeError: The selected provider fails after recording usage.
        """
        if preflight:
            return
        (directory / f"{role}.json").write_text(
            ChunkReview(complete=True, summary="Done", findings=[], requests=[]).model_dump_json()
        )
        # Require overlap rather than inferring parallelism from execution duration.
        assert json.loads(payload)["head"] == "a" * 40
        started.wait()
        # Keep the usage evidence observable even when this provider reports failure.
        (directory / f"{role}.usage.json").write_text("recorded")
        if role == failed_role:
            raise RuntimeError("synthetic provider failure")

    # Exercise the real CLI dispatcher with the paid boundary replaced.
    monkeypatch.setattr(work, "call_model", model)
    monkeypatch.setattr(sys, "argv", ["reviewer", "review", "--directory", str(tmp_path)])
    if failed_role:
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 1
    else:
        cli.main()
    # Returning from the stage must mean both calls have finished, including failure paths.
    for role in ("claude", "openai"):
        assert (
            work.attempt_directory(tmp_path, "whole", role, 0) / f"{role}.usage.json"
        ).read_text() == "recorded"


def test_invalid_anchor_stops_before_consolidator_purchase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A valid report contract with an invalid diff line never reaches the paid API."""
    # Capture a real fixture packet, then move its finding beyond the available hunk.
    server = Server()
    context = cli.prepare(server.gh, "owner/repo", 1, "reviewer")
    _, reviews, _ = bundle(server)
    reviews[0].findings[0].line = 999
    (tmp_path / "context.json").write_text(context.model_dump_json())
    from tests.test_work import fake_model

    monkeypatch.setattr(work, "call_model", fake_model)
    work.review_chunks(context, tmp_path)
    target = work.attempt_directory(tmp_path, "whole", "claude", 0) / "claude.json"
    target.write_text(ChunkReview(**reviews[0].model_dump(), requests=[]).model_dump_json())
    calls = []
    monkeypatch.setattr(work, "call_model", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["review", "consolidate", "--directory", str(tmp_path)])
    # Reject rather than discard the finding or pay for a useless consolidation.
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert calls == []
    assert target.exists()


def test_chunked_cli_handoff_uses_manifest_and_large_consolidation_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The deployed stage commands validate chunk coverage before the single review write."""
    from ai_reviewer import work
    from tests.test_work import fake_model, large_context

    # Bind synthetic providers and GitHub to real stage/artifact handling.
    context = large_context()
    (tmp_path / "context.json").write_text(context.model_dump_json())
    server = Server()
    server.files = [{"filename": f.filename, "patch": f.patch} for f in context.files]
    monkeypatch.setattr(work, "call_model", fake_model)
    monkeypatch.setattr(cli, "GitHub", lambda token: server.gh)
    monkeypatch.setenv("AI_REVIEWER_TOKEN", "fixture-token")
    # Run all downstream commands, then require exactly one completed publication.
    for stage in ("review", "consolidate", "publish"):
        monkeypatch.setattr(sys, "argv", ["review", stage, "--directory", str(tmp_path)])
        cli.main()
    assert len(server.writes) == 1
    assert server.writes[0][1]["event"] == "APPROVE"
    assert (tmp_path / "manifest.json").exists()
