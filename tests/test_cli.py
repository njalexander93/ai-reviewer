"""Exercise the stage artifact handoff through the real CLI and REST adapter."""

import json
import sys
from pathlib import Path

import pytest

from ai_reviewer import cli
from ai_reviewer.contracts import Consolidation, Review
from tests.test_github import Server


def test_pipeline_posts_one_clean_review_and_logs_costs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Independent reports reach publication through files, without prices in feedback."""
    # Bind the CLI to a synthetic repository and isolated artifact directory.
    server = Server()
    monkeypatch.setattr(cli, "GitHub", lambda token: server.gh)
    for key, value in {
        "GH_TOKEN": "test",
        "AI_REVIEWER_TOKEN": "test",
        "GITHUB_REPOSITORY": "owner/repo",
        "PR_NUMBER": "1",
    }.items():
        monkeypatch.setenv(key, value)
    payloads: dict[str, dict] = {}

    def model(role: str, payload: str, directory: Path) -> None:
        """Write deterministic reports at the paid-provider boundary.

        Args:
            role: Reviewer or consolidator selected by the CLI stage.
            payload: JSON context or review pair, retained for assertions.
            directory: Isolated stage directory receiving the synthetic report.
        """
        # Retain the actual stage input before selecting its response contract.
        payloads[role] = json.loads(payload)
        report = (
            Consolidation(summary="Both reviewers found no blockers.", groups=[])
            if role == "consolidate"
            else Review(complete=True, summary=f"{role} completed", findings=[])
        )
        # Match the artifact interface consumed by the next real CLI stage.
        (directory / f"{role}.json").write_text(report.model_dump_json())

    # Run the real stage dispatcher with only external boundaries replaced.
    monkeypatch.setattr(cli, "call_model", model)
    for stage in ["prepare", "claude", "openai", "consolidate", "publish", "costs"]:
        monkeypatch.setattr(sys, "argv", ["review", stage, "--directory", str(tmp_path)])
        cli.main()
    # Verify identical reviewer input, the published verdict and log-only accounting.
    assert payloads["claude"] == payloads["openai"]
    assert payloads["claude"]["head"] == "a" * 40
    assert [r["summary"] for r in payloads["consolidate"]["reviews"]] == [
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
    # Write one complete report and one report declaring insufficient context.
    for role in ("claude", "openai"):
        (tmp_path / f"{role}.json").write_text(
            Review(
                complete=role == "claude", summary="secret-content", findings=[]
            ).model_dump_json()
        )
    monkeypatch.setattr(sys, "argv", ["review", stage, "--directory", str(tmp_path)])
    # Require a failed stage without exposing report text or creating consolidation.
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert "secret-content" not in capsys.readouterr().out
    assert not (tmp_path / "consolidate.json").exists()
