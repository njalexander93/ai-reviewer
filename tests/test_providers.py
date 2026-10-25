"""Verify credential-driven runs and fail-closed replay using offline model boundaries.

Expected selections come from the approved key matrix. Real planning, artifact
replay, reconciliation and publication run against synthetic external responses.
"""

import json
import sys
from pathlib import Path

import pytest

from ai_reviewer import cli, consolidation, work
from ai_reviewer.contracts import FinalConsolidation
from ai_reviewer.github import publish
from tests.test_github import Server, bundle
from tests.test_work import fake_model, large_context


@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize(
    "claude,openai,expected,compiler",
    [
        ("", "key", ["openai"], "sol"),
        ("key", " ", ["claude"], "sonnet"),
        ("key", "key", ["claude", "openai"], "sonnet"),
    ],
)
def test_selected_providers_complete_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    large: bool,
    claude: str,
    openai: str,
    expected: list[str],
    compiler: str,
) -> None:
    """Only configured reviewers run; replay and publication retain their coverage."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", claude)
    monkeypatch.setenv("OPENAI_API_KEY", openai)
    context = large_context() if large else bundle(Server())[0]
    calls = []

    def model(role: str, payload: str, directory: Path, **kwargs: object) -> None:
        """Record paid roles while delegating synthetic responses to the shared fixture."""
        if not kwargs.get("preflight"):
            calls.append(role)
        fake_model(role, payload, directory, **kwargs)

    monkeypatch.setattr(work, "call_model", model)
    monkeypatch.setattr(consolidation, "call_model", model)
    work.review_chunks(context, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["providers"] == expected
    assert set(calls) == set(expected)
    assert len(calls) == len(manifest["assignments"]) * len(expected)
    # Credential changes cannot reinterpret the recorded coverage or compiler choice.
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("OPENAI_API_KEY")
    covered = work.load_reviews(context, tmp_path)
    assert len(covered.reviews) == len(calls)
    # Force summary reduction too, exercising the second consolidation stage.
    monkeypatch.setattr(consolidation, "CONSOLIDATION_BYTES", 1)
    consolidation.consolidate_chunks(context, tmp_path)
    assert set(calls) == {*expected, compiler}
    assert (tmp_path / "consolidation-summaries" / "0" / f"{compiler}.json").exists()
    server = Server()
    server.files = [{"filename": f.filename, "patch": f.patch} for f in context.files]
    result = FinalConsolidation.model_validate_json((tmp_path / "consolidate.json").read_text())
    assert publish(server.gh, context, covered, result)
    assert server.writes[0][1]["event"] == "APPROVE"
    assert f"completed by {', '.join(expected)}" in server.writes[0][1]["body"]


@pytest.mark.parametrize("empty", [None, "", " \n"])
def test_no_credentials_fails_before_provider_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    empty: str | None,
) -> None:
    """Absent or blank credentials produce a clear CLI error without paid work."""
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
        if empty is not None:
            monkeypatch.setenv(key, empty)
    context = bundle(Server())[0]
    (tmp_path / "context.json").write_text(context.model_dump_json())
    calls = []
    monkeypatch.setattr(work, "call_model", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["reviewer", "review", "--directory", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert calls == []
    assert (
        "Configure at least one nonempty ANTHROPIC_API_KEY or OPENAI_API_KEY"
        in capsys.readouterr().out
    )
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("role", ["claude", "openai"])
@pytest.mark.parametrize("during_preflight", [False, True])
def test_configured_failure_never_reduces_required_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    role: str,
    during_preflight: bool,
) -> None:
    """Authentication or exhausted-credit failures never become single-provider approval."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "invalid-or-depleted")
    monkeypatch.setenv("OPENAI_API_KEY", "invalid-or-depleted")
    context = bundle(Server())[0]
    generated = []

    def model(selected: str, payload: str, directory: Path, **kwargs: object) -> None:
        """Fail the selected provider at one external boundary without exposing secrets."""
        if selected == role and bool(kwargs.get("preflight")) == during_preflight:
            raise RuntimeError("synthetic authentication or credit failure")
        if not kwargs.get("preflight"):
            generated.append(selected)
        fake_model(selected, payload, directory, **kwargs)

    monkeypatch.setattr(work, "call_model", model)
    monkeypatch.setattr(consolidation, "call_model", model)
    with pytest.raises((ValueError, RuntimeError)):
        work.review_chunks(context, tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises((ValueError, OSError)):
        consolidation.consolidate_chunks(context, tmp_path)
    assert not (tmp_path / "consolidate.json").exists()
    if during_preflight:
        assert generated == []
    else:
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["providers"] == ["claude", "openai"]
        assert {r["role"]: r["status"] for r in manifest["results"]}[role] == "failed"


@pytest.mark.parametrize("damage", ["report", "selection", "legacy", "empty", "duplicate"])
def test_missing_dual_provider_evidence_cannot_be_reinterpreted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    damage: str,
) -> None:
    """Replay rejects missing artifacts and altered selections even with one current key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr(work, "call_model", fake_model)
    context = bundle(Server())[0]
    work.review_chunks(context, tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    if damage == "report":
        (tmp_path / "work" / "whole" / "openai" / "0" / "openai.json").unlink()
    elif damage == "legacy":
        del manifest["providers"]
    else:
        manifest["providers"] = {
            "selection": ["claude"],
            "empty": [],
            "duplicate": ["claude", "claude"],
        }[damage]
        manifest["results"] = [r for r in manifest["results"] if r["role"] == "claude"]
    path.write_text(json.dumps(manifest))
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises((ValueError, OSError)):
        work.load_reviews(context, tmp_path)


@pytest.mark.parametrize(
    "key,compiler", [("OPENAI_API_KEY", "sol"), ("ANTHROPIC_API_KEY", "sonnet")]
)
def test_consolidation_uses_only_the_validated_provider_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    key: str,
    compiler: str,
) -> None:
    """Removing the manifest after canonical validation cannot change consolidation routing."""
    from ai_reviewer.contracts import CoveredReviews, ReviewContext

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(key, "synthetic-key")
    monkeypatch.setattr(work, "call_model", fake_model)
    monkeypatch.setattr(consolidation, "call_model", fake_model)
    context = bundle(Server())[0]
    work.review_chunks(context, tmp_path)

    def load_once(snapshot: ReviewContext, directory: Path) -> CoveredReviews:
        """Return real validated coverage, then remove its no-longer-needed manifest."""
        covered = work.load_reviews(snapshot, directory)
        (directory / "manifest.json").unlink()
        return covered

    monkeypatch.setattr(consolidation, "load_reviews", load_once)
    consolidation.consolidate_chunks(context, tmp_path)
    assert (tmp_path / "consolidation" / "0" / f"{compiler}.json").is_file()
    result = FinalConsolidation.model_validate_json((tmp_path / "consolidate.json").read_text())
    assert result.groups == []
    assert result.summary == "Assessments agree."
