"""Provider contract and accounting tests using synthetic SDK response fixtures."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_reviewer import models
from ai_reviewer.contracts import Consolidation, Review
from ai_reviewer.models import Usage, call_model, print_costs


@pytest.mark.parametrize("role", ["claude", "openai", "consolidate"])
@pytest.mark.parametrize("complete", [True, False])
def test_model_completion_and_usage_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role: str, complete: bool
) -> None:
    """Only completed parsed responses become reports; partial calls still report usage."""
    calls: list[dict] = []
    report = (
        Consolidation(summary="Clean", groups=[])
        if role == "consolidate"
        else Review(complete=True, summary="Clean", findings=[])
    )

    def parse(**kwargs: object) -> SimpleNamespace:
        """Simulate documented SDK response fields, retaining request parameters."""
        calls.append(kwargs)
        return SimpleNamespace(
            status="completed" if complete else "incomplete",
            output_parsed=report,
            parsed_output=report,
            stop_reason="end_turn" if complete else "max_tokens",
            usage=SimpleNamespace(
                input_tokens=1000,
                output_tokens=200,
                input_tokens_details=SimpleNamespace(cached_tokens=100),
                cache_read_input_tokens=100,
                cache_creation_input_tokens=0,
            ),
        )

    client = SimpleNamespace(
        responses=SimpleNamespace(parse=parse), messages=SimpleNamespace(parse=parse)
    )
    monkeypatch.setattr(models.openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(models.anthropic, "Anthropic", lambda **kwargs: client)
    if complete:
        call_model(role, "untrusted context", tmp_path)
        assert json.loads((tmp_path / f"{role}.json").read_text())["summary"] == "Clean"
    else:
        with pytest.raises(ValueError, match="incomplete"):
            call_model(role, "untrusted context", tmp_path)
        assert not (tmp_path / f"{role}.json").exists()
    usage = json.loads((tmp_path / f"{role}.usage.json").read_text())
    assert usage["output"] == 200
    assert usage["input"] == (900 if role == "openai" else 1000)
    assert "tools" not in calls[0]
    assert (
        calls[0]["model"]
        == {
            "claude": "claude-opus-5-5",
            "openai": "gpt-6.1-sol",
            "consolidate": "claude-sonnet-5-5",
        }[role]
    )


def test_failed_call_records_unknown_cost(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A failed network call must never be reported as free."""

    def fail(**kwargs: object) -> None:
        """Raise a synthetic transport error at the provider boundary."""
        raise TimeoutError("request interrupted")

    monkeypatch.setattr(models.openai, "OpenAI", fail)
    with pytest.raises(TimeoutError):
        call_model("openai", "context", tmp_path)
    assert (tmp_path / "openai.usage.json").read_text() == "null"


def test_cost_example_matches_hand_calculation(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """100k/20k reviewer tokens plus 30k/5k Sonnet tokens cost $1.31 at stated rates."""
    for role, count in [
        ("claude", (100000, 20000)),
        ("openai", (100000, 20000)),
        ("consolidate", (30000, 5000)),
    ]:
        (tmp_path / f"{role}.usage.json").write_text(
            Usage(input=count[0], output=count[1], cached=0, cache_write=0).model_dump_json()
        )
    print_costs(tmp_path)
    output = capsys.readouterr().out
    assert "Claude subtotal: $0.9100" in output
    assert "OpenAI subtotal: $0.4000" in output
    assert "Estimated API total: $1.3100" in output


def test_unknown_usage_is_not_zero(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Absent artifacts leave both provider totals explicitly incomplete."""
    print_costs(tmp_path)
    output = capsys.readouterr().out
    assert "Claude subtotal: $0.0000 + unknown" in output
    assert "OpenAI subtotal: $0.0000 + unknown" in output


def test_cache_counts_use_separate_prices() -> None:
    """1M fresh, 1M cached and 1M writes cost $9.20 for Opus, without output."""
    assert Usage(input=1000000, output=0, cached=1000000, cache_write=1000000).cost("claude") == 9.2
