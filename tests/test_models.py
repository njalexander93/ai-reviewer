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
    # Choose the role-specific report and provider completion state.
    calls: list[dict] = []
    report = (
        Consolidation(summary="Clean", groups=[])
        if role == "consolidate"
        else Review(complete=True, summary="Clean", findings=[])
    )

    def parse(**kwargs: object) -> SimpleNamespace:
        """Return a synthetic provider response and record the outgoing request.

        Args:
            **kwargs: Parameters sent by the production provider adapter.

        Returns:
            SDK-shaped response with the chosen completion status and fixed usage.
        """
        # Capture the request independently of the response fixture.
        calls.append(kwargs)
        # Expose both SDK response shapes through one deterministic fixture.
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

    # Replace client construction so no API key or network request is needed.
    client = SimpleNamespace(
        responses=SimpleNamespace(
            parse=parse,
            input_tokens=SimpleNamespace(count=lambda **kwargs: SimpleNamespace(input_tokens=1000)),
        ),
        messages=SimpleNamespace(
            parse=parse, count_tokens=lambda **kwargs: SimpleNamespace(input_tokens=1000)
        ),
    )
    monkeypatch.setattr(models.openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(models.anthropic, "Anthropic", lambda **kwargs: client)
    # Completed responses produce reports; truncated responses retain only usage.
    if complete:
        call_model(role, "untrusted context", tmp_path)
        assert json.loads((tmp_path / f"{role}.json").read_text())["summary"] == "Clean"
    else:
        with pytest.raises(ValueError, match="incomplete"):
            call_model(role, "untrusted context", tmp_path)
        assert not (tmp_path / f"{role}.json").exists()
    # Check provider-specific normalization and the requested model identity.
    usage = json.loads((tmp_path / f"{role}.usage.json").read_text())
    # Exercise completion or truncation, then inspect saved output and usage.
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

    # Fail generation after a successful token-count preflight.
    def fail(**kwargs: object) -> None:
        """Reject generation with a deterministic transport failure.

        Args:
            **kwargs: Ignored SDK constructor options.

        Raises:
            TimeoutError: Always, simulating an interrupted provider request.
        """
        raise TimeoutError("request interrupted")

    client = SimpleNamespace(
        responses=SimpleNamespace(
            parse=fail,
            input_tokens=SimpleNamespace(count=lambda **kwargs: SimpleNamespace(input_tokens=1000)),
        )
    )
    monkeypatch.setattr(models.openai, "OpenAI", lambda **kwargs: client)
    # Require failed calls to leave an explicitly unknown usage record.
    with pytest.raises(TimeoutError):
        call_model("openai", "context", tmp_path)
    assert (tmp_path / "openai.usage.json").read_text() == "null"


def test_cost_example_matches_hand_calculation(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """100k/20k reviewer tokens plus 30k/5k Sonnet tokens cost $1.31 at stated rates."""
    # Write fixed token counts with independently calculated expected costs.
    for role, count in [
        ("claude", (100000, 20000)),
        ("openai", (100000, 20000)),
        ("consolidate", (30000, 5000)),
    ]:
        (tmp_path / f"{role}.usage.json").write_text(
            Usage(input=count[0], output=count[1], cached=0, cache_write=0).model_dump_json()
        )
    # Exercise the real formatter over the synthetic usage artifacts.
    print_costs(tmp_path)
    output = capsys.readouterr().out
    # Compare logged subtotals and the total against the hand calculation.
    assert "Claude subtotal: $0.9100" in output
    assert "OpenAI subtotal: $0.4000" in output
    assert "Estimated API total: $1.3100" in output


def test_unknown_usage_is_not_zero(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Attempted calls with unavailable usage leave totals explicitly incomplete."""
    # Mark attempted calls whose usage could not be retrieved.
    for role in ("claude", "openai"):
        (tmp_path / f"{role}.usage.json").write_text("null")
    print_costs(tmp_path)
    output = capsys.readouterr().out
    # Require both provider subtotals to disclose unavailable usage.
    assert "Claude subtotal: $0.0000 + unknown" in output
    assert "OpenAI subtotal: $0.0000 + unknown" in output


def test_cache_counts_use_separate_prices() -> None:
    """1M fresh, 1M cached and 1M writes cost $9.20 for Opus, without output."""
    assert Usage(input=1000000, output=0, cached=1000000, cache_write=1000000).cost("claude") == 9.2


def test_unattempted_calls_cost_zero(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A stage never attempted in the job does not make its cost unknown."""
    print_costs(tmp_path)
    output = capsys.readouterr().out
    assert "consolidate: not run ($0)" in output
    assert "Estimated API total: $0.0000\n" in output
    assert "unknown" not in output


@pytest.mark.parametrize("role", ["claude", "openai"])
def test_provider_count_blocks_generation_before_capacity_overflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role: str
) -> None:
    """A provider count above 250k blocks generation even for a tiny byte payload."""
    generated = []
    client = SimpleNamespace(
        responses=SimpleNamespace(
            parse=lambda **kwargs: generated.append(kwargs),
            input_tokens=SimpleNamespace(
                count=lambda **kwargs: SimpleNamespace(input_tokens=250_001)
            ),
        ),
        messages=SimpleNamespace(
            parse=lambda **kwargs: generated.append(kwargs),
            count_tokens=lambda **kwargs: SimpleNamespace(input_tokens=250_001),
        ),
    )
    monkeypatch.setattr(models.openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(models.anthropic, "Anthropic", lambda **kwargs: client)
    with pytest.raises(ValueError, match="capacity"):
        call_model(role, "tiny", tmp_path, chunked=True)
    assert generated == []
    assert not (tmp_path / f"{role}.json").exists()
    assert not (tmp_path / f"{role}.usage.json").exists()


def test_costs_include_every_isolated_attempt(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Three Opus attempts at $0.40 and two Sol attempts at $0.20 total $1.60."""
    for role, count in (("claude", 3), ("openai", 2)):
        for index in range(count):
            target = tmp_path / "work" / str(index) / role
            target.mkdir(parents=True)
            (target / f"{role}.usage.json").write_text(
                Usage(input=100_000, output=0, cached=0, cache_write=0).model_dump_json()
            )
    failed = tmp_path / "work" / "failed"
    failed.mkdir()
    (failed / "claude.usage.json").write_text("null")
    print_costs(tmp_path)
    output = capsys.readouterr().out
    assert "Claude subtotal: $1.2000 + unknown" in output
    assert "OpenAI subtotal: $0.4000" in output
    assert "Estimated API total: $1.6000 + unknown" in output
