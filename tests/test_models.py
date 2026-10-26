"""Provider contract and accounting tests using synthetic SDK response fixtures."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_reviewer import models
from ai_reviewer.contracts import Consolidation, Review
from ai_reviewer.models import Usage, call_model, print_costs


@pytest.mark.parametrize("role", ["claude", "openai", "sonnet", "sol"])
@pytest.mark.parametrize("complete", [True, False])
def test_model_completion_and_usage_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role: str, complete: bool
) -> None:
    """Only completed parsed responses become reports; partial calls still report usage."""
    # Choose the role-specific report and provider completion state.
    calls: list[dict] = []
    report = (
        Consolidation(summary="Clean", groups=[])
        if role in ("sonnet", "sol")
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

    def absent_client(**kwargs: object) -> None:
        """Reject construction of the provider absent from this request's route."""
        raise AssertionError("Unused provider client constructed")

    if role in ("openai", "sol"):
        monkeypatch.setattr(models.anthropic, "Anthropic", absent_client)
    else:
        monkeypatch.setattr(models.openai, "OpenAI", absent_client)
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
    assert usage["input"] == (900 if role in ("openai", "sol") else 1000)
    assert "tools" not in calls[0]
    if role in ("openai", "sol"):
        assert calls[0]["input"] == [{"role": "user", "content": "untrusted context"}]
        assert calls[0]["reasoning"] == {"effort": "medium"}
        assert calls[0]["truncation"] == "disabled"
        assert calls[0]["store"] is False
        assert calls[0]["service_tier"] == "default"
    else:
        assert calls[0]["thinking"] == {"type": "adaptive"}
        assert calls[0]["output_config"] == {"effort": "medium"}
        assert calls[0]["service_tier"] == "standard_only"

    assert (
        calls[0]["model"]
        == {
            "claude": "claude-opus-5-5",
            "openai": "gpt-6.1-sol",
            "sonnet": "claude-sonnet-5-5",
            "sol": "gpt-6.1-sol",
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
    """100k/20k reviewer tokens plus 30k/5k Sonnet tokens cost $1.3100 at stated rates."""
    # Write fixed token counts with independently calculated expected costs.
    for role, count in [
        ("claude", (100000, 20000)),
        ("openai", (100000, 20000)),
        ("sonnet", (30000, 5000)),
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
    assert "sonnet: not run ($0)" in output
    assert "Estimated API total: $0.0000\n" in output
    assert "unknown" not in output


@pytest.mark.parametrize("role", ["claude", "openai"])
def test_provider_count_blocks_generation_before_capacity_overflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role: str
) -> None:
    """A provider count above 250k blocks generation even for a tiny byte payload."""
    # Expose a provider count over policy and observe whether generation starts.
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
        call_model(role, "tiny", tmp_path, kind="local")
    assert generated == []
    assert not (tmp_path / f"{role}.json").exists()
    assert not (tmp_path / f"{role}.usage.json").exists()


def test_costs_include_every_isolated_attempt(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Three Opus attempts at $0.40 and two Sol attempts at $0.20 total $1.60."""
    # Create nested successful attempts plus unknown usage before aggregating costs.
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


@pytest.mark.parametrize("role", ["claude", "openai"])
def test_preflight_never_generates_or_records_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role: str
) -> None:
    """Counting-only requests must not cross the paid-generation boundary."""
    # Keep counting successful while making generation observable.
    generated = []
    client = SimpleNamespace(
        responses=SimpleNamespace(
            parse=lambda **kwargs: generated.append(kwargs),
            input_tokens=SimpleNamespace(count=lambda **kwargs: SimpleNamespace(input_tokens=1000)),
        ),
        messages=SimpleNamespace(
            parse=lambda **kwargs: generated.append(kwargs),
            count_tokens=lambda **kwargs: SimpleNamespace(input_tokens=1000),
        ),
    )
    monkeypatch.setattr(models.openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(models.anthropic, "Anthropic", lambda **kwargs: client)
    call_model(role, "context", tmp_path, kind="local", preflight=True)
    assert generated == []
    assert list(tmp_path.iterdir()) == []


def test_provider_validation_failure_saves_redacted_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """An SDK parse failure records provider and schema location before propagating."""
    from pydantic import ValidationError

    # Inject a real validation exception carrying private input at the SDK boundary.
    error = ValidationError.from_exception_data(
        "Review", [{"type": "string_type", "loc": ("summary",), "input": {"secret": "content"}}]
    )

    def parse(**kwargs: object) -> None:
        """Raise the captured real ValidationError at the SDK parse boundary.

        All SDK keyword options are accepted but ignored. No provider/network call occurs;
        the exception intentionally carries private input for the diagnostic assertion.
        """
        raise error

    client = SimpleNamespace(
        messages=SimpleNamespace(
            parse=parse, count_tokens=lambda **kwargs: SimpleNamespace(input_tokens=1000)
        )
    )
    monkeypatch.setattr(models.anthropic, "Anthropic", lambda **kwargs: client)
    with pytest.raises(ValidationError):
        call_model("claude", "private-payload", tmp_path)
    data = json.loads((tmp_path / "claude.error.json").read_text())
    assert data["role"] == "claude"
    assert data["model"] == "claude-opus-5-5"
    assert data["phase"] == "generation"
    assert data["validation_errors"] == [{"location": ["summary"], "code": "string_type"}]
    assert (tmp_path / "claude.usage.json").read_text() == "null"
    assert not (tmp_path / "claude.json").exists()
    assert "secret" not in capsys.readouterr().out


@pytest.mark.parametrize("role", ["claude", "openai"])
@pytest.mark.parametrize(
    "kind,framing",
    [
        ("whole", "complete captured PR"),
        ("local", "bounded local assignment"),
        ("integration", "Assess PR-wide compatibility"),
    ],
)
def test_reviewer_prompt_names_scope_and_always_allows_source_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role: str, kind: str, framing: str
) -> None:
    """Both providers receive accurate scope framing and the same request-capable contract."""
    from ai_reviewer.contracts import ChunkReview

    # Capture count requests for each role/scope without purchasing a completion.
    calls = []

    def count(**kwargs: object) -> SimpleNamespace:
        """Capture kwargs and return an SDK-shaped synthetic count of 100 input tokens.

        The calls list retains the actual request options for assertions. No generation,
        network access or artifact writes occur in this replacement count endpoint.
        """
        calls.append(kwargs)
        return SimpleNamespace(input_tokens=100)

    client = SimpleNamespace(
        responses=SimpleNamespace(input_tokens=SimpleNamespace(count=count)),
        messages=SimpleNamespace(count_tokens=count),
    )
    monkeypatch.setattr(models.openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(models.anthropic, "Anthropic", lambda **kwargs: client)
    call_model(role, "unchanged payload", tmp_path, kind=kind, preflight=True)
    call = calls[0]
    if role == "openai":
        assert call["input"] == [{"role": "user", "content": "unchanged payload"}]
    instructions = call["instructions"] if role in ("openai", "sol") else call["system"]
    assert framing in instructions
    assert "Request missing evidence" in instructions
    assert "Markdown files also have sections by LEFT/RIGHT" in instructions
    assert "Requests can read changed files" in " ".join(instructions.split())
    assert "Missing entries never prove evidence is absent" in instructions
    if kind == "whole":
        assert "within a larger PR" not in instructions
    schema = (
        call["text"]["format"]["schema"]
        if role in ("openai", "sol")
        else call["output_format"].model_json_schema()
    )
    assert schema == ChunkReview.model_json_schema()


@pytest.mark.parametrize("role", ["claude", "openai"])
def test_repository_contract_excludes_legacy_requests(role: str) -> None:
    """Repository reviews cannot emit the legacy source-request field in their schema."""
    schema, instructions = models.request_contract(role, "whole", repository_enabled=True)
    assert "Read-only repository tools are available" in instructions
    assert "requests" not in schema.model_json_schema()["properties"]
    assert "Request missing evidence using requests" not in instructions


@pytest.mark.parametrize("role", ["sonnet", "sol"])
def test_consolidator_prices_follow_published_standard_rates(role: str) -> None:
    """Published $2/$10/$0.10/$2.50 rates price fresh/output/read/write tokens.

    Sources: Sonnet 5.5 overview and GPT-6.1 Sol model pages linked in README.
    The example costs $0.20 + $0.01 + $0.0001 + $0.0025 = $0.2126.
    """
    usage = Usage(input=100_000, output=1_000, cached=1_000, cache_write=1_000)
    assert usage.cost(role) == pytest.approx(0.2126)


def test_sol_usage_is_attributed_to_openai(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """30k fresh and 5k output Sol tokens cost $0.1100 under documented standard rates."""
    (tmp_path / "sol.usage.json").write_text(
        Usage(input=30_000, output=5_000, cached=0, cache_write=0).model_dump_json()
    )
    print_costs(tmp_path)
    output = capsys.readouterr().out
    assert "gpt-6.1-sol" in output
    assert "OpenAI subtotal: $0.1100" in output
    assert "Claude subtotal: $0.0000" in output
