"""Check real Pydantic failures for useful metadata without untrusted error text."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_reviewer.contracts import Review
from ai_reviewer.diagnostics import failure_details, report_failure


def test_validation_diagnostics_redact_inputs_messages_and_unknown_keys(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Keep known field/index/code metadata while removing secrets from every error channel."""
    # Put secrets in every untrusted validation channel before recording diagnostics.
    error = ValidationError.from_exception_data(
        "secret-model-title",
        [
            {"type": "int_type", "loc": ("findings", 0, "line"), "input": "secret-input"},
            {
                "type": "value_error",
                "loc": ("secret-key",),
                "input": "secret-input",
                "ctx": {"error": ValueError("secret-message")},
            },
        ],
    )
    # Exercise artifact and stdout paths together, then compare the safe projection.
    target = tmp_path / "claude.error.json"
    report_failure(error, target, role="claude", phase="generation")
    data = json.loads(target.read_text())
    assert data == {
        "role": "claude",
        "phase": "generation",
        "error": "ValidationError",
        "validation_error_count": 2,
        "validation_errors": [
            {"location": ["findings", 0, "line"], "code": "int_type"},
            {"location": ["<redacted>"], "code": "value_error"},
        ],
    }
    assert "secret" not in target.read_text() + capsys.readouterr().out


def test_validation_diagnostics_bound_error_details() -> None:
    """A large malformed response retains its error count but only 20 safe entries."""
    # Generate more independent invalid fields than the diagnostic detail allowance.
    with pytest.raises(ValidationError) as caught:
        Review.model_validate(
            {
                "complete": True,
                "summary": "Review",
                "findings": [],
                **{f"private-key-{i}": "private-input" for i in range(25)},
            }
        )
    # Keep the total count while bounding and redacting the emitted detail list.
    details = failure_details(caught.value)
    assert details["validation_error_count"] == 25
    assert len(details["validation_errors"]) == 20
    assert "private" not in json.dumps(details)


def test_transport_diagnostics_keep_status_without_exception_body(
    capsys: pytest.CaptureFixture,
) -> None:
    """HTTP metadata is useful; an arbitrary provider error body must never be printed."""
    import httpx
    import openai

    # Construct an SDK error with useful status and deliberately private content.
    error = openai.RateLimitError(
        "secret-message",
        response=httpx.Response(429, request=httpx.Request("POST", "https://example.invalid")),
        body={"message": "secret-body"},
    )
    # Require status/class metadata while excluding the response and message.
    report_failure(error, role="openai", phase="preflight")
    output = capsys.readouterr().out
    assert '"http_status": 429' in output
    assert '"error": "RateLimitError"' in output
    assert "secret" not in output


def test_exhaustion_diagnostics_bound_and_redact_provider_requests() -> None:
    """Unknown paths and provider prose cannot enter the bounded coordinate report."""
    from ai_reviewer.contracts import SourceRequest
    from ai_reviewer.diagnostics import FollowupExhausted

    requests = [
        SourceRequest(path="secret-path", side="RIGHT", start=1, end=2, reason="secret-reason")
        for _ in range(25)
    ]
    details = failure_details(FollowupExhausted(3, requests, {"app.py"}))
    assert details["outstanding_request_count"] == 25
    assert len(details["outstanding_requests"]) == 20
    assert details["outstanding_requests"][0] == {
        "path": "<outside_snapshot>",
        "side": "RIGHT",
        "start": 1,
        "end": 2,
    }
    assert "secret" not in json.dumps(details)
