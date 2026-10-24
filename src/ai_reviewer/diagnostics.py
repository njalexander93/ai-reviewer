"""Record bounded failure metadata without exception messages or response content.

Only schema-owned fields, validation codes and bounded exhaustion coordinates reach logs/artifacts.
Unknown locations are redacted because extra keys can contain untrusted text.
"""

import json
from pathlib import Path
from typing import Any, get_args

from pydantic import ValidationError
from pydantic_core import ErrorType

from ai_reviewer.contracts import ChunkReview, Consolidation, ReviewContext, SourceRequest


class FollowupExhausted(ValueError):
    """Identify incomplete work after the last allowed report.

    Attributes:
        attempts: Number of validated reports, including the final incomplete report.
        requests: Bounded safe coordinates; provider reasons and source are omitted.
        request_count: Full outstanding count, before diagnostic truncation.
    """

    def __init__(self, attempts: int, requests: list[SourceRequest], paths: set[str]) -> None:
        """Retain at most 20 requests, accepting paths only from the captured snapshot.

        Long paths are truncated to 240 characters. Unknown paths are redacted.
        Requests and paths are read without mutation; no source is fetched.
        """
        super().__init__("Required follow-up work exhausted")
        self.attempts = attempts
        self.request_count = len(requests)
        self.requests = [
            {
                "path": request.path[:240] if request.path in paths else "<outside_snapshot>",
                "side": request.side,
                "start": request.start,
                "end": request.end,
            }
            for request in requests[:20]
        ]


def failure_details(error: Exception) -> dict[str, Any]:
    """Return bounded, redacted metadata for error without printing or mutating it.

    Includes its class name, an integer HTTP status in 100..599 when present, and
    at most 20 Pydantic errors with at most 12 location components each. Only known
    schema field names, integer indices and built-in validation codes are retained.
    Exhaustion adds validated-report counts and bounded snapshot request coordinates.
    Exception messages, inputs, contexts, URLs, bodies and headers are omitted.
    """
    # Admit only a class name and numeric HTTP status from generic exceptions.
    details: dict[str, Any] = {"error": type(error).__name__}
    if isinstance(error, FollowupExhausted):
        details.update(
            attempts=error.attempts,
            completed_followups=max(0, error.attempts - 1),
            outstanding_request_count=error.request_count,
            outstanding_requests=error.requests,
        )
    status = getattr(error, "status_code", None)
    if type(status) is int and 100 <= status <= 599:
        details["http_status"] = status
    # Derive safe field names/codes from owned schemas rather than exception text.
    if isinstance(error, ValidationError):
        fields = set()
        for model in (ChunkReview, Consolidation, ReviewContext):
            schema = model.model_json_schema()
            for definition in (schema, *schema.get("$defs", {}).values()):
                fields.update(definition.get("properties", {}))
        codes = set(get_args(ErrorType))
        details["validation_error_count"] = error.error_count()
        details["validation_errors"] = [
            {
                "location": [
                    part if type(part) is int or part in fields else "<redacted>"
                    for part in item["loc"][:12]
                ],
                "code": item["type"] if item["type"] in codes else "custom_error",
            }
            for item in error.errors(include_input=False, include_context=False, include_url=False)[
                :20
            ]
        ]
    return details


def report_failure(error: Exception, path: Path | None = None, **context: str) -> None:
    """Print safe diagnostics and optionally write them without masking write failures.

    Args:
        error: Exception to classify without exposing its message or response content.
        path: Optional artifact path; its parent must already exist.
        **context: Trusted internal string labels only, such as provider or stage.

    Writes a bounded JSON object to stdout and, when requested, path. Artifact OSError
    is handled with a generic warning; the caller still owns propagation of error.
    """
    # Trusted call-site labels supplement redacted provider/schema metadata.
    serialized = json.dumps({**context, **failure_details(error)}, ensure_ascii=True)
    print(f"Review failure: {serialized}", flush=True)
    # Artifact storage is best effort and must not replace the original failure.
    if path is not None:
        try:
            path.write_text(serialized + "\n")
        except OSError:
            print("Review failure metadata could not be saved.", flush=True)
