"""Record bounded failure metadata without exception messages or response content.

Only schema-owned field names and built-in validation codes reach logs/artifacts.
Unknown locations are redacted because extra keys can contain untrusted text.
"""

import json
from pathlib import Path
from typing import Any, get_args

from pydantic import ValidationError
from pydantic_core import ErrorType

from ai_reviewer.contracts import ChunkReview, Consolidation, ReviewContext


def failure_details(error: Exception) -> dict[str, Any]:
    """Return bounded, redacted metadata for error without printing or mutating it.

    Includes its class name, an integer HTTP status in 100..599 when present, and
    at most 20 Pydantic errors with at most 12 location components each. Only known
    schema field names, integer indices and built-in validation codes are retained.
    Exception messages, inputs, contexts, URLs, bodies and headers are omitted.
    """
    # Admit only a class name and numeric HTTP status from generic exceptions.
    details: dict[str, Any] = {"error": type(error).__name__}
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
