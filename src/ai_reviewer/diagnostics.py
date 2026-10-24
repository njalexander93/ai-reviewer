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
    """Extract error class, numeric HTTP status and at most 20 redacted schema errors."""
    details: dict[str, Any] = {"error": type(error).__name__}
    status = getattr(error, "status_code", None)
    if type(status) is int and 100 <= status <= 599:
        details["http_status"] = status
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
    """Log safe metadata and optionally save it; context must be trusted internal labels.

    Artifact-write failures must not replace the original exception. Exception
    messages, validation input/context, response bodies and headers are omitted.
    """
    serialized = json.dumps({**context, **failure_details(error)}, ensure_ascii=True)
    print(f"Review failure: {serialized}", flush=True)
    if path is not None:
        try:
            path.write_text(serialized + "\n")
        except OSError:
            print("Review failure metadata could not be saved.", flush=True)
