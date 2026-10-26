"""Check OpenAI request history through the real SDK and an offline HTTP transport."""

import json
from pathlib import Path

import httpx
import openai
import pytest

from ai_reviewer import models
from ai_reviewer.contracts import ChunkReview
from tests.test_repository import make_repository


def test_openai_tool_history_survives_multiple_sdk_round_trips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep the prompt, reasoning and matching results through two repository calls.

    Synthetic responses follow OpenAI's function-calling guide and installed SDK
    ResponseReasoningItem/ResponseFunctionToolCall contracts. No API calls occur.
    https://developers.openai.com/api/docs/guides/function-calling
    """
    _, repository = make_repository(tmp_path)
    payload = '{"scope":"review intact Unicode: café\\nsecond line"}'
    requests: list[tuple[str, dict]] = []
    report = {"complete": True, "summary": "Checked both declarations.", "findings": []}
    tool_outputs = [
        [
            {
                "type": "reasoning",
                "id": f"rs_{turn}",
                "summary": [],
                "encrypted_content": f"opaque-reasoning-{turn}",
                "status": "completed",
            },
            {
                "type": "message",
                "id": f"msg_{turn}",
                "role": "assistant",
                "phase": "commentary",
                "status": "completed",
                "content": [{"type": "output_text", "text": json.dumps(report), "annotations": []}],
            },
            {
                "type": "function_call",
                "id": f"fc_{turn}",
                "call_id": f"call-{turn}",
                "name": "search_repository",
                "arguments": json.dumps(
                    {
                        "revision": "head",
                        "path_prefix": "",
                        "cursor": "0:0",
                        "limit": 10,
                        "query": query,
                    }
                ),
                "status": "completed",
            },
        ]
        for turn, query in enumerate(["edit_roles", "DEFAULT_ROLE"], 1)
    ]
    final_output = [
        {
            "type": "message",
            "id": "msg_final",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": json.dumps(report), "annotations": []}],
        }
    ]

    def respond(request: httpx.Request) -> httpx.Response:
        """Serve count and generation responses while recording serialized requests."""
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == "/v1/responses/input_tokens":
            return httpx.Response(
                200, json={"object": "response.input_tokens", "input_tokens": 100}
            )
        assert request.url.path == "/v1/responses"
        turn = sum(path == "/v1/responses" for path, _ in requests) - 1
        return httpx.Response(
            200,
            json={
                "id": f"resp_{turn}",
                "object": "response",
                "created_at": 0,
                "model": "gpt-6.1-sol",
                "status": "completed",
                "output": [*tool_outputs, final_output][turn],
                "usage": {
                    "input_tokens": 100,
                    "input_tokens_details": {"cached_tokens": 20},
                    "output_tokens": 10,
                    "output_tokens_details": {"reasoning_tokens": 5},
                    "total_tokens": 110,
                },
            },
        )

    # Exercise SDK serialization/parsing and real repository evidence, not adapter stubs.
    with openai.OpenAI(
        api_key="offline-test",
        base_url="https://offline.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        max_retries=0,
    ) as client:
        monkeypatch.setattr(models.openai, "OpenAI", lambda **kwargs: client)
        target = tmp_path / "attempt"
        models.call_model("openai", payload, target, repository=repository)

    # Every count/generation pair must carry the same complete ordered history.
    assert [path for path, _ in requests] == [
        "/v1/responses/input_tokens",
        "/v1/responses",
        "/v1/responses/input_tokens",
        "/v1/responses",
        "/v1/responses/input_tokens",
        "/v1/responses",
    ]
    events = [
        json.loads(row) for row in (target / "repository-tools.jsonl").read_text().splitlines()
    ]
    assert len(events) == 2
    assert events[0]["result"]["matches"][0]["line"] == 469
    expected = [{"role": "user", "content": payload}]
    for turn in range(3):
        count, generation = requests[2 * turn][1], requests[2 * turn + 1][1]
        assert count["input"] == expected
        assert generation["input"] == expected
        assert generation["include"] == ["reasoning.encrypted_content"]
        if turn < 2:
            result = requests[2 * turn + 3][1]["input"][-1]
            assert result["type"] == "function_call_output"
            assert result["call_id"] == f"call-{turn + 1}"
            assert json.loads(result["output"]) == events[turn]["result"]
            expected = [*expected, *tool_outputs[turn], result]

    # Only a final parsed report is published, with all three turns charged.
    assert ChunkReview.model_validate_json((target / "openai.json").read_text()).model_dump() == {
        **report,
        "requests": [],
    }
    assert json.loads((target / "openai.usage.json").read_text()) == {
        "input": 240,
        "output": 30,
        "cached": 60,
        "cache_write": 0,
    }
