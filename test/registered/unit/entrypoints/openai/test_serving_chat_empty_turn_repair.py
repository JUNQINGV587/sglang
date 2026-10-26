"""Unit tests for the empty terminal turn repair.

A model can close its thinking block and stop on whitespace: the turn then
carries no visible text and no tool call. Agent clients score that as a dead
turn — the run stalls and the user sees a stream error — even though the engine
finished the request normally. These tests pin the detection rules and the
single bounded retry that answers such a turn with thinking switched off.
"""

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageContentTextPart,  # noqa: F401  (import guard)
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    FunctionResponse,
    ToolCall,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _chat_response(
    content: str | None,
    *,
    reasoning: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str = "stop",
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="chatcmpl-test",
        created=0,
        model="qwen3-test",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=content,
                    reasoning_content=reasoning,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


def _bare_serving() -> OpenAIServingChat:
    """A serving object with only the attributes the repair path needs."""
    return OpenAIServingChat.__new__(OpenAIServingChat)


class TestEmptyTurnDetection(CustomTestCase):
    def setUp(self):
        self.serving = _bare_serving()

    def test_whitespace_only_answer_after_thinking_is_an_empty_turn(self):
        # The reproduced pathology: thinking runs, then the turn stops on "\n\n".
        self.assertTrue(
            self.serving._is_empty_terminal_turn(
                "stop", "\n\n", 0, reasoning_tokens=221
            )
        )
        self.assertTrue(
            self.serving._is_empty_terminal_turn("stop", None, 0, reasoning_tokens=61)
        )

    def test_a_real_answer_is_not_an_empty_turn(self):
        self.assertFalse(
            self.serving._is_empty_terminal_turn(
                "stop", "done — the file is written.", 0, reasoning_tokens=221
            )
        )

    def test_tool_calls_are_never_repaired(self):
        self.assertFalse(
            self.serving._is_empty_terminal_turn(
                "stop", "\n\n", 1, reasoning_tokens=221
            )
        )

    def test_only_natural_stops_are_repaired(self):
        for finish_reason in ("length", "abort", "content_filter", "tool_calls"):
            self.assertFalse(
                self.serving._is_empty_terminal_turn(
                    finish_reason, "", 0, reasoning_tokens=221
                )
            )

    def test_a_turn_without_reasoning_is_left_alone(self):
        # An empty answer that never involved thinking may be what was asked for.
        self.assertFalse(self.serving._is_empty_terminal_turn("stop", "", 0, 0))
        self.assertTrue(
            self.serving._is_empty_terminal_turn(
                "stop", "", 0, 0, reasoning_text="I thought about it."
            )
        )

    def test_stream_prefilter_matches_the_reproduced_tokens(self):
        # 223 completion tokens of which 221 were reasoning -> 2 visible tokens.
        self.assertTrue(self.serving._stream_turn_may_be_empty("stop", False, 223, 221))
        self.assertFalse(self.serving._stream_turn_may_be_empty("stop", False, 223, 0))
        self.assertFalse(
            self.serving._stream_turn_may_be_empty("stop", False, 900, 200)
        )
        self.assertFalse(
            self.serving._stream_turn_may_be_empty("length", False, 223, 221)
        )
        self.assertFalse(self.serving._stream_turn_may_be_empty("stop", True, 223, 221))


class TestEmptyTurnResponseCheck(CustomTestCase):
    def setUp(self):
        self.serving = _bare_serving()

    def test_whitespace_response_is_empty(self):
        response = _chat_response("\n\n", reasoning="planning ...")
        meta = {"meta_info": {"reasoning_tokens": 221}}
        self.assertTrue(self.serving._response_is_empty_turn(response, meta))

    def test_answered_response_is_not_empty(self):
        response = _chat_response("All tests pass.", reasoning="planning ...")
        self.assertFalse(self.serving._response_is_empty_turn(response, None))

    def test_tool_call_response_is_not_empty(self):
        call = ToolCall(
            id="call-1",
            index=0,
            function=FunctionResponse(name="bash", arguments='{"command": "ls"}'),
        )
        response = _chat_response(None, reasoning="planning ...", tool_calls=[call])
        self.assertFalse(self.serving._response_is_empty_turn(response, None))

    def test_reasoning_tokens_alone_mark_a_thinking_turn(self):
        # No reasoning_content in the message, but the engine counted thinking.
        response = _chat_response("\n", reasoning=None)
        meta = {"meta_info": {"reasoning_tokens": 40}}
        self.assertTrue(self.serving._response_is_empty_turn(response, meta))


class TestEmptyTurnRepair(CustomTestCase):
    def _serving_with(self, generate_request):
        serving = _bare_serving()
        serving.tokenizer_manager = SimpleNamespace(generate_request=generate_request)
        serving._build_chat_response = lambda request, ret, created: _chat_response(
            ret[0]["text"]
        )
        return serving

    def test_repair_retries_once_with_thinking_off(self):
        seen = {}

        def _convert(request, raw_request=None):
            seen["request"] = request
            return "adapted", request

        async def _generate(adapted, raw_request):
            yield {"text": "the answer", "meta_info": {"id": "x"}}

        serving = self._serving_with(_generate)
        serving._convert_to_internal_request = _convert

        request = ChatCompletionRequest(
            model="qwen3-test",
            messages=[{"role": "user", "content": "plan and then answer"}],
            chat_template_kwargs={"enable_thinking": True},
        )
        repaired = asyncio.run(serving._repair_empty_terminal_turn(request, None))

        self.assertIsNotNone(repaired)
        self.assertEqual(repaired.choices[0].message.content, "the answer")
        retry = seen["request"]
        self.assertFalse(retry.stream)
        self.assertIsNone(retry.stream_options)
        self.assertIs(retry.chat_template_kwargs["enable_thinking"], False)
        self.assertIs(retry.chat_template_kwargs["thinking"], False)
        self.assertEqual(retry.chat_template_kwargs["thinking_mode"], "disabled")
        # The caller's request is untouched: a repair never mutates its input.
        self.assertIs(request.chat_template_kwargs["enable_thinking"], True)

    def test_repair_failure_falls_back_to_the_original_answer(self):
        async def _generate(adapted, raw_request):
            raise RuntimeError("engine went away")
            yield  # pragma: no cover

        serving = self._serving_with(_generate)
        serving._convert_to_internal_request = lambda request, raw_request=None: (
            "adapted",
            request,
        )

        request = ChatCompletionRequest(
            model="qwen3-test",
            messages=[{"role": "user", "content": "hello"}],
            chat_template_kwargs={"enable_thinking": True},
        )
        self.assertIsNone(
            asyncio.run(serving._repair_empty_terminal_turn(request, None))
        )

    def test_env_switch_disables_the_repair(self):
        serving = _bare_serving()
        self.assertTrue(serving._empty_turn_repair_enabled())
        with patch.dict(os.environ, {"SGLANG_DISABLE_EMPTY_TURN_REPAIR": "1"}):
            self.assertFalse(serving._empty_turn_repair_enabled())

    def test_repaired_tool_call_is_streamed_as_one_complete_chunk(self):
        serving = _bare_serving()
        call = ToolCall(
            id="call-7",
            index=0,
            function=FunctionResponse(name="bash", arguments='{"command": "ls"}'),
        )
        request = ChatCompletionRequest(
            model="qwen3-test", messages=[{"role": "user", "content": "run ls"}]
        )

        chunk = serving._stream_tool_call_chunk(
            chunk_id="chatcmpl-test",
            created=0,
            request=request,
            index=0,
            position=0,
            call=call,
        )

        self.assertTrue(chunk.startswith("data: "))
        self.assertTrue(chunk.endswith("\n\n"))
        payload = json.loads(chunk[len("data: ") : -2])
        delta = payload["choices"][0]["delta"]
        self.assertIsNone(payload["choices"][0]["finish_reason"])
        self.assertEqual(delta["tool_calls"][0]["id"], "call-7")
        self.assertEqual(delta["tool_calls"][0]["index"], 0)
        self.assertEqual(delta["tool_calls"][0]["function"]["name"], "bash")
        self.assertEqual(
            delta["tool_calls"][0]["function"]["arguments"], '{"command": "ls"}'
        )


if __name__ == "__main__":
    unittest.main()
