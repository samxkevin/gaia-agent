import json
from types import SimpleNamespace

import pytest
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
)

from models import FailoverModel


def message(arguments_marker=None, *, content="I determined the answer.", calls=True, raw=None):
    tool_calls = None
    if calls:
        tool_calls = [
            ChatMessageToolCall(
                function=ChatMessageToolCallFunction(
                    name="final_answer", arguments=arguments_marker
                ),
                id="call_final",
                type="function",
            )
        ]
    return ChatMessage(
        role=MessageRole.ASSISTANT,
        content=content,
        tool_calls=tool_calls,
        raw=raw,
    )


class Delegate:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error is not None:
            raise self.error
        return self.response


def repair(original, repaired):
    delegate = Delegate(repaired)
    tool = SimpleNamespace(name="final_answer")
    result = FailoverModel._repair_malformed_final_answer(
        FailoverModel.__new__(FailoverModel),
        delegate=delegate,
        result=original,
        messages=[{"role": "user", "content": "question and tool evidence"}],
        final_answer_tool=tool,
        stop_sequences=None,
        response_format=None,
        kwargs={},
    )
    return result, delegate, tool


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"answer": None},
        {"answer": ""},
        {"answer": " "},
        "3",
        "{invalid json}",
    ],
)
def test_malformed_final_answer_triggers_exactly_one_repair(arguments):
    fixed = message(json.dumps({"answer": "3"}), content=None)
    result, delegate, tool = repair(message(arguments), fixed)

    assert result is fixed
    assert len(delegate.calls) == 1
    repair_messages, options = delegate.calls[0]
    assert options["tools_to_call_from"] == [tool]
    assert [item.name for item in options["tools_to_call_from"]] == ["final_answer"]
    assert "Do not research or reconsider" in repair_messages[-1]["content"]


@pytest.mark.parametrize("answer", ["yes", 0])
def test_valid_final_answer_is_accepted_without_repair(answer):
    original = message({"answer": answer})
    result, delegate, _ = repair(original, message({"answer": "changed"}))
    assert result is original
    assert delegate.calls == []


def test_no_final_answer_call_does_not_trigger_repair():
    original = message(content="continue", calls=False)
    result, delegate, _ = repair(original, message({"answer": "changed"}))
    assert result is original
    assert delegate.calls == []


def test_invalid_repair_preserves_original_without_fabricating_answer():
    original = message({})
    invalid = message(content="3", calls=False)
    result, delegate, _ = repair(original, invalid)
    assert result is original
    assert len(delegate.calls) == 1


def test_actual_chat_message_uses_raw_reasoning_when_content_is_none():
    provider_message = SimpleNamespace(
        reasoning_content="The three qualifying albums mean the answer is 3.",
        reasoning=None,
        model_extra={},
    )
    raw = SimpleNamespace(choices=[SimpleNamespace(message=provider_message)])
    malformed = message({}, content=None, raw=raw)
    fixed = message({"answer": "3"}, content=None)

    result, delegate, _ = repair(malformed, fixed)

    assert isinstance(result, ChatMessage)
    assert result is fixed
    assert result.tool_calls[0].function.arguments == {"answer": "3"}
    prompt = delegate.calls[0][0][-1]["content"]
    assert "three qualifying albums" in prompt
    assert "Preceding reasoning" in prompt


def test_repair_exception_preserves_original_after_one_attempt():
    original = message({})
    delegate = Delegate(error=RuntimeError("repair endpoint failed"))
    tool = SimpleNamespace(name="final_answer")

    result = FailoverModel._repair_malformed_final_answer(
        FailoverModel.__new__(FailoverModel),
        delegate=delegate,
        result=original,
        messages=[{"role": "user", "content": "question and evidence"}],
        final_answer_tool=tool,
        stop_sequences=None,
        response_format=None,
        kwargs={},
    )

    assert result is original
    assert len(delegate.calls) == 1
    assert delegate.calls[0][1]["tools_to_call_from"] == [tool]


def test_multiple_calls_are_not_accepted_as_valid_repair():
    original = message({})
    invalid = message({"answer": "3"})
    invalid.tool_calls.append(
        ChatMessageToolCall(
            function=ChatMessageToolCallFunction(name="search", arguments={"q": "x"}),
            id="call_search",
            type="function",
        )
    )
    result, delegate, _ = repair(original, invalid)
    assert result is original
    assert len(delegate.calls) == 1
