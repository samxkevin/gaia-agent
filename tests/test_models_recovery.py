from types import SimpleNamespace

from models import FailoverModel
from smolagents.models import ChatMessage, MessageRole


class DummyTool:
    def __init__(self, name):
        self.name = name


def test_parse_python_style_tool_calls():
    text = (
        "Calling tools:\n"
        "[{'id': 'call_1', 'type': 'function', 'function': "
        "{'name': 'web_search', 'arguments': {'query': 'Mercedes Sosa'}}}]"
    )

    calls = FailoverModel._parse_serialized_tool_calls(text)

    assert len(calls) == 1
    assert calls[0]["id"] == "call_1"
    assert calls[0]["function"]["name"] == "web_search"


def test_repair_text_tool_call_before_smolagents_parser():
    result = SimpleNamespace(
        content=(
            "Calling tools:\n"
            "[{'id': 'call_1', 'type': 'function', 'function': "
            "{'name': 'web_search', 'arguments': {'query': 'example'}}}]"
        ),
        tool_calls=None,
    )

    repaired = FailoverModel._repair_text_tool_calls(
        result,
        [DummyTool("web_search")],
    )

    assert repaired.tool_calls is not None
    assert len(repaired.tool_calls) == 1
    assert repaired.tool_calls[0].id == "call_1"
    assert repaired.tool_calls[0].function.name == "web_search"


def test_native_retry_preserves_cohere_tool_protocol():
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=[{"type": "text", "text": "System instructions"}],
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=[{"type": "text", "text": "Find something"}],
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content=[{"type": "text", "text": "I will search."}],
        ),
        ChatMessage(
            role=MessageRole.TOOL_CALL,
            content=[
                {
                    "type": "text",
                    "text": (
                        "Calling tools:\n"
                        "[{'id': 'call_1', 'type': 'function', 'function': "
                        "{'name': 'web_search', 'arguments': "
                        "{'query': 'example'}}]"
                    ),
                }
            ],
        ),
        ChatMessage(
            role=MessageRole.TOOL_RESPONSE,
            content=[{"type": "text", "text": "Observation:\nresult text"}],
        ),
    ]

    native = FailoverModel._cohere_native_messages(messages)

    assert native[0] == {
        "role": "system",
        "content": "System instructions",
    }
    assert native[1] == {
        "role": "user",
        "content": "Find something",
    }
    assert native[2]["role"] == "assistant"
    assert native[2]["tool_calls"][0]["id"] == "call_1"
    assert native[2]["tool_calls"][0]["function"]["name"] == "web_search"
    assert native[3]["role"] == "tool"
    assert native[3]["tool_call_id"] == "call_1"
    assert native[3]["content"][0]["type"] == "document"
    assert native[3]["content"][0]["document"]["data"] == "Observation:\nresult text"
