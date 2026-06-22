import json
from eujeno.net.tools import (extract_tool_calls, flatten_content, normalize_messages,
                              openai_stream_chunks)


def test_single_tool_call():
    text = 'Controllo il meteo.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Roma"}}\n</tool_call>'
    content, calls = extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Roma"}
    assert "<tool_call>" not in content


def test_no_tool_calls():
    content, calls = extract_tool_calls("Ciao, come va?")
    assert calls == []
    assert content == "Ciao, come va?"


def test_multiple_tool_calls():
    text = '<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":{"x":1}}</tool_call>'
    content, calls = extract_tool_calls(text)
    assert [c["function"]["name"] for c in calls] == ["a", "b"]
    assert json.loads(calls[1]["function"]["arguments"]) == {"x": 1}


def test_malformed_tool_call_ignored():
    content, calls = extract_tool_calls("<tool_call>non-json</tool_call> resto")
    assert calls == []


def test_flatten_content_passthrough():
    assert flatten_content("hello") == "hello"
    assert flatten_content(None) is None


def test_flatten_content_openai_parts():
    # OpenAI/PI structured content: a list of typed parts -> joined text
    parts = [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]
    assert flatten_content(parts) == "Hello world"


def test_normalize_messages_mixed():
    # A real PI request: user content as parts, assistant tool-call turn (content=None)
    msgs = normalize_messages([
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c0"}]},
        {"role": "tool", "tool_call_id": "c0", "content": "ok"},
    ])
    assert msgs[0]["content"] == "hi"
    assert msgs[1]["content"] is None and msgs[1]["tool_calls"] == [{"id": "c0"}]
    assert msgs[2]["content"] == "ok"


def _parse_sse(lines):
    chunks = []
    for ln in lines:
        assert ln.startswith("data: ") and ln.endswith("\n\n")
        payload = ln[len("data: "):].strip()
        chunks.append(payload)
    return chunks


def test_stream_chunks_content():
    out = list(openai_stream_chunks("Hello", None, "stop", "m", "id1", 123))
    chunks = _parse_sse(out)
    assert chunks[-1] == "[DONE]"
    objs = [json.loads(c) for c in chunks[:-1]]
    assert objs[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert objs[1]["choices"][0]["delta"] == {"content": "Hello"}
    assert objs[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(o["object"] == "chat.completion.chunk" for o in objs)


def test_stream_chunks_tool_calls():
    tcs = [{"id": "call_0", "type": "function",
            "function": {"name": "f", "arguments": "{}"}}]
    out = list(openai_stream_chunks(None, tcs, "tool_calls", "m", "id2", 9))
    objs = [json.loads(c) for c in _parse_sse(out)[:-1]]
    delta_tc = objs[1]["choices"][0]["delta"]["tool_calls"][0]
    assert delta_tc["index"] == 0 and delta_tc["function"]["name"] == "f"
    assert objs[-1]["choices"][0]["finish_reason"] == "tool_calls"
