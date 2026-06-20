import json
from eujeno.net.tools import extract_tool_calls


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
