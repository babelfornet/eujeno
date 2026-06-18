import json
from axyn.ui.agent import run_tool_loop


def test_loop_executes_tool_then_finishes():
    calls = {"n": 0}

    def call_model(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"role": "assistant", "content": None,
                    "tool_calls": [{"id": "c0", "type": "function",
                                    "function": {"name": "echo__echo", "arguments": json.dumps({"text": "ciao"})}}]}
        return {"role": "assistant", "content": "Il tool ha detto: echo: ciao"}

    executed = []

    def call_tool(name, args):
        executed.append((name, args))
        return "echo: ciao"

    result = run_tool_loop([{"role": "user", "content": "usa echo"}],
                           tools=[{"type": "function", "function": {"name": "echo__echo"}}],
                           call_model=call_model, call_tool=call_tool, max_iters=4)
    assert result["content"] == "Il tool ha detto: echo: ciao"
    assert executed == [("echo__echo", {"text": "ciao"})]
    assert any(s["role"] == "tool" for s in result["messages"])


def test_loop_no_tools_returns_immediately():
    def call_model(messages, tools):
        return {"role": "assistant", "content": "ciao diretto"}
    result = run_tool_loop([{"role": "user", "content": "ciao"}], tools=[],
                           call_model=call_model, call_tool=lambda n, a: "", max_iters=4)
    assert result["content"] == "ciao diretto"
    assert result["tool_runs"] == []
