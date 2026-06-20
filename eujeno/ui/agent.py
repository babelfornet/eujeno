# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import json


def run_tool_loop(messages, tools, call_model, call_tool, max_iters=6):
    """Tool-calling loop: calls the model; if it returns tool_calls it runs them (call_tool),
    feeds the results back as role:'tool' messages, and repeats until the model gives a
    final answer (or the iterations run out). call_model(messages, tools)->message dict;
    call_tool(name, args_dict)->str."""
    convo = list(messages)
    tool_runs = []
    last = {"role": "assistant", "content": ""}
    for _ in range(max_iters):
        last = call_model(convo, tools)
        tcs = last.get("tool_calls") or []
        if not tcs:
            break
        convo.append({"role": "assistant", "content": last.get("content"), "tool_calls": tcs})
        for tc in tcs:
            fn = tc["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            result = call_tool(fn["name"], args)
            tool_runs.append({"name": fn["name"], "arguments": args, "result": result})
            convo.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": str(result)})
    return {"content": last.get("content") or "", "messages": convo, "tool_runs": tool_runs}
