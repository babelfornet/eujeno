import json


def run_tool_loop(messages, tools, call_model, call_tool, max_iters=6):
    """Loop di tool-calling: chiama il modello; se ritorna tool_calls li esegue (call_tool),
    rimanda i risultati come messaggi role:'tool', e ripete finché il modello dà una
    risposta finale (o si esauriscono le iterazioni). call_model(messages, tools)->message dict;
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
