# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import json
import re

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def flatten_content(content):
    """Normalize an OpenAI message ``content`` to a plain string.

    Real clients (PI, the OpenAI SDK, LiteLLM, ...) may send ``content`` as a
    list of typed parts, e.g. ``[{"type": "text", "text": "hi"}]`` instead of a
    bare string. The HF chat templates expect a string, so collapse the text
    parts; leave strings and ``None`` (assistant tool-call turns) untouched.
    """
    if isinstance(content, list):
        return "".join(
            (p.get("text") or p.get("content") or "") if isinstance(p, dict) else str(p)
            for p in content
        )
    return content


def normalize_messages(messages):
    """Return a copy of ``messages`` with every ``content`` collapsed to a string."""
    return [{**m, "content": flatten_content(m.get("content"))} for m in messages]


def openai_stream_chunks(content, tool_calls, finish_reason, model_id, chunk_id, created):
    """Yield OpenAI-style SSE lines for an already-generated completion.

    Eujeno generates the whole response (store-and-forward), so rather than true
    token-by-token streaming this emits a short ``chat.completion.chunk`` stream:
    a role delta, the content (or tool_calls) delta, then a terminal chunk with
    the ``finish_reason`` and ``[DONE]``. That's enough for streaming clients
    (PI, IDE agents) that require SSE and a final ``finish_reason``.
    """
    def line(delta, fr=None):
        return "data: " + json.dumps({
            "id": chunk_id, "object": "chat.completion.chunk", "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": fr}],
        }) + "\n\n"

    yield line({"role": "assistant"})
    if tool_calls:
        yield line({"tool_calls": [{**t, "index": i} for i, t in enumerate(tool_calls)]})
    elif content:
        yield line({"content": content})
    yield line({}, finish_reason)
    yield "data: [DONE]\n\n"


def extract_tool_calls(text: str):
    """Extracts tool calls from the Qwen2.5 format (<tool_call>{json}</tool_call>) and
    converts them to the OpenAI format. Returns (content_without_toolcall, [tool_call...])."""
    calls = []
    for i, block in enumerate(_TOOL_CALL_RE.findall(text)):
        try:
            obj = json.loads(block)
        except Exception:
            continue
        calls.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {
                "name": obj.get("name", ""),
                "arguments": json.dumps(obj.get("arguments", {})),
            },
        })
    content = _TOOL_CALL_RE.sub("", text).strip()
    return content, calls
