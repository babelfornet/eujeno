# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import json
import re

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


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
