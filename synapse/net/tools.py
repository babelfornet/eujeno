import json
import re

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def extract_tool_calls(text: str):
    """Estrae i tool call dal formato Qwen2.5 (<tool_call>{json}</tool_call>) e li
    converte nel formato OpenAI. Ritorna (content_senza_toolcall, [tool_call...])."""
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
