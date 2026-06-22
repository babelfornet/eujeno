#!/usr/bin/env python3
"""Minimal agentic code-agent (PI / codex-style) backed by an Eujeno network.

It drives the model served by the distributed Eujeno swarm through the
coordinator's OpenAI-compatible ``/v1/chat/completions`` endpoint, and runs a
real coding loop:

  1. tool-calling  — asks the model to call ``write_file`` to create the file;
  2. fallback      — if the model can't drive tool-calls (tiny models often
                     can't), it falls back to plain fenced code-gen + extract;
  3. self-repair   — runs the file, and on failure feeds the code + traceback
                     back to the model, rewrites it, and retries (up to 3x).

Pure standard library (no ``openai`` / ``requests`` needed).

Prerequisite — an OPERATIONAL network, e.g. on the default port 9000:

    eujeno up --model Qwen/Qwen2.5-1.5B-Instruct      # coordinator + one node

Then:

    python specs/examples/code_agent.py               # default: write fib.py, run it

Configure via environment:

    EUJENO_BASE   coordinator /v1 base url   (default http://127.0.0.1:9000/v1)
    EUJENO_MODEL  model id                   (default "eujeno" — the served model)
    EUJENO_FILE   filename to produce        (default fib.py)
    EUJENO_TASK   tool-calling instruction   (the agentic prompt)
    EUJENO_SPEC   plain code-gen instruction (the fallback prompt)
    EUJENO_SEED   if set, skip generation and repair THIS (broken) source instead
    EUJENO_DEBUG  set to print finish_reason for each call

Tip: ~1.5B is the practical floor for reliable tool-calling. A 0.5B model only
exercises the *mechanism* (it works, but collapses to empty turns / malformed
tool JSON often enough that the fallback + repair paths carry the run).
"""
import json, os, re, subprocess, sys, tempfile, textwrap, urllib.request

BASE = os.environ.get("EUJENO_BASE", "http://127.0.0.1:9000/v1")
MODEL = os.environ.get("EUJENO_MODEL", "eujeno")
DEBUG = bool(os.environ.get("EUJENO_DEBUG"))
WORK = os.path.join(tempfile.gettempdir(), "eujeno-agent-workspace")
os.makedirs(WORK, exist_ok=True)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write text content to a file (relative path inside the workspace).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "filename, e.g. fib.py"},
                "content": {"type": "string", "description": "full file contents"},
            },
            "required": ["path", "content"],
        },
    },
}]


def chat(messages, tools=TOOLS):
    """One /v1/chat/completions call. Pass tools=None for plain code-gen."""
    payload = {"model": MODEL, "messages": messages,
               "temperature": 0.1, "max_tokens": 512}
    if tools:                              # omit entirely for plain code-gen
        payload["tools"] = tools
        payload["tool_choice"] = "auto"    # (the coordinator always behaves as "auto")
    req = urllib.request.Request(BASE + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        choice = json.load(r)["choices"][0]
    if DEBUG:
        print(f"  [debug] finish_reason={choice.get('finish_reason')}")
    return choice["message"]


def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text or "", re.S)
    return m.group(1) if m else None


def _unescape(s):
    return (s.replace('\\\\', '\x00').replace('\\n', '\n').replace('\\t', '\t')
             .replace('\\"', '"').replace("\\'", "'").replace('\x00', '\\'))


def loose_toolcall(text):
    """Tolerant parser for a ``<tool_call>{...}`` block that a small model often
    emits with truncated/invalid JSON the strict server-side parser drops."""
    if not text or "<tool_call>" not in text:
        return None
    blob = text.split("<tool_call>", 1)[1].split("</tool_call>")[0]
    try:                                            # happy path: valid JSON
        obj = json.loads(blob)
        return obj.get("arguments", obj)
    except Exception:
        pass
    pm = re.search(r'"path"\s*:\s*"([^"]+)"', blob)
    cm = re.search(r'"content"\s*:\s*"(.*)', blob, re.S)
    if not cm:
        return None
    raw = re.sub(r'["}\s]*$', '', cm.group(1))      # trim dangling json closers
    return {"path": pm.group(1) if pm else "out.py", "content": _unescape(raw)}


def run_write_file(args):
    path = os.path.basename(args["path"])           # sandbox: no path traversal
    full = os.path.join(WORK, path)
    with open(full, "w") as f:
        f.write(args["content"])
    print(f"  [tool] write_file -> {full} ({len(args['content'])} bytes)")
    return full, f"wrote {len(args['content'])} bytes to {path}"


def looks_runnable(code):
    """Reject content that lost its newlines/indentation (a tiny-model failure)."""
    return code and "def " in code and "\n" in code


def plain_codegen(filename, spec, tries=4):
    """Fallback: ask for a fenced code block (no tools). Far more reliable for a
    tiny model than escaping a multi-line file through tool-call JSON."""
    for t in range(tries):
        msg = chat([
            {"role": "system", "content": "You are a coding assistant. Output only code."},
            {"role": "user", "content": spec},
        ], tools=None)
        code = extract_code(msg.get("content")) or (msg.get("content") or "")
        if looks_runnable(code):
            full, _ = run_write_file({"path": filename, "content": code})
            return full
        print(f"  [codegen] attempt {t+1}: unusable output, retrying…")
    return None


def agent(task, filename, spec, max_steps=4):
    print(f"== TASK ==\n{task}\n== AGENT (eujeno backend: {MODEL}) ==")
    messages = [
        {"role": "system", "content":
         "You are a coding agent. Use the write_file tool to create the requested "
         "file(s). After writing, reply with a one-line confirmation."},
        {"role": "user", "content": task},
    ]
    # Phase 1 — agentic tool-calling loop (retries absorb a tiny model's empty turns)
    for step in range(max_steps):
        msg = chat(messages)
        tcs = msg.get("tool_calls")
        if tcs:
            try:
                args = json.loads(tcs[0]["function"]["arguments"])
            except json.JSONDecodeError:
                args = None
            if args and looks_runnable(args.get("content", "")):
                full, _ = run_write_file(args)
                return [full]
        else:
            args = loose_toolcall(msg.get("content") or "")
            if args and looks_runnable(args.get("content", "")):
                print("  [recover] salvaged a malformed tool call from raw output")
                full, _ = run_write_file(args)
                return [full]
        print(f"  [step {step+1}] no usable tool call yet, retrying…")
    # Phase 2 — robust fallback: plain fenced code-gen (what real agents fall back to)
    print("  [fallback] tool loop exhausted -> plain code-gen")
    full = plain_codegen(filename, spec)
    return [full] if full else []


def run_file(path):
    out = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=30)
    return out.returncode, (out.stdout + out.stderr).strip()


def repair_loop(path, max_fixes=3):
    """Run the file; on failure feed the code + error back to the model, write the
    corrected version, and retry. This is the loop a real code-agent runs."""
    name = os.path.basename(path)
    for attempt in range(max_fixes + 1):
        rc, output = run_file(path)
        print(f"  $ python {name}   (exit {rc})")
        print(textwrap.indent(output or "(no output)", "  > "))
        if rc == 0:
            print(f"  [repair] success after {attempt} fix(es)")
            return True
        if attempt == max_fixes:
            break
        print(f"  [repair] failed -> asking the model to fix it (fix {attempt+1}/{max_fixes})")
        code = open(path).read()
        fix_prompt = (
            f"This Python file fails when run.\n\nFile {name}:\n```python\n{code}\n```\n\n"
            f"Error:\n{output}\n\nFix the bug and return the corrected COMPLETE file as "
            "one ```python block. It must run with no errors and print OK.")
        msg = chat([
            {"role": "system", "content": "You are a coding assistant. Output only code."},
            {"role": "user", "content": fix_prompt},
        ], tools=None)
        new_code = extract_code(msg.get("content")) or (msg.get("content") or "")
        if not looks_runnable(new_code):
            print("  [repair] model returned unusable output; stopping")
            break
        run_write_file({"path": name, "content": new_code})
        print(f"  --- {name} (revised) ---")
        print(textwrap.indent(new_code, "  | "))
    print(f"  [repair] still failing after {max_fixes} attempt(s)")
    return False


if __name__ == "__main__":
    fname = os.environ.get("EUJENO_FILE", "fib.py")
    task = os.environ.get("EUJENO_TASK",
        "Create a file fib.py with a function fib(n) that returns the n-th "
        "Fibonacci number (fib(0)=0, fib(1)=1) and an `if __name__` block that "
        "prints fib(10). Use the write_file tool.")
    spec = os.environ.get("EUJENO_SPEC",
        "Write a Python function fib(n) that returns the n-th Fibonacci number "
        "(fib(0)=0, fib(1)=1). Then add code that prints fib(10).")
    seed = os.environ.get("EUJENO_SEED")     # demo: start from a deliberately broken file
    if seed:
        full, _ = run_write_file({"path": fname, "content": seed})
        print(f"== SEEDED a broken {fname}; the agent must repair it ==")
        files = [full]
    else:
        files = [f for f in agent(task, fname, spec) if f]
    print("\n== VERIFY + SELF-REPAIR ==")
    all_ok = True
    for f in files:
        if f.endswith(".py"):
            print(f"--- {f} (initial) ---")
            print(textwrap.indent(open(f).read(), "  "))
            all_ok = repair_loop(f) and all_ok
    if not files:
        print("  (no files were written by the agent)")
    else:
        print("\nRESULT:", "all files run clean" if all_ok else "some files still failing")
    sys.exit(0 if (files and all_ok) else 1)
