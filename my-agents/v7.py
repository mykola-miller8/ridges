import os
import re
import json
import uuid
import time
import ast
from typing import Any, Dict, List, Optional, Tuple

import requests


# v7 agent: generic Python code agent that proposes a full-file patch for main.py only.
# - Never embeds problem-specific constants or dataset names
# - Uses only the inference gateway exposed via INFERENCE_URL/SANDBOX_PROXY_URL
# - Returns a unified diff that replaces main.py entirely
# - Adds robust code block extraction and syntax validation with a self-repair loop


DEFAULT_PROXY_URL = (
    os.getenv("INFERENCE_URL")
    or os.getenv("SANDBOX_PROXY_URL")
)

if not DEFAULT_PROXY_URL:
    # Enforce use of the provided inference gateway only (no hardcoded defaults)
    raise RuntimeError(
        "Missing INFERENCE_URL/SANDBOX_PROXY_URL. Agent must use the provided inference gateway."
    )

AGENT_MODELS: List[str] = [
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "zai-org/GLM-4.5-FP8",
    "moonshotai/Kimi-K2-Instruct",
    "deepseek-ai/DeepSeek-V3-0324",
]


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _build_single_file_patch(filename: str, new_content: str) -> str:
    """Build a full-file unified diff for filename replacing its contents with new_content."""
    old = _read(filename)
    old_lines = old.splitlines()
    new_lines = new_content.splitlines()
    header = [
        f"diff --git a/{filename} b/{filename}",
        "index 0000000..1111111 100644",
        f"--- a/{filename}",
        f"+++ b/{filename}",
        f"@@ -1,{max(1, len(old_lines))} +1,{max(1, len(new_lines))} @@",
    ]
    body: List[str] = []
    if old_lines:
        body.extend(["-" + ln for ln in old_lines])
    else:
        body.append("-")
    if new_lines:
        body.extend(["+" + ln for ln in new_lines])
    else:
        body.append("+")
    return "\n".join(header + body) + "\n"


def _extract_main_py(response: str) -> str:
    """Extract main.py source from an LLM response.

    Tries multiple robust strategies in order:
    1) A python fenced block starting with a '# main.py' header
    2) Any python/py fenced block that contains likely Python code
    3) Any generic fenced block that contains likely Python code
    4) A direct '# main.py' header followed by code until next fence/end
    """
    if not response:
        return ""

    # 1) Exact format with header
    patterns = [
        r"```python\s*\n#\s*main\.py\s*\n([\s\S]*?)\n```",
        r"```py\s*\n#\s*main\.py\s*\n([\s\S]*?)\n```",
    ]
    for pat in patterns:
        m = re.findall(pat, response, re.DOTALL | re.IGNORECASE)
        if m and m[0].strip():
            return m[0].strip()

    # 2) Any python/py fenced block
    for pat in (r"```python\s*\n([\s\S]*?)\n```", r"```py\s*\n([\s\S]*?)\n```"):
        m = re.findall(pat, response, re.DOTALL)
        for block in m:
            code = block.strip()
            if _looks_like_python(code):
                return code

    # 3) Any generic fenced block
    m = re.findall(r"```\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    for block in m:
        code = block.strip()
        if _looks_like_python(code):
            return code

    # 4) Loose header hint
    m = re.search(r"#\s*main\.py\s*\n([\s\S]+)$", response, re.IGNORECASE)
    if m:
        chunk = m.group(1).strip()
        if _looks_like_python(chunk):
            return chunk

    return ""


def _looks_like_python(code: str) -> bool:
    """Heuristic to detect Python code blocks before parsing."""
    if not code:
        return False
    if any(tok in code for tok in ("def ", "class ", "import ", "from ")):
        return True
    # Allow simple constant-only modules too
    return bool(re.search(r"^[\w\s#'\"_=:+\-/*%().,]+$", code, re.MULTILINE))


def _validate_python(code: str) -> Tuple[bool, Optional[str]]:
    """Return (is_valid, error_message)."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        # Provide a concise, stable message back to the model if we ask for repair
        loc = f"line {e.lineno}, column {e.offset}" if e.lineno else "unknown location"
        msg = e.msg or "invalid syntax"
        return False, f"SyntaxError at {loc}: {msg}"
    except Exception as e:
        return False, f"ParseError: {e}"


def _call_llm(messages: List[Dict[str, str]], run_id: str, attempt: int, timeout_s: int = 240) -> str:
    url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
    headers = {"Content-Type": "application/json"}
    model = AGENT_MODELS[attempt % len(AGENT_MODELS)]
    try:
        uuid.UUID(str(run_id))
        valid_run_id = str(run_id)
    except Exception:
        valid_run_id = str(uuid.uuid4())
    body = {
        "run_id": valid_run_id,
        "messages": messages,
        "temperature": 0.0,
        "agent_id": "agent-v7",
        "model": model,
    }
    last_err: Optional[Exception] = None
    # Light exponential backoff with jitter
    for r in range(3):
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("choices"):
                return (data["choices"][0].get("message", {}) or {}).get("content") or ""
            if isinstance(data, str):
                return data
            return json.dumps(data)
        except Exception as e:
            last_err = e
            # Backoff: 0.5, 1.0, 2.0 (+small jitter)
            delay = (0.5 * (2 ** r)) + (0.05 * (r + 1))
            time.sleep(delay)
    raise last_err if last_err else RuntimeError("LLM call failed")


def _gen_messages(problem_statement: str, summary: str, mode: str) -> List[Dict[str, str]]:
    system_msg = (
        "You are a senior Python engineer.\n"
        + ("Do not modify tests.py; only change main.py.\n" if mode == "tests_available" else "")
        + "Follow these constraints strictly:\n"
        "- Implement only in main.py. Keep public API/signatures from the skeleton.\n"
        "- Deterministic, side-effect-free (no prints), no network or file I/O.\n"
        "- Handle input validation and edge cases explicitly with clear exceptions.\n"
        "- Prefer readability: type hints, docstrings, small helpers.\n"
        "- Python only. Do not use external resources.\n"
        "Output format (MANDATORY):\n"
        "```python\n# main.py\n[complete code]\n```\n"
        "Return exactly one code block and nothing else."
    )
    user_msg = (
        f"Problem Statement (trimmed if long):\n{problem_statement[:12000]}\n\n"
        f"Repository Summary:\n{summary}\n\n"
        "Implement so that all available tests (if present) would pass."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _gen_repair_messages(problem_statement: str, summary: str, broken_code: str, parse_error: str, mode: str) -> List[Dict[str, str]]:
    system_msg = (
        "You are a senior Python engineer fixing a syntax error in main.py.\n"
        + ("Do not modify tests.py; only change main.py.\n" if mode == "tests_available" else "")
        + "Correct the code to resolve the parser error while preserving the intended behavior.\n"
        "Output format (MANDATORY):\n"
        "```python\n# main.py\n[complete corrected code]\n```\n"
        "Return exactly one code block and nothing else."
    )
    user_msg = (
        "The previous main.py had a syntax/parsing error.\n"
        f"Error: {parse_error}\n\n"
        "Here is the broken code to fix:\n"
        f"```python\n# main.py\n{broken_code}\n```\n\n"
        f"Problem Statement:\n{problem_statement[:8000]}\n\n"
        f"Repository Summary:\n{summary}"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False) -> str:
    """Entry point required by the evaluation harness.

    Returns a unified diff patch that fully replaces main.py.
    """
    run_id = (input_dict or {}).get("run_id", os.getenv("RUN_ID", str(uuid.uuid4())))

    if repo_dir and os.path.exists(repo_dir):
        try:
            os.chdir(repo_dir)
        except Exception:
            pass

    problem_statement = (input_dict or {}).get("problem_statement", "") or ""
    mode = (input_dict or {}).get("problem_category", None)
    if mode not in ("spec_only", "tests_available"):
        mode = "tests_available" if os.path.exists("tests.py") else "spec_only"

    # Compact repository summary for context (generic, no problem-specific assumptions)
    parts: List[str] = []
    for name in ("main.py", "tests.py"):
        content = _read(name)
        if content:
            parts.append(f"### {name}\n```python\n{content[:8000]}\n```")
    summary = "\n\n".join(parts)

    messages = _gen_messages(problem_statement, summary, mode)

    # Try multiple models and attempts; include a self-repair loop on syntax errors
    num_models = len(AGENT_MODELS)
    for attempt in range(num_models):
        try:
            resp = _call_llm(messages, run_id, attempt, 300)
        except Exception:
            continue

        main_src = _extract_main_py(resp)
        if not main_src:
            # One shot: ask the model to output only the required block if it didn't
            nudged_messages = messages + [
                {
                    "role": "system",
                    "content": "Output was invalid. Return exactly one code block formatted as ```python\\n# main.py\\n...\\n``` with no extra text.",
                }
            ]
            try:
                resp2 = _call_llm(nudged_messages, run_id, attempt, 240)
                main_src = _extract_main_py(resp2)
            except Exception:
                main_src = ""

        if not main_src:
            continue

        ok, err = _validate_python(main_src)
        if ok:
            return _build_single_file_patch("main.py", main_src)

        # Attempt a small number of syntax-repair iterations for this model
        repair_attempts = 2
        broken = main_src
        for _ in range(repair_attempts):
            repair_messages = _gen_repair_messages(problem_statement, summary, broken, err or "invalid syntax", mode)
            try:
                resp_fix = _call_llm(repair_messages, run_id, attempt, 240)
            except Exception:
                break
            candidate = _extract_main_py(resp_fix)
            if not candidate:
                break
            ok2, err2 = _validate_python(candidate)
            if ok2:
                return _build_single_file_patch("main.py", candidate)
            broken, err = candidate, err2

    # If all attempts fail, return empty string (no-op)
    return ""

