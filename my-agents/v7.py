import os
import re
import json
import uuid
import time
import subprocess
import ast
from typing import Any, Dict, List, Tuple

import requests


# v7 seed agent: generic code agent that proposes a full-file patch for main.py only.
# - Never embeds problem-specific constants or dataset names
# - Uses only the inference gateway exposed via INFERENCE_URL/SANDBOX_PROXY_URL
# - Returns a unified diff that replaces main.py entirely


DEFAULT_PROXY_URL = (
    os.getenv("INFERENCE_URL")
    or os.getenv("SANDBOX_PROXY_URL")
    or "http://172.17.0.1:1234"
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


def _strip_code_fences(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text.strip()
    # Remove a single surrounding ```...``` block if present
    if s.startswith("```") and s.endswith("```"):
        m = re.findall(r"```(?:python)?\s*\n([\s\S]*?)\n```", s, re.DOTALL)
        if m and m[0].strip():
            return m[0].strip()
    return s


def _ensure_header(code: str) -> str:
    code = code.lstrip("\n")
    if not code.startswith("# main.py"):
        return "# main.py\n" + code
    return code


def _extract_main_py(response: str) -> str:
    if not response:
        return ""
    # 1) Prefer explicitly headed python block
    m = re.findall(r"```python\s*\n#\s*main\.py\n([\s\S]*?)\n```", response, re.DOTALL)
    if m and m[0].strip():
        return _ensure_header(m[0].strip())
    # 2) Any python block
    m2 = re.findall(r"```python\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    if m2 and m2[0].strip():
        return _ensure_header(m2[0].strip())
    # 3) Any code fence without language
    m3 = re.findall(r"```\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    if m3 and m3[0].strip():
        return _ensure_header(m3[0].strip())
    # 4) Fallback to raw (strip if contains def/class)
    raw = response.strip()
    if ("def " in raw) or ("class " in raw):
        return _ensure_header(_strip_code_fences(raw))
    return ""


def _validate_python(code: str) -> Tuple[bool, str]:
    try:
        # Strip header comment before parsing
        src = re.sub(r"^#\s*main\.py\s*\n", "", code)
        ast.parse(src)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}, col {e.offset}"
    except Exception as e:
        return False, f"Error: {e}"


def _repair_syntax(run_id: str, bad_code: str, error_text: str, attempt_index: int) -> str:
    """Ask the LLM to repair syntax only, returning a single code block for main.py."""
    system = (
        "You are a Python code fixer.\n"
        "Task: Repair ONLY syntax/parse errors in the provided Python file.\n"
        "Do not change behavioral intent or API. Do not add tests.\n"
        "Return EXACTLY one code block formatted as:\n"
        "```python\n# main.py\n[complete file]\n```\n"
        "No prose, no extra blocks. Deterministic output."
    )
    user = (
        "The following Python file fails to parse. Fix syntax only.\n\n"
        f"Parser error:\n{error_text}\n\n"
        "Current file:\n```python\n# main.py\n" + bad_code.strip() + "\n```\n"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        resp = _call_llm(messages, run_id, attempt_index, 180)
        fixed = _extract_main_py(resp)
        return fixed or ""
    except Exception:
        return ""


def _call_llm(messages: List[Dict[str, str]], run_id: str, attempt: int, timeout_s: int = 240) -> str:
    url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
    headers = {"Content-Type": "application/json"}
    model = AGENT_MODELS[attempt % len(AGENT_MODELS)]
    body = {
        "run_id": run_id,
        "messages": messages,
        "temperature": 0.0,
        "agent_id": "agent-v7",
        "model": model,
    }
    last_err: Exception | None = None
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
            # Linear backoff to be gentle on gateway
            time.sleep(1 + r)
    raise last_err if last_err else RuntimeError("LLM call failed")


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

    system_msg = (
        "You are a senior Python engineer.\n"
        + ("Do not modify tests.py; only change main.py.\n" if mode == "tests_available" else "")
        + "Strict output format: return EXACTLY one code block with the full file.\n"
        "Use this format only:\n```python\n# main.py\n[complete Python file]\n```\n"
        "Requirements:\n"
        "- Deterministic, no randomness, no I/O, no prints/logging.\n"
        "- Only Python standard library.\n"
        "- Precise input validation and clear exceptions consistent with the spec.\n"
        "- Type hints and concise docstrings for public functions/classes.\n"
        "- Keep code readable and minimal; avoid unnecessary abstractions.\n"
    )
    user_msg = (
        f"Problem Statement (trimmed if long):\n{problem_statement[:12000]}\n\n"
        f"Repository Summary:\n{summary}\n\n"
        "Implement strictly so all tests (if present) pass."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Rotate starting model index to diversify across runs
    start_idx = (hash(run_id) % len(AGENT_MODELS)) if AGENT_MODELS else 0
    indices = [(start_idx + i) % len(AGENT_MODELS) for i in range(len(AGENT_MODELS))]

    # Try models; validate syntax; attempt auto-repair up to 2 times per model
    for attempt_pos, model_idx in enumerate(indices):
        try:
            resp = _call_llm(messages, run_id, model_idx, 300)
            main_src = _extract_main_py(resp)
            if not main_src:
                continue
            ok, err = _validate_python(main_src)
            if ok:
                return _build_single_file_patch("main.py", main_src)
            # Try up to 2 syntax repairs
            repaired = main_src
            for fix_round in range(2):
                fixed = _repair_syntax(f"{run_id}-fix{attempt_pos}-{fix_round+1}", repaired, err, model_idx)
                if not fixed:
                    break
                ok2, err2 = _validate_python(fixed)
                if ok2:
                    return _build_single_file_patch("main.py", fixed)
                repaired, err = fixed, err2
        except Exception:
            continue

    return ""

