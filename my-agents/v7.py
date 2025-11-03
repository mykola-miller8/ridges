import os
import re
import ast
import json
import uuid
import time
from typing import Any, Dict, List

import requests


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
    """Read file contents, returning empty string on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _validate_syntax(code: str) -> bool:
    """Check if code has valid Python syntax."""
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def _build_single_file_patch(filename: str, new_content: str) -> str:
    """Build unified diff patch replacing file contents."""
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
    """Extract Python code from LLM response."""
    if not response:
        return ""
    # Prefer explicitly headed block
    m = re.findall(r"```python\s*\n#\s*main\.py\n([\s\S]*?)\n```", response, re.DOTALL)
    if m and m[0].strip():
        return m[0].strip()
    # Fallback: first python block
    m2 = re.findall(r"```python\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    return m2[0].strip() if m2 and m2[0].strip() else ""


def _call_llm(messages: List[Dict[str, str]], run_id: str, model_idx: int) -> str:
    """Call inference gateway with specified model."""
    url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
    headers = {"Content-Type": "application/json"}
    model = AGENT_MODELS[model_idx % len(AGENT_MODELS)]
    body = {
        "run_id": run_id,
        "messages": messages,
        "temperature": 0.0,
        "agent_id": "agent-v7",
        "model": model,
    }
    for retry in range(3):
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("choices"):
                return (data["choices"][0].get("message", {}) or {}).get("content") or ""
            if isinstance(data, str):
                return data
            return json.dumps(data)
        except Exception:
            if retry == 2:
                raise
            time.sleep(1 + retry)
    return ""


def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False) -> str:
    """
    Main entry point for the evaluation harness.
    Generates a solution and returns a unified diff patch for main.py.
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

    # Build repository context
    parts: List[str] = []
    for name in ("main.py", "tests.py"):
        content = _read(name)
        if content:
            parts.append(f"### {name}\n```python\n{content[:8000]}\n```")
    repo_summary = "\n\n".join(parts)

    system_msg = (
        "You are an expert Python engineer. Write correct, bug-free code.\n"
        + ("Do not modify tests.py; only change main.py.\n" if mode == "tests_available" else "")
        + "Return ONLY:\n```python\n# main.py\n[code]\n```"
    )
    
    user_msg = f"""Problem:
{problem_statement[:12000]}

Repository:
{repo_summary}

Implement complete solution following these rules:

1. For list[str] returns: Each element is ONE line, not multiple lines joined with \\n
2. Test edge cases: first item, last item, empty inputs
3. For tuple validation: len(item) < 2 catches both empty and incomplete (not len < 1)
4. Match error messages and exception types exactly as specified

Implement now."""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Try each model
    for model_idx in range(len(AGENT_MODELS)):
        try:
            response = _call_llm(messages, run_id, model_idx)
            code = _extract_main_py(response)
            
            if code and _validate_syntax(code):
                return _build_single_file_patch("main.py", code)
        except Exception:
            continue
    
    return ""
