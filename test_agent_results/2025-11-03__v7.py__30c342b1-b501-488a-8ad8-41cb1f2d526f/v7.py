import os
import re
import ast
import json
import uuid
import time
from typing import Any, Dict, List, Optional

import requests


# v7 seed agent: generic code agent that proposes a full-file patch for main.py only.
# - Never embeds problem-specific constants or dataset names
# - Uses only the inference gateway exposed via INFERENCE_URL/SANDBOX_PROXY_URL
# - Returns a unified diff that replaces main.py entirely
# - Multi-candidate generation with diversity sampling


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
    """Read a file and return its contents, or empty string if not found."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _validate_syntax(code: str) -> Optional[str]:
    """Validate Python syntax. Returns None if valid, or error message."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return f"Parse error: {str(e)}"


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
    """Extract Python code from LLM response, preferring blocks with '# main.py' header."""
    if not response:
        return ""
    # Prefer explicitly headed block
    m = re.findall(r"```python\s*\n#\s*main\.py\n([\s\S]*?)\n```", response, re.DOTALL)
    if m and m[0].strip():
        return m[0].strip()
    # Fallback: first python block
    m2 = re.findall(r"```python\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    return m2[0].strip() if m2 and m2[0].strip() else ""


def _call_llm(
    messages: List[Dict[str, str]], 
    run_id: str, 
    attempt: int, 
    temperature: float = 0.0,
    timeout_s: int = 240
) -> str:
    """Call the inference gateway with retry logic."""
    url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
    headers = {"Content-Type": "application/json"}
    model = AGENT_MODELS[attempt % len(AGENT_MODELS)]
    body = {
        "run_id": run_id,
        "messages": messages,
        "temperature": temperature,
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
            time.sleep(1 + r)
    raise last_err if last_err else RuntimeError("LLM call failed")


def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False) -> str:
    """
    Entry point required by the evaluation harness.
    
    Strategy: Generate multiple diverse candidates, return first valid one.
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

    # Repository summary
    parts: List[str] = []
    for name in ("main.py", "tests.py"):
        content = _read(name)
        if content:
            parts.append(f"### {name}\n```python\n{content[:8000]}\n```")
    summary = "\n\n".join(parts)

    # Careful, detailed prompt for generation
    system_msg = (
        "You are a senior Python engineer who writes simple, correct code.\n"
        + ("Do not modify tests.py; only change main.py.\n" if mode == "tests_available" else "")
        + "Return ONLY:\n```python\n# main.py\n[complete code]\n```\n"
        "No explanations."
    )
    
    user_msg = (
        f"Problem:\n{problem_statement[:12000]}\n\n"
        f"Repository:\n{summary}\n\n"
        "Implement a complete solution with careful input validation.\n\n"
        "VALIDATION CHECKLIST (critical for correctness):\n"
        "1. Check input type FIRST (is it None? wrong type?)\n"
        "2. For sequences (tuples, lists), check minimum length needed\n"
        "   - If validating tuples that need N elements, check len(tuple) >= N\n"
        "   - Don't just check len > 0, check for the actual minimum needed\n"
        "3. Check element types and values AFTER length validation\n"
        "4. Match error messages EXACTLY as specified in problem\n"
        "5. Use correct exception types (TypeError vs ValueError)\n\n"
        "Example of CORRECT tuple validation pattern:\n"
        "```python\n"
        "# If tuple needs at least 2 elements:\n"
        "if not isinstance(item, tuple) or len(item) < 2:\n"
        "    raise TypeError(\"Item incomplete\")\n"
        "# Now safe to access item[0], item[1]\n"
        "```\n\n"
        "Implement the solution following this validation pattern."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Try multiple models with different temperatures for diversity
    candidates = []
    
    # First, try with temperature=0 (deterministic)
    for attempt in range(len(AGENT_MODELS)):
        try:
            resp = _call_llm(messages, run_id, attempt, temperature=0.0, timeout_s=300)
            code = _extract_main_py(resp)
            
            if code and not _validate_syntax(code):
                candidates.append(code)
                # If we got a valid solution, try one more with higher temperature for diversity
                if len(candidates) == 1:
                    try:
                        resp2 = _call_llm(messages, run_id, attempt, temperature=0.3, timeout_s=300)
                        code2 = _extract_main_py(resp2)
                        if code2 and not _validate_syntax(code2) and code2 != code:
                            candidates.append(code2)
                    except Exception:
                        pass
                break
        except Exception:
            continue
    
    # If we have candidates, return the first one
    # (could add selection logic here, but first valid is often good)
    if candidates:
        return _build_single_file_patch("main.py", candidates[0])
    
    # Fallback: try remaining models if no solution yet
    for attempt in range(len(AGENT_MODELS)):
        try:
            resp = _call_llm(messages, run_id, attempt, temperature=0.5, timeout_s=300)
            code = _extract_main_py(resp)
            
            if code and not _validate_syntax(code):
                return _build_single_file_patch("main.py", code)
        except Exception:
            continue
    
    return ""
