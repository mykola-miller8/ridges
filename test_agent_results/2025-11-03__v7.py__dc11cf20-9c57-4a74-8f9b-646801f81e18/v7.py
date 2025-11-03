import os
import re
import ast
import json
import uuid
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


# v7 seed agent: generic code agent that proposes a full-file patch for main.py only.
# - Never embeds problem-specific constants or dataset names
# - Uses only the inference gateway exposed via INFERENCE_URL/SANDBOX_PROXY_URL
# - Returns a unified diff that replaces main.py entirely
# - Proactive bug prevention with explicit validation guidance


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


def _call_llm(messages: List[Dict[str, str]], run_id: str, attempt: int, timeout_s: int = 240) -> str:
    """Call the inference gateway with retry logic."""
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
            time.sleep(1 + r)
    raise last_err if last_err else RuntimeError("LLM call failed")


def _review_code(code: str, problem_statement: str, run_id: str, attempt: int) -> Tuple[bool, str]:
    """
    Review code focusing on validation logic correctness.
    Returns (approved, refined_code).
    """
    review_system = (
        "You are a code reviewer specializing in finding validation bugs.\n"
        "Test the code mentally with edge cases, especially incomplete/malformed inputs.\n\n"
        "If CORRECT, respond: APPROVED\n"
        "If buggy, provide FIXED code:\n```python\n# main.py\n[fixed code]\n```"
    )
    
    review_user = (
        f"Problem:\n{problem_statement[:7000]}\n\n"
        f"Code:\n```python\n{code}\n```\n\n"
        "Check validation logic carefully:\n"
        "- Does it catch empty AND incomplete inputs (e.g., tuple with 1 element when 3 needed)?\n"
        "- Are length checks correct? (< 2 not < 1 if minimum is 2 elements)\n"
        "- Are error messages exactly as specified?\n"
        "- Do all edge cases work: empty, single item, incomplete, malformed?\n\n"
        "Is this code correct?"
    )
    
    review_messages = [
        {"role": "system", "content": review_system},
        {"role": "user", "content": review_user},
    ]
    
    try:
        review_resp = _call_llm(review_messages, run_id, attempt, 300)
        
        if "APPROVED" in review_resp.upper():
            return True, code
        
        refined_code = _extract_main_py(review_resp)
        if refined_code and not _validate_syntax(refined_code):
            return False, refined_code
    except Exception:
        pass
    
    return True, code


def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False) -> str:
    """
    Entry point required by the evaluation harness.
    
    Strategy: Prevent bugs at generation time with explicit guidance.
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

    # Generation prompt with explicit validation guidance
    system_msg = (
        "You are a senior Python engineer who writes SIMPLE, CORRECT, BUG-FREE code.\n"
        + ("Do not modify tests.py; only change main.py.\n" if mode == "tests_available" else "")
        + "Return ONLY:\n```python\n# main.py\n[complete code]\n```\n"
        "No prose."
    )
    
    user_msg = (
        f"Problem:\n{problem_statement[:12000]}\n\n"
        f"Repository:\n{summary}\n\n"
        "Implement a complete solution. CRITICAL VALIDATION GUIDANCE:\n\n"
        "When validating input structures (lists, tuples, etc.), avoid these common bugs:\n"
        "? BAD: if len(item) < 1  # Only catches empty, not incomplete (e.g., 1 element when 3 needed)\n"
        "? GOOD: if len(item) < 3  # Catches both empty AND incomplete\n\n"
        "? BAD: Checking length after accessing elements\n"
        "? GOOD: Check length FIRST, then access elements\n\n"
        "? BAD: Generic length check for all types\n"
        "? GOOD: Type-specific validation (each type needs different length)\n\n"
        "For input validation:\n"
        "1. Check if input exists and has correct type FIRST\n"
        "2. Check if it has MINIMUM required elements (not just > 0)\n"
        "3. Validate each element's type and value\n"
        "4. Use exact error messages as specified in problem\n\n"
        "Test your logic mentally with:\n"
        "- Empty input: [], (), None\n"
        "- Incomplete: tuple with 1 element when 3 needed\n"
        "- Malformed: wrong types\n"
        "- Valid: correct structure\n\n"
        "Implement the solution now."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    best_code = ""
    
    # Try multiple models with review
    for attempt in range(min(3, len(AGENT_MODELS))):
        try:
            # Generate solution
            resp = _call_llm(messages, run_id, attempt, 300)
            main_src = _extract_main_py(resp)
            
            if not main_src or _validate_syntax(main_src):
                continue
            
            # Review the generated code
            approved, reviewed_code = _review_code(main_src, problem_statement, run_id, attempt)
            
            if not approved and reviewed_code != main_src:
                # Code was refined, use the refined version
                best_code = reviewed_code
            else:
                best_code = reviewed_code
            
            if best_code:
                break
                
        except Exception:
            continue

    # Fallback if no solution yet
    if not best_code:
        for attempt in range(3, len(AGENT_MODELS)):
            try:
                resp = _call_llm(messages, run_id, attempt, 300)
                main_src = _extract_main_py(resp)
                
                if main_src and not _validate_syntax(main_src):
                    best_code = main_src
                    break
            except Exception:
                continue

    return _build_single_file_patch("main.py", best_code) if best_code else ""
