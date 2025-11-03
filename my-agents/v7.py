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
# - Includes multi-iteration self-review and syntax validation


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
    """
    Validate Python syntax of the given code.
    Returns None if valid, or an error message if invalid.
    """
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


def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False) -> str:
    """
    Entry point required by the evaluation harness.
    
    Generates a complete solution with multi-iteration self-review:
    1. Generate initial solution
    2. Validate syntax
    3. Self-review and refine multiple times
    4. Return unified diff patch that fully replaces main.py
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

    # Initial system message for solution generation
    system_msg = (
        "You are a senior Python engineer who writes correct, robust code.\n"
        + ("Do not modify tests.py; only change main.py.\n" if mode == "tests_available" else "")
        + "Return ONLY one code block containing the complete main.py with a '# main.py' header.\n"
        "Format exactly as:\n```python\n# main.py\n[complete code]\n```\n"
        "No prose. Write careful, deterministic code that handles all edge cases correctly."
    )
    
    user_msg = (
        f"Problem Statement:\n{problem_statement[:12000]}\n\n"
        f"Repository Summary:\n{summary}\n\n"
        "Implement a complete, correct solution. Handle all edge cases, error conditions, and boundary scenarios."
    )

    # Try multiple models to generate initial solution
    best_code = ""
    best_attempt = 0
    
    for attempt in range(len(AGENT_MODELS)):
        try:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            resp = _call_llm(messages, run_id, attempt, 300)
            main_src = _extract_main_py(resp)
            
            if not main_src:
                continue
                
            # Validate syntax
            syntax_err = _validate_syntax(main_src)
            if syntax_err:
                continue  # Try next model
                
            best_code = main_src
            best_attempt = attempt
            break
            
        except Exception:
            continue

    if not best_code:
        return ""

    # Multi-iteration self-review: perform 2 review passes to catch subtle bugs
    for iteration in range(2):
        try:
            # Alternate between different review strategies
            if iteration == 0:
                # First pass: deep logic analysis
                review_system = (
                    "You are an expert code reviewer. Analyze the code for correctness.\n"
                    "Check: logic errors, off-by-one bugs, state management, boundary conditions, algorithm correctness.\n"
                    "If correct, respond ONLY with: APPROVED\n"
                    "If there are bugs, provide corrected code:\n```python\n# main.py\n[corrected code]\n```"
                )
                review_user = (
                    f"Problem:\n{problem_statement[:8000]}\n\n"
                    f"Code to review:\n```python\n{best_code}\n```\n\n"
                    "Trace through the logic step-by-step. Check if it correctly handles:\n"
                    "- All examples in the problem statement\n"
                    "- Edge cases (empty input, single element, maximum size)\n"
                    "- Boundary conditions and state transitions\n"
                    "Are there any logic errors or incorrect assumptions?"
                )
            else:
                # Second pass: focus on missed edge cases and algorithm validation
                review_system = (
                    "You are a thorough code auditor. Your job is to find remaining bugs.\n"
                    "Focus on: algorithm correctness, hidden edge cases, incorrect data structure usage.\n"
                    "If the code is now correct, respond ONLY with: APPROVED\n"
                    "If bugs remain, provide fixed code:\n```python\n# main.py\n[fixed code]\n```"
                )
                review_user = (
                    f"Problem:\n{problem_statement[:6000]}\n\n"
                    f"Current code:\n```python\n{best_code}\n```\n\n"
                    "This code passed initial review. Do a final check:\n"
                    "- Validate the algorithm's correctness against problem requirements\n"
                    "- Check for subtle bugs in loops, recursion, or data structure operations\n"
                    "- Verify all edge cases are handled properly\n"
                    "Any remaining issues?"
                )
            
            review_messages = [
                {"role": "system", "content": review_system},
                {"role": "user", "content": review_user},
            ]
            
            # Use different model for diversity in second iteration
            review_attempt = (best_attempt + iteration + 1) % len(AGENT_MODELS)
            review_resp = _call_llm(review_messages, run_id, review_attempt, 300)
            
            # Check if reviewer approved or provided corrections
            if "APPROVED" in review_resp.upper():
                # Code passed this review iteration
                break
            
            refined_code = _extract_main_py(review_resp)
            if refined_code:
                # Validate refined code syntax
                syntax_err = _validate_syntax(refined_code)
                if not syntax_err:
                    best_code = refined_code
                    # Continue to next review iteration
        
        except Exception:
            # If review fails, continue with current best_code
            pass

    return _build_single_file_patch("main.py", best_code)
