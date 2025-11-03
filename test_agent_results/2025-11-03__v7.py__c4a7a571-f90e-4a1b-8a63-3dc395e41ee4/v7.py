import os
import re
import ast
import json
import uuid
import time
from typing import Any, Dict, List, Tuple

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


def _validate_syntax(code: str) -> Tuple[bool, str]:
    """
    Check if code has valid Python syntax.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


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
    """Extract Python code from LLM response using multiple strategies."""
    if not response:
        return ""
    
    # Strategy 1: Explicitly headed block with main.py comment
    m = re.findall(r"```python\s*\n#\s*main\.py\n([\s\S]*?)\n```", response, re.DOTALL)
    if m and m[0].strip():
        return m[0].strip()
    
    # Strategy 2: Any python code block
    m2 = re.findall(r"```python\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    if m2:
        for block in m2:
            if block.strip():
                return block.strip()
    
    # Strategy 3: Code block without language specifier
    m3 = re.findall(r"```\n([\s\S]*?)\n```", response, re.DOTALL)
    if m3:
        for block in m3:
            # Check if it looks like Python code (has def, class, or import)
            if block.strip() and any(keyword in block for keyword in ['def ', 'class ', 'import ']):
                return block.strip()
    
    return ""


def _call_llm(
    messages: List[Dict[str, str]], 
    run_id: str, 
    model_idx: int,
    temperature: float = 0.0
) -> str:
    """Call inference gateway with specified model and parameters."""
    url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
    headers = {"Content-Type": "application/json"}
    model = AGENT_MODELS[model_idx % len(AGENT_MODELS)]
    body = {
        "run_id": run_id,
        "messages": messages,
        "temperature": temperature,
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
    
    Args:
        input_dict: Dictionary containing problem_statement, run_id, and other metadata
        repo_dir: Directory containing the problem files (main.py, tests.py, etc.)
        test_mode: Whether running in test mode (unused, for compatibility)
    
    Returns:
        Unified diff patch string for main.py, or empty string on failure
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

    # Build repository context with larger context window
    parts: List[str] = []
    for name in ("main.py", "tests.py"):
        content = _read(name)
        if content:
            # Increased from 8000 to 15000 for better context
            parts.append(f"### {name}\n```python\n{content[:15000]}\n```")
    repo_summary = "\n\n".join(parts)

    system_msg = (
        "You are an expert Python engineer. Your task is to write production-quality, "
        "bug-free Python code that correctly implements the given specification.\n"
        + ("IMPORTANT: Only modify main.py. Do not change tests.py.\n" if mode == "tests_available" else "")
        + "Return your solution in this exact format:\n"
        "```python\n# main.py\n<your complete implementation here>\n```"
    )
    
    user_msg = f"""# Problem Statement

{problem_statement[:15000]}

# Repository Files

{repo_summary}

# Implementation Guidelines

Write a complete, correct implementation following these critical rules:

1. **Loop Safety**: In while loops with continue, ensure the loop variable advances before continue!
   - Wrong: `while i < n: if cond: continue` (infinite loop - i never increments)
   - Correct: `while i < n: if cond: i += 1; continue`

2. **Return Types**: For functions returning list[str], each element is a single line, not multiple lines joined with \\n

3. **Edge Cases**: Always handle:
   - Empty inputs (empty strings, empty lists)
   - First and last elements in sequences
   - Single-element collections
   - Boundary conditions

4. **Validation**: For tuple/list validation, use appropriate length checks:
   - Use `len(item) < 2` to catch both empty and single-element cases
   - Not `len(item) < 1` which only catches empty

5. **Exceptions**: Match error messages and exception types exactly as specified in the problem statement

6. **Imports**: Include all necessary imports at the top of the file

7. **Logic**: Ensure your implementation handles all cases described in the problem statement

Provide your complete implementation now."""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Try each model with iterative refinement
    for model_idx in range(len(AGENT_MODELS)):
        try:
            # First attempt
            response = _call_llm(messages, run_id, model_idx)
            code = _extract_main_py(response)
            
            if not code:
                continue
            
            # Validate syntax
            is_valid, error_msg = _validate_syntax(code)
            
            if is_valid:
                # Success - return the patch
                return _build_single_file_patch("main.py", code)
            
            # If syntax error, try one refinement attempt with this model
            refinement_msg = {
                "role": "user",
                "content": f"""The code has a syntax error:

{error_msg}

Please fix the error and provide the corrected complete implementation of main.py.
Return it in the same format:
```python
# main.py
<corrected implementation>
```"""
            }
            
            refined_messages = messages + [
                {"role": "assistant", "content": response},
                refinement_msg
            ]
            
            refined_response = _call_llm(refined_messages, run_id, model_idx)
            refined_code = _extract_main_py(refined_response)
            
            if refined_code:
                is_valid_refined, _ = _validate_syntax(refined_code)
                if is_valid_refined:
                    return _build_single_file_patch("main.py", refined_code)
                    
        except Exception:
            # Move to next model on exception
            continue
    
    # All models failed
    return ""
