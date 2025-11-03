import ast
import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import requests


# v7 agent: generic Python code agent that proposes a full-file patch for main.py.
# - Never embeds problem-specific constants or dataset names
# - Uses only the inference gateway exposed via INFERENCE_URL/SANDBOX_PROXY_URL
# - Returns a unified diff that replaces main.py entirely
# - Validates syntax before returning patches


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
    """Check if Python code is syntactically valid."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _extract_python_code(response: str) -> Optional[str]:
    """
    Extract Python code from LLM response using multiple strategies.
    Returns the first valid code block found, or None if extraction fails.
    """
    if not response:
        return None
    
    # Strategy 1: Look for code block with explicit "# main.py" header
    pattern1 = r"```python\s*\n#\s*main\.py\s*\n([\s\S]*?)```"
    matches = re.findall(pattern1, response, re.DOTALL)
    for match in matches:
        code = match.strip()
        if code and _validate_syntax(code):
            return code
    
    # Strategy 2: Look for any Python code block
    pattern2 = r"```python\s*\n([\s\S]*?)```"
    matches = re.findall(pattern2, response, re.DOTALL)
    for match in matches:
        code = match.strip()
        if code and _validate_syntax(code):
            return code
    
    # Strategy 3: Look for code block without language tag
    pattern3 = r"```\s*\n([\s\S]*?)```"
    matches = re.findall(pattern3, response, re.DOTALL)
    for match in matches:
        code = match.strip()
        # Only accept if it looks like Python and has valid syntax
        if code and ("def " in code or "class " in code or "import " in code):
            if _validate_syntax(code):
                return code
    
    return None


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


def _call_llm(
    messages: List[Dict[str, str]], 
    run_id: str, 
    model_name: str,
    timeout_s: int = 300
) -> str:
    """
    Call the inference gateway with the given messages and model.
    Returns the LLM response text or raises an exception on failure.
    """
    url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
    headers = {"Content-Type": "application/json"}
    body = {
        "run_id": run_id,
        "messages": messages,
        "temperature": 0.0,
        "agent_id": "agent-v7",
        "model": model_name,
    }
    
    # Retry up to 3 times with exponential backoff
    last_err: Optional[Exception] = None
    for retry in range(3):
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            
            # Handle different response formats
            if isinstance(data, dict) and data.get("choices"):
                content = (data["choices"][0].get("message", {}) or {}).get("content")
                return content or ""
            if isinstance(data, str):
                return data
            return json.dumps(data)
        except Exception as e:
            last_err = e
            time.sleep(1 + retry)
    
    raise last_err if last_err else RuntimeError("LLM call failed")


def _build_prompt(problem_statement: str, main_py: str, tests_py: str, mode: str) -> List[Dict[str, str]]:
    """
    Build the prompt messages for the LLM.
    Emphasizes implementing functions from the skeleton and passing tests.
    """
    # Build context summary
    parts: List[str] = []
    if main_py:
        parts.append(f"### main.py (current skeleton)\n```python\n{main_py[:8000]}\n```")
    if tests_py and mode == "tests_available":
        parts.append(f"### tests.py (reference - DO NOT MODIFY)\n```python\n{tests_py[:8000]}\n```")
    
    summary = "\n\n".join(parts) if parts else "No existing files found."
    
    system_msg = (
        "You are a senior Python engineer tasked with implementing production-quality code.\n\n"
        "CRITICAL REQUIREMENTS:\n"
        "1. Study the main.py skeleton carefully - implement ALL functions/classes with correct signatures\n"
        "2. Return ONLY a complete, syntactically valid main.py file\n"
        "3. Format your response as: ```python\\n# main.py\\n[complete implementation]\\n```\n"
        "4. Include all necessary imports at the top\n"
        "5. Write clean, efficient, well-structured code\n"
        "6. Ensure your implementation handles all edge cases\n"
    )
    
    if mode == "tests_available":
        system_msg += "7. Do NOT modify tests.py - only implement main.py\n"
    
    system_msg += "\nReturn ONLY the code block - no explanations, no prose."
    
    user_msg = (
        f"## Problem Statement\n{problem_statement[:12000]}\n\n"
        f"## Repository Context\n{summary}\n\n"
        f"## Task\n"
        f"Implement a complete, working solution in main.py that satisfies all requirements."
    )
    
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False) -> str:
    """
    Entry point required by the evaluation harness.
    
    Args:
        input_dict: Contains problem_statement, run_id, and optional problem_category
        repo_dir: Directory containing the problem files (main.py, tests.py, etc.)
        test_mode: Unused, for compatibility
    
    Returns:
        A unified diff patch that fully replaces main.py
    """
    run_id = (input_dict or {}).get("run_id", os.getenv("RUN_ID", str(uuid.uuid4())))
    
    # Change to repo directory
    if repo_dir and os.path.exists(repo_dir):
        try:
            os.chdir(repo_dir)
        except Exception:
            pass
    
    # Extract problem statement
    problem_statement = (input_dict or {}).get("problem_statement", "") or ""
    
    # Detect mode: spec_only vs tests_available
    mode = (input_dict or {}).get("problem_category", None)
    if mode not in ("spec_only", "tests_available"):
        mode = "tests_available" if os.path.exists("tests.py") else "spec_only"
    
    # Read existing files
    main_py = _read("main.py")
    tests_py = _read("tests.py") if mode == "tests_available" else ""
    
    # Build prompt
    messages = _build_prompt(problem_statement, main_py, tests_py, mode)
    
    # Try each model with multiple attempts
    max_attempts_per_model = 2
    for model_idx, model_name in enumerate(AGENT_MODELS):
        for attempt in range(max_attempts_per_model):
            try:
                # Call LLM
                response = _call_llm(messages, run_id, model_name, timeout_s=300)
                
                # Extract and validate code
                code = _extract_python_code(response)
                if code:
                    # Return patch immediately upon success
                    return _build_single_file_patch("main.py", code)
                
            except Exception:
                # Continue to next attempt/model on any error
                time.sleep(0.5)
                continue
    
    # All attempts failed - return empty patch
    return ""
