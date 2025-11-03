import os
import re
import ast
import json
import uuid
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


# v7 improved agent: generic code agent with enhanced robustness
# - Never embeds problem-specific constants or dataset names
# - Uses only the inference gateway exposed via INFERENCE_URL/SANDBOX_PROXY_URL
# - Returns a unified diff that replaces main.py entirely
# - Enhanced with: syntax validation, better extraction, exponential backoff, self-correction


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
    """Read file content safely."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def _smart_truncate(content: str, max_chars: int) -> str:
    """Truncate content while preserving structure at line boundaries."""
    if len(content) <= max_chars:
        return content
    
    lines = content.splitlines(keepends=True)
    truncated = []
    char_count = 0
    
    for line in lines:
        if char_count + len(line) > max_chars:
            break
        truncated.append(line)
        char_count += len(line)
    
    result = "".join(truncated)
    if char_count < len(content):
        result += "\n# ... (truncated)\n"
    return result


def _validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Validate Python syntax and return (is_valid, error_message)."""
    if not code or not code.strip():
        return False, "Empty code"
    
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Parse error: {str(e)}"


def _extract_code_from_response(response: str) -> Optional[str]:
    """Extract Python code from LLM response using multiple strategies."""
    if not response or not response.strip():
        return None
    
    # Strategy 1: Explicitly marked main.py block
    patterns = [
        # ```python\n# main.py\n...```
        r"```python\s*\n#\s*main\.py\s*\n([\s\S]*?)\n```",
        # ```python\n# File: main.py\n...```
        r"```python\s*\n#\s*[Ff]ile:\s*main\.py\s*\n([\s\S]*?)\n```",
        # ```python\n"""main.py"""\n...```
        r"```python\s*\n['\"][\"\'][\"\']main\.py['\"][\"\'][\"\']\s*\n([\s\S]*?)\n```",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        if matches and matches[0].strip():
            return matches[0].strip()
    
    # Strategy 2: First python code block
    python_blocks = re.findall(r"```python\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    for block in python_blocks:
        code = block.strip()
        if code and len(code) > 20:  # Meaningful code length
            return code
    
    # Strategy 3: Code block without language tag
    generic_blocks = re.findall(r"```\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    for block in generic_blocks:
        code = block.strip()
        # Check if it looks like Python (has common Python keywords)
        if code and any(kw in code for kw in ["def ", "class ", "import ", "from "]):
            if len(code) > 20:
                return code
    
    # Strategy 4: Extract code between markers
    if "# main.py" in response:
        idx = response.find("# main.py")
        remainder = response[idx:]
        # Find the end of code (next markdown heading or end of message)
        end_markers = ["```", "\n# ", "\n## ", "\n### "]
        end_idx = len(remainder)
        for marker in end_markers:
            pos = remainder.find(marker, 20)  # Skip past the initial marker
            if pos != -1 and pos < end_idx:
                end_idx = pos
        return remainder[:end_idx].strip()
    
    return None


def _build_unified_diff(filename: str, old_content: str, new_content: str) -> str:
    """Build a proper unified diff for filename."""
    old_lines = old_content.splitlines(keepends=False)
    new_lines = new_content.splitlines(keepends=False)
    
    header = [
        f"diff --git a/{filename} b/{filename}",
        "index 0000000..1111111 100644",
        f"--- a/{filename}",
        f"+++ b/{filename}",
        f"@@ -1,{len(old_lines) if old_lines else 1} +1,{len(new_lines) if new_lines else 1} @@",
    ]
    
    body: List[str] = []
    
    # Remove old lines
    if old_lines:
        body.extend(["-" + line for line in old_lines])
    
    # Add new lines
    if new_lines:
        body.extend(["+" + line for line in new_lines])
    
    return "\n".join(header + body) + "\n"


def _call_llm(
    messages: List[Dict[str, str]],
    run_id: str,
    attempt: int,
    timeout_s: int = 300,
    max_retries: int = 5
) -> Optional[str]:
    """Call LLM with exponential backoff retry logic."""
    url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
    headers = {"Content-Type": "application/json"}
    model = AGENT_MODELS[attempt % len(AGENT_MODELS)]
    
    body = {
        "run_id": run_id,
        "messages": messages,
        "temperature": 0.0,
        "agent_id": "agent-v7-improved",
        "model": model,
    }
    
    last_error: Optional[Exception] = None
    
    for retry in range(max_retries):
        try:
            resp = requests.post(
                url,
                json=body,
                headers=headers,
                timeout=timeout_s
            )
            resp.raise_for_status()
            
            data = resp.json()
            
            # Handle different response formats
            if isinstance(data, dict):
                if data.get("choices"):
                    content = data["choices"][0].get("message", {}).get("content")
                    if content:
                        return content
                elif data.get("content"):
                    return data["content"]
                elif data.get("response"):
                    return data["response"]
            elif isinstance(data, str):
                return data
            
            # Fallback: serialize as JSON
            return json.dumps(data)
            
        except requests.exceptions.Timeout:
            last_error = Exception(f"Timeout after {timeout_s}s")
        except requests.exceptions.RequestException as e:
            last_error = e
        except Exception as e:
            last_error = e
        
        # Exponential backoff with jitter
        if retry < max_retries - 1:
            wait_time = min(2 ** retry + (retry * 0.5), 30)
            time.sleep(wait_time)
    
    # Return None on failure to allow caller to try next model
    return None


def _build_system_prompt(mode: str) -> str:
    """Build system prompt based on mode."""
    base = (
        "You are an expert Python software engineer. "
        "Your task is to write clean, correct, and complete Python code.\n\n"
    )
    
    if mode == "tests_available":
        base += (
            "CRITICAL RULES:\n"
            "1. Do NOT modify tests.py - only modify main.py\n"
            "2. Your code must pass ALL tests in tests.py\n"
            "3. Read the tests carefully to understand requirements\n\n"
        )
    else:
        base += (
            "CRITICAL RULES:\n"
            "1. Implement the solution based on the problem statement\n"
            "2. Write clean, production-ready code\n"
            "3. Handle edge cases appropriately\n\n"
        )
    
    base += (
        "OUTPUT FORMAT (STRICT):\n"
        "Return ONLY a single Python code block with the complete main.py content.\n"
        "Format EXACTLY as:\n"
        "```python\n"
        "# main.py\n"
        "[your complete implementation here]\n"
        "```\n\n"
        "Do NOT include:\n"
        "- Explanations or prose\n"
        "- Multiple code blocks\n"
        "- Partial code or placeholders\n"
        "- Comments about what you changed\n\n"
        "Return the COMPLETE, working main.py file."
    )
    
    return base


def _build_user_prompt(problem_statement: str, repo_summary: str) -> str:
    """Build user prompt with context."""
    return (
        f"Problem Statement:\n{problem_statement}\n\n"
        f"Repository Context:\n{repo_summary}\n\n"
        "Provide the complete, corrected main.py that solves this problem."
    )


def _attempt_solution(
    problem_statement: str,
    repo_summary: str,
    mode: str,
    run_id: str,
    attempt: int,
    previous_error: Optional[str] = None
) -> Optional[str]:
    """Attempt to generate a solution, optionally incorporating previous error feedback."""
    system_msg = _build_system_prompt(mode)
    user_msg = _build_user_prompt(problem_statement, repo_summary)
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    
    # If there was a previous syntax error, add it as feedback
    if previous_error:
        messages.append({
            "role": "assistant",
            "content": "Here is my implementation:\n```python\n# (previous attempt)\n```"
        })
        messages.append({
            "role": "user",
            "content": (
                f"The previous code had a syntax error:\n{previous_error}\n\n"
                "Please provide a corrected version of main.py that fixes this error. "
                "Return the COMPLETE file in the same format."
            )
        })
    
    # Call LLM with retries
    response = _call_llm(messages, run_id, attempt, timeout_s=300, max_retries=5)
    if not response:
        return None
    
    # Extract code from response
    code = _extract_code_from_response(response)
    if not code:
        return None
    
    # Validate syntax
    is_valid, error_msg = _validate_python_syntax(code)
    if is_valid:
        return code
    
    # If syntax is invalid and this is first try, attempt one self-correction
    if previous_error is None and attempt == 0:
        return _attempt_solution(
            problem_statement,
            repo_summary,
            mode,
            run_id,
            attempt,
            previous_error=error_msg
        )
    
    return None


def agent_main(
    input_dict: Dict[str, Any],
    repo_dir: str = "repo",
    test_mode: bool = False
) -> str:
    """Entry point required by the evaluation harness.

    Returns a unified diff patch that fully replaces main.py.
    
    Args:
        input_dict: Dictionary containing problem_statement, run_id, etc.
        repo_dir: Directory containing the repository to modify
        test_mode: Whether running in test mode (unused but required)
    
    Returns:
        Unified diff string or empty string on failure
    """
    # Extract run_id
    run_id = (input_dict or {}).get("run_id") or os.getenv("RUN_ID") or str(uuid.uuid4())
    
    # Change to repo directory if it exists
    if repo_dir and os.path.exists(repo_dir):
        try:
            os.chdir(repo_dir)
        except Exception:
            pass
    
    # Extract problem statement
    problem_statement = (input_dict or {}).get("problem_statement", "") or ""
    if not problem_statement:
        return ""
    
    # Determine mode
    mode = (input_dict or {}).get("problem_category")
    if mode not in ("spec_only", "tests_available"):
        mode = "tests_available" if os.path.exists("tests.py") else "spec_only"
    
    # Build repository context summary
    repo_parts: List[str] = []
    
    # Include main.py if it exists
    main_content = _read("main.py")
    if main_content:
        truncated_main = _smart_truncate(main_content, 10000)
        repo_parts.append(f"### Current main.py\n```python\n{truncated_main}\n```")
    
    # Include tests.py if in tests_available mode
    if mode == "tests_available":
        tests_content = _read("tests.py")
        if tests_content:
            truncated_tests = _smart_truncate(tests_content, 12000)
            repo_parts.append(f"### tests.py\n```python\n{truncated_tests}\n```")
    
    # Include other relevant files (README, requirements, etc.)
    for filename in ["README.md", "requirements.txt", "setup.py"]:
        content = _read(filename)
        if content:
            truncated = _smart_truncate(content, 2000)
            repo_parts.append(f"### {filename}\n```\n{truncated}\n```")
    
    repo_summary = "\n\n".join(repo_parts)
    
    # Truncate problem statement if too long
    truncated_problem = _smart_truncate(problem_statement, 15000)
    
    # Try multiple models until we get a valid solution
    for attempt in range(len(AGENT_MODELS) * 2):  # Try each model twice
        try:
            code = _attempt_solution(
                truncated_problem,
                repo_summary,
                mode,
                run_id,
                attempt
            )
            
            if code:
                # Generate unified diff
                old_content = main_content
                patch = _build_unified_diff("main.py", old_content, code)
                return patch
                
        except Exception:
            # Continue to next attempt on any error
            continue
    
    # If all attempts fail, return empty string
    return ""
