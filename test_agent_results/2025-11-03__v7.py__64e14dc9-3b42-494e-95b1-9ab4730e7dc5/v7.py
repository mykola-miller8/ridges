"""
Generic Python code-solving agent with validation and self-review.

Entry point: agent_main(input_dict, repo_dir='repo', test_mode=False) -> str
Returns a unified diff patch for main.py.
"""

import ast
import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

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
    """Read file contents, return empty string on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _validate_syntax(code: str) -> Tuple[bool, str]:
    """
    Validate Python syntax using ast.parse.
    
    Returns:
        (is_valid, error_message)
    """
    if not code.strip():
        return False, "Empty code"
    
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Parse error: {str(e)}"


def _extract_code_blocks(response: str) -> List[str]:
    """Extract all Python code blocks from response, preferring those with # main.py header."""
    if not response:
        return []
    
    # First try to find blocks with # main.py header
    pattern = r"```python\s*\n#\s*main\.py\s*\n([\s\S]*?)\n```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return [m.strip() for m in matches if m.strip()]
    
    # Fallback: all python blocks
    pattern = r"```python\s*\n([\s\S]*?)\n```"
    matches = re.findall(pattern, response, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]


def _build_single_file_patch(filename: str, new_content: str) -> str:
    """Build a full-file unified diff for filename."""
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
    attempt: int,
    timeout_s: int = 300
) -> str:
    """Call inference gateway with retry logic."""
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
    
    last_err: Optional[Exception] = None
    for retry in range(3):
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
            time.sleep(1 + retry)
    
    raise last_err if last_err else RuntimeError("LLM call failed")


def agent_main(
    input_dict: Dict[str, Any],
    repo_dir: str = "repo",
    test_mode: bool = False
) -> str:
    """
    Entry point for the evaluation harness.
    
    Args:
        input_dict: Contains problem_statement, run_id, etc.
        repo_dir: Working directory containing main.py and instruction.md
        test_mode: Whether running in test mode (unused)
    
    Returns:
        Unified diff patch for main.py
    """
    run_id = (input_dict or {}).get("run_id", os.getenv("RUN_ID", str(uuid.uuid4())))

    if repo_dir and os.path.exists(repo_dir):
        try:
            os.chdir(repo_dir)
        except Exception:
            pass

    problem_statement = (input_dict or {}).get("problem_statement", "") or ""

    # Build context from available files (main.py only - tests.py not available at runtime)
    main_py_content = _read("main.py")
    repo_summary = ""
    if main_py_content:
        repo_summary = f"### main.py (skeleton)\n```python\n{main_py_content[:10000]}\n```"

    # Generic system prompt for diverse problem types
    system_msg = (
        "You are an expert Python engineer who writes correct, production-ready code.\n\n"
        "APPROACH:\n"
        "1. Read the problem statement carefully, understanding:\n"
        "   - The core requirements and functionality needed\n"
        "   - Input/output specifications and formats\n"
        "   - All examples provided (what they demonstrate)\n"
        "   - Edge cases and validation rules\n"
        "   - Any exceptions that must be raised\n\n"
        "2. Analyze the problem type and choose appropriate approach:\n"
        "   - State management: Track state across method calls correctly\n"
        "   - Algorithms: Choose efficient data structures (lists, dicts, sets, deques)\n"
        "   - Parsing: Handle input format precisely (whitespace, newlines, delimiters)\n"
        "   - Validation: Check inputs and raise exceptions with meaningful messages\n"
        "   - Business logic: Implement all rules and special cases exactly as specified\n\n"
        "3. Trace through examples mentally:\n"
        "   - For EACH example, walk through your logic step-by-step\n"
        "   - Verify the output matches expectations\n"
        "   - Check edge cases (empty inputs, boundaries, special states)\n\n"
        "4. Implementation requirements:\n"
        "   - Implement ALL methods completely (never leave empty or with just 'pass')\n"
        "   - Initialize state properly in __init__ methods\n"
        "   - Handle all validation (raise exceptions with messages as required)\n"
        "   - Parse inputs exactly as specified\n"
        "   - Implement all business rules and special cases\n"
        "   - Use appropriate data structures for efficiency\n"
        "   - Handle edge cases (empty, boundary, invalid inputs)\n\n"
        "COMMON PATTERNS:\n"
        "- Classes with state: Initialize instance variables in __init__, update in methods\n"
        "- Validation: Check constraints and raise ValueError/Exception with clear messages\n"
        "- Parsing: Use split(), strip(), int() carefully for the exact format\n"
        "- Algorithms: Consider BFS/DFS for graphs, dynamic programming for optimization\n"
        "- Edge cases: Test mentally with empty, single element, maximum, invalid inputs\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a single Python code block with the complete main.py.\n"
        "Start with '# main.py' as the first line.\n"
        "Format: ```python\\n# main.py\\n[complete code]\\n```\n"
        "No explanations or prose."
    )

    user_msg = (
        f"# Problem Statement\n{problem_statement[:15000]}\n\n"
        f"# Current Repository\n{repo_summary}\n\n"
        "Implement a complete, correct solution that:\n"
        "1. Implements all methods fully (no empty bodies)\n"
        "2. Initializes and manages state correctly\n"
        "3. Handles all validation and raises exceptions as specified\n"
        "4. Parses inputs in the exact format required\n"
        "5. Implements all business logic and special cases\n"
        "6. Works correctly for all examples and edge cases"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Try generate-review-refine cycle with multiple models
    max_attempts = len(AGENT_MODELS) * 2
    
    for attempt in range(max_attempts):
        try:
            # Step 1: Generate initial code
            response = _call_llm(messages, run_id, attempt, 300)
            code_blocks = _extract_code_blocks(response)
            
            for code in code_blocks:
                # Validate syntax
                is_valid, error_msg = _validate_syntax(code)
                
                if not is_valid:
                    # Syntax error - add feedback and retry
                    if attempt < max_attempts - 1:
                        messages.append({"role": "assistant", "content": response})
                        messages.append({
                            "role": "user",
                            "content": (
                                f"Syntax error: {error_msg}\n\n"
                                "Fix the syntax and return corrected code.\n"
                                "Format: ```python\\n# main.py\\n[corrected code]\\n```"
                            )
                        })
                        break
                    continue
                
                # Check for incomplete implementations (empty methods)
                try:
                    tree = ast.parse(code)
                    has_empty_methods = False
                    empty_method_names: List[str] = []
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            # Check if function body is just 'pass' or empty
                            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                                has_empty_methods = True
                                empty_method_names.append(node.name)
                    
                    if has_empty_methods:
                        if attempt < max_attempts - 1:
                            messages.append({"role": "assistant", "content": response})
                            methods_str = ", ".join(empty_method_names[:5])
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"The code has empty method bodies: {methods_str}\n"
                                    "Implement ALL methods with complete logic based on the problem requirements.\n"
                                    "Format: ```python\\n# main.py\\n[complete implementation]\\n```"
                                )
                            })
                            break
                        continue
                except Exception:
                    pass
                
                # Step 2: Self-review with emphasis on correctness
                if attempt < len(AGENT_MODELS):
                    review_messages = [
                        {"role": "system", "content": (
                            "You are a meticulous code reviewer who verifies correctness."
                        )},
                        {"role": "user", "content": (
                            f"# Problem\n{problem_statement[:15000]}\n\n"
                            f"# Code\n```python\n{code}\n```\n\n"
                            "Review this code by checking:\n"
                            "1. All methods are fully implemented (no empty bodies)\n"
                            "2. State is initialized and managed correctly\n"
                            "3. Input parsing handles the exact format specified\n"
                            "4. All validation rules are implemented (exceptions raised as needed)\n"
                            "5. Business logic matches all requirements and special cases\n"
                            "6. Examples work correctly:\n"
                            "   - Trace through at least one simple example\n"
                            "   - Trace through at least one complex example\n"
                            "   - Does the code produce correct results?\n"
                            "7. Edge cases are handled (empty, boundary, invalid inputs)\n\n"
                            "Reply 'APPROVED' if correct for all cases, or list specific issues found."
                        )}
                    ]
                    
                    try:
                        review_response = _call_llm(review_messages, run_id, attempt, 120)
                        
                        if review_response and "APPROVED" not in review_response.upper():
                            if attempt < max_attempts - 1:
                                messages.append({"role": "assistant", "content": response})
                                messages.append({
                                    "role": "user",
                                    "content": (
                                        f"Review found issues:\n{review_response}\n\n"
                                        "Fix all issues. Ensure the code:\n"
                                        "- Implements all logic completely\n"
                                        "- Handles all examples and edge cases correctly\n"
                                        "- Follows all requirements and special rules\n"
                                        "Return corrected code.\n"
                                        "Format: ```python\\n# main.py\\n[corrected code]\\n```"
                                    )
                                })
                                break
                    except Exception:
                        pass
                
                # Code passed validation and review
                return _build_single_file_patch("main.py", code)
            
        except Exception:
            continue

    # All attempts exhausted
    return ""
