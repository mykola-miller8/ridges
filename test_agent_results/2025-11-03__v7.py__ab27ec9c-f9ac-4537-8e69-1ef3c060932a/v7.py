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
        "1. Read the problem statement THOROUGHLY AND COMPLETELY:\n"
        "   - Core requirements and ALL functionality needed\n"
        "   - Input/output specifications and formats\n"
        "   - ALL examples (understand what each demonstrates)\n"
        "   - ALL validation rules and constraints (simple AND complex)\n"
        "   - EVERY exception/error condition that must be raised\n"
        "   - Special sections like 'Exception messages' or 'Error handling'\n"
        "   - Final/boundary cases with special rules (last frame, final round, end state)\n"
        "   - Edge cases and boundary conditions\n"
        "   - Interactions between rules (when rule A AND rule B both apply)\n\n"
        "2. Analyze problem type and design your solution:\n"
        "   - State management: Plan what state to track, how to initialize, how to update\n"
        "   - Validation first: List EVERY error condition (simple AND complex combinations)\n"
        "   - Data structures: Choose appropriate types (list, dict, set, deque, etc.)\n"
        "   - Algorithms: Select efficient approaches for the problem type\n"
        "   - Business rules: List ALL special cases, especially final/boundary states\n\n"
        "3. Implement with COMPLETENESS:\n"
        "   - Initialize ALL instance variables in __init__\n"
        "   - Implement EVERY method fully (never leave empty or with just 'pass')\n"
        "   - Add validation at the START of methods that receive input\n"
        "   - Raise exceptions with MEANINGFUL MESSAGES (not just 'Exception()')\n"
        "   - Handle ALL edge cases (empty, boundary, invalid, out-of-order)\n"
        "   - Implement ALL business logic including special rules\n"
        "   - Pay extra attention to final/last/boundary states (often have unique rules)\n\n"
        "4. CRITICAL - Validation and Exception handling:\n"
        "   - Read problem for ALL scenarios requiring exceptions\n"
        "   - Identify BOTH simple validation (range, type) AND complex validation (state combinations)\n"
        "   - Raise appropriate exception types (ValueError, IndexError, Exception, etc.)\n"
        "   - ALWAYS include descriptive message: raise ValueError(\"clear message\")\n"
        "   - Validate BEFORE processing (fail fast with clear errors)\n"
        "   - Check boundaries, ranges, state validity, input constraints\n"
        "   - For complex validation: consider current state + input + position\n"
        "   - Special validation for final/boundary states (often stricter rules)\n\n"
        "5. Verify mentally:\n"
        "   - Trace through examples step-by-step\n"
        "   - Test edge cases mentally (empty, boundary, invalid)\n"
        "   - Test final/boundary state cases (last item, end condition, special rules)\n"
        "   - Verify ALL exceptions are raised correctly (simple AND complex cases)\n"
        "   - Check state updates happen correctly\n\n"
        "COMMON PATTERNS:\n"
        "- State tracking: Use instance variables, update in each method call\n"
        "- Validation: Check at method start, raise with message immediately\n"
        "- Simple validation: Range checks, type checks, null checks\n"
        "- Complex validation: State-dependent checks (\"if in final state, then different rules\")\n"
        "- Boundary/final state: Often has special rules (last frame, final round, end game)\n"
        "- Complete logic: Implement all rules, cases, and special scenarios\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a single Python code block with the complete main.py.\n"
        "Start with '# main.py' as the first line.\n"
        "Format: ```python\\n# main.py\\n[complete code]\\n```\n"
        "No explanations or prose."
    )

    user_msg = (
        f"# Problem Statement\n{problem_statement[:15000]}\n\n"
        f"# Current Repository\n{repo_summary}\n\n"
        "Implement a complete, correct solution:\n\n"
        "CRITICAL REQUIREMENTS:\n"
        "1. Implement ALL methods with complete logic (no empty bodies)\n"
        "2. Initialize ALL state variables in __init__\n"
        "3. Read problem for ALL validation rules - both SIMPLE and COMPLEX:\n"
        "   - Simple: range checks, type checks, basic constraints\n"
        "   - Complex: state-dependent validation, combinations of conditions\n"
        "   - Final/boundary state validation (last frame, end game, special rules)\n"
        "4. Raise exceptions with MEANINGFUL MESSAGES for EVERY error condition\n"
        "5. Implement ALL business logic, rules, and special cases\n"
        "6. Pay special attention to final/boundary states (often have unique validation)\n"
        "7. Handle ALL edge cases (empty, boundary, invalid inputs, state errors)\n"
        "8. Parse inputs exactly as specified\n"
        "9. Verify your solution works for all provided examples"
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
                
                # Check for incomplete implementations and missing validation
                try:
                    tree = ast.parse(code)
                    has_empty_methods = False
                    empty_method_names: List[str] = []
                    has_raise_statements = False
                    methods_with_params: List[str] = []
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            # Check if function body is just 'pass' or empty
                            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                                has_empty_methods = True
                                empty_method_names.append(node.name)
                            
                            # Track methods with parameters (likely need validation)
                            if node.name != "__init__" and len(node.args.args) > 1:  # More than just 'self'
                                methods_with_params.append(node.name)
                        
                        # Check if code has any exception raising
                        if isinstance(node, ast.Raise):
                            has_raise_statements = True
                    
                    # Flag if empty methods found
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
                    
                    # Flag if validation likely missing (has methods with params but no raises)
                    # Only check if problem mentions "exception" or "error" or "raise"
                    problem_lower = problem_statement.lower()
                    needs_validation = any(keyword in problem_lower for keyword in 
                                          ["exception", "error", "raise", "invalid", "cannot"])
                    
                    if needs_validation and methods_with_params and not has_raise_statements:
                        if attempt < len(AGENT_MODELS):  # Only on first pass
                            messages.append({"role": "assistant", "content": response})
                            messages.append({
                                "role": "user",
                                "content": (
                                    "The problem requires exception handling but no validation is implemented.\n"
                                    "Read the problem for ALL error conditions and validation requirements.\n"
                                    "Add proper validation that raises exceptions with meaningful messages.\n"
                                    "Format: ```python\\n# main.py\\n[complete implementation with validation]\\n```"
                                )
                            })
                            break
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
                            "Review this code thoroughly:\n\n"
                            "1. Implementation completeness:\n"
                            "   - Are ALL methods fully implemented (no empty bodies)?\n"
                            "   - Are ALL instance variables initialized in __init__?\n\n"
                            "2. Exception handling (CRITICAL - check CAREFULLY):\n"
                            "   - Read problem for ALL error/exception requirements\n"
                            "   - Simple validation: Are range, type, basic checks implemented?\n"
                            "   - Complex validation: Are state-dependent validations implemented?\n"
                            "   - Final/boundary validation: Do final states have special validation rules?\n"
                            "   - Are exceptions raised with MEANINGFUL messages?\n"
                            "   - Are ALL boundary/constraint checks implemented?\n"
                            "   - Test mentally: What if invalid input in normal state? In final state?\n\n"
                            "3. State management:\n"
                            "   - Is state tracked correctly across method calls?\n"
                            "   - Are state updates correct for all cases INCLUDING final states?\n"
                            "   - Are state validity checks in place?\n\n"
                            "4. Business logic:\n"
                            "   - Are ALL requirements and rules implemented?\n"
                            "   - Are special cases handled correctly (especially final/boundary cases)?\n"
                            "   - Does logic match problem specification exactly?\n"
                            "   - Do final/boundary states have different rules than normal states?\n\n"
                            "5. Examples verification:\n"
                            "   - Trace through simple example step-by-step\n"
                            "   - Trace through complex example\n"
                            "   - Trace through boundary case (final state, last item)\n"
                            "   - Does code produce correct results?\n\n"
                            "6. Edge cases:\n"
                            "   - Empty/boundary inputs handled?\n"
                            "   - Invalid inputs raise exceptions (in ALL states)?\n"
                            "   - Out-of-order operations caught?\n"
                            "   - Final/last state edge cases handled with correct validation?\n\n"
                            "Reply 'APPROVED' if fully correct, or list SPECIFIC issues found."
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
