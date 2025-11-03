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

    # Build context from available files
    parts: List[str] = []
    for name in ("main.py", "tests.py"):
        content = _read(name)
        if content:
            parts.append(f"### {name}\n```python\n{content[:10000]}\n```")
    repo_summary = "\n\n".join(parts)

    # Enhanced system prompt with stronger algorithmic guidance
    system_msg = (
        "You are an expert Python engineer who writes flawless, production-ready code.\n\n"
        "CRITICAL WORKFLOW - FOLLOW THESE STEPS:\n"
        "1. READ THE ENTIRE PROBLEM: Study every detail, example, constraint, and rule\n"
        "2. UNDERSTAND EXAMPLES: If examples are provided, trace through them COMPLETELY to understand:\n"
        "   - What inputs are given and their format/structure\n"
        "   - What outputs are expected and why\n"
        "   - The underlying logic that produces each output\n"
        "   - Edge cases or special patterns demonstrated\n"
        "3. DESIGN THE ALGORITHM: Before writing ANY code, plan:\n"
        "   - What algorithm/approach is needed (BFS/DFS/dynamic programming/parsing/etc.)\n"
        "   - What data structures to use and why\n"
        "   - How to handle input parsing (especially complex formats like grids, trees, graphs)\n"
        "   - How to correctly compute the result\n"
        "4. IMPLEMENT WITH PRECISION: Write code that:\n"
        "   - Correctly parses input in the exact format provided\n"
        "   - Implements the algorithm with correct logic\n"
        "   - Handles ALL edge cases and boundaries\n"
        "   - Validates inputs and raises exceptions for violations\n"
        "   - Produces output in the exact format required\n\n"
        "ALGORITHM CORRECTNESS (CRITICAL):\n"
        "- For graph/grid/connectivity problems: Implement proper traversal (BFS/DFS/Union-Find)\n"
        "- For spatial problems: Correctly model neighbor relationships (consider offsets, coordinates, adjacency)\n"
        "- For parsing problems: Handle whitespace, delimiters, and formatting exactly as specified\n"
        "- For stateful problems: Track state transitions correctly and handle all cases\n"
        "- Test your logic mentally against EVERY example before finalizing\n\n"
        "INPUT PARSING AND DATA STRUCTURES:\n"
        "- Pay close attention to input format (grids with indentation, nested structures, etc.)\n"
        "- If the input has special formatting (spaces, tabs, newlines), parse it correctly\n"
        "- Choose appropriate data structures (2D arrays, graphs, dictionaries, sets, etc.)\n"
        "- Model the problem domain accurately in your data structures\n\n"
        "VALIDATION AND EXCEPTIONS:\n"
        "- Implement ALL validation rules mentioned in the problem\n"
        "- Raise descriptive exceptions for invalid inputs\n"
        "- Handle edge cases gracefully\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a single Python code block with the complete main.py.\n"
        "Start with '# main.py' as the first line.\n"
        "Format: ```python\\n# main.py\\n[complete code]\\n```\n"
        "No explanations, no prose."
    )

    user_msg = (
        f"# Problem Statement\n{problem_statement[:15000]}\n\n"
        f"# Current Repository\n{repo_summary}\n\n"
        "IMPLEMENTATION CHECKLIST:\n"
        "1. Read and understand the ENTIRE problem statement, including all examples\n"
        "2. If examples exist, trace through them to understand the required behavior\n"
        "3. Design your algorithm/approach before coding\n"
        "4. Implement with correct parsing, algorithm logic, and validation\n"
        "5. Mentally verify your solution works for ALL provided examples\n\n"
        "Implement a complete, correct solution."
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
                
                # Step 2: Self-review for correctness, algorithms, and validation
                # Only do review on first pass through models to balance quality vs speed
                if attempt < len(AGENT_MODELS):
                    review_messages = [
                        {"role": "system", "content": (
                            "You are a meticulous code reviewer. Your job is to find ANY issues.\n"
                            "Focus especially on: algorithm correctness, input parsing, logic errors, "
                            "edge cases, and validation."
                        )},
                        {"role": "user", "content": (
                            f"# Problem Statement\n{problem_statement[:15000]}\n\n"
                            f"# Proposed Code\n```python\n{code}\n```\n\n"
                            "Review this code thoroughly. Check:\n"
                            "1. INPUT PARSING: Does it correctly parse the input format? "
                            "Are there any issues with whitespace, delimiters, or structure?\n"
                            "2. ALGORITHM: Is the core algorithm correct? For graph/grid problems, "
                            "are neighbor relationships and traversal logic correct?\n"
                            "3. EXAMPLES: If the problem includes examples, trace through them. "
                            "Would this code produce the correct output for each example?\n"
                            "4. EDGE CASES: Does it handle empty inputs, single elements, boundaries, etc.?\n"
                            "5. VALIDATION: Does it implement required validation and raise exceptions?\n\n"
                            "Mentally trace through examples step-by-step to verify correctness.\n"
                            "Respond with 'APPROVED' if correct, or list specific issues with line numbers."
                        )}
                    ]
                    
                    try:
                        review_response = _call_llm(review_messages, run_id, attempt, 120)
                        
                        # If review finds issues, request refinement
                        if review_response and "APPROVED" not in review_response.upper():
                            if attempt < max_attempts - 1:
                                messages.append({"role": "assistant", "content": response})
                                messages.append({
                                    "role": "user",
                                    "content": (
                                        f"Code review found issues:\n{review_response}\n\n"
                                        "Revise the code to fix ALL issues. Pay special attention to:\n"
                                        "- Correct input parsing and data structure modeling\n"
                                        "- Accurate algorithm implementation\n"
                                        "- Verification against examples\n"
                                        "Format: ```python\\n# main.py\\n[revised code]\\n```"
                                    )
                                })
                                break  # Retry with feedback
                    except Exception:
                        # Review failed, but code is syntactically valid, so accept it
                        pass
                
                # Code is syntactically valid and passed review (or review skipped)
                return _build_single_file_patch("main.py", code)
            
        except Exception:
            # LLM call failed, try next model
            continue

    # All attempts exhausted, return empty patch
    return ""
