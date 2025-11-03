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

    # Enhanced system prompt with decomposition and careful implementation
    system_msg = (
        "You are an expert Python engineer who writes flawless, production-ready code.\n\n"
        "CRITICAL WORKFLOW - FOLLOW EVERY STEP:\n\n"
        "STEP 1: DEEP UNDERSTANDING\n"
        "- Read the COMPLETE problem statement word-by-word\n"
        "- Identify the core task and requirements\n"
        "- Note all constraints, rules, and edge cases mentioned\n"
        "- If examples are provided, study each one carefully:\n"
        "  * What is the input format and structure?\n"
        "  * What is the expected output?\n"
        "  * Why does that input produce that output?\n"
        "  * What pattern or logic connects input to output?\n\n"
        "STEP 2: PROBLEM DECOMPOSITION\n"
        "- Break the problem into logical sub-tasks\n"
        "- For complex problems, plan to use helper methods/functions\n"
        "- Identify what data structures are needed\n"
        "- Plan your algorithm approach (e.g., graph traversal, dynamic programming, parsing, etc.)\n\n"
        "STEP 3: ALGORITHM DESIGN\n"
        "- Choose the RIGHT algorithm for the problem:\n"
        "  * Graph/connectivity: BFS, DFS, or Union-Find\n"
        "  * Grid/spatial: Model coordinates and adjacency correctly\n"
        "  * Parsing: Handle format precisely (whitespace, delimiters, structure)\n"
        "  * State machines: Track state transitions accurately\n"
        "- Think through the algorithm step-by-step\n"
        "- Consider edge cases and how to handle them\n\n"
        "STEP 4: MENTAL VERIFICATION\n"
        "- Before writing code, trace through at least one example mentally:\n"
        "  * Start with the input\n"
        "  * Walk through your algorithm step-by-step\n"
        "  * Verify you get the expected output\n"
        "- If your mental trace doesn't work, revise your approach\n\n"
        "STEP 5: IMPLEMENTATION\n"
        "- Write clean, well-structured code:\n"
        "  * Use helper methods for complex sub-tasks\n"
        "  * Use descriptive variable names\n"
        "  * Add comments for non-obvious logic\n"
        "  * Handle all edge cases explicitly\n"
        "- Implement ALL required methods completely (no 'pass' stubs)\n"
        "- Parse input formats exactly as specified\n"
        "- Validate inputs and raise exceptions for rule violations\n"
        "- Return output in the exact format required\n\n"
        "CRITICAL IMPLEMENTATION RULES:\n"
        "- NEVER leave method bodies empty or with just 'pass'\n"
        "- ALWAYS implement complete logic in every method\n"
        "- For graph/grid problems: Implement proper traversal (BFS/DFS)\n"
        "- For spatial problems: Get neighbor relationships exactly right\n"
        "- For parsing: Handle whitespace, newlines, and delimiters correctly\n"
        "- Test your logic mentally against ALL examples\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a single Python code block with the complete main.py.\n"
        "Start with '# main.py' as the first line.\n"
        "Format: ```python\\n# main.py\\n[complete code]\\n```\n"
        "No explanations, no prose, no incomplete implementations."
    )

    user_msg = (
        f"# Problem Statement\n{problem_statement[:15000]}\n\n"
        f"# Current Repository\n{repo_summary}\n\n"
        "IMPLEMENT A COMPLETE SOLUTION:\n"
        "1. Study the problem and examples thoroughly\n"
        "2. Design your algorithm before coding\n"
        "3. Break complex logic into helper methods\n"
        "4. Implement ALL methods with complete logic (no empty bodies!)\n"
        "5. Mentally trace through examples to verify correctness\n"
        "6. Handle all edge cases and validation\n\n"
        "Your solution must be COMPLETE and CORRECT."
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
                if "pass" in code:
                    # Verify if 'pass' is actually used in method bodies (not just in docstrings/comments)
                    try:
                        tree = ast.parse(code)
                        has_empty_methods = False
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                # Check if function body is just 'pass'
                                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                                    has_empty_methods = True
                                    break
                        
                        if has_empty_methods:
                            if attempt < max_attempts - 1:
                                messages.append({"role": "assistant", "content": response})
                                messages.append({
                                    "role": "user",
                                    "content": (
                                        "ERROR: The code contains empty method bodies (just 'pass').\n\n"
                                        "You MUST implement ALL methods with complete logic. "
                                        "No method should be left as just 'pass'.\n\n"
                                        "Implement the complete solution with full logic in every method.\n"
                                        "Format: ```python\\n# main.py\\n[complete implementation]\\n```"
                                    )
                                })
                                break
                    except Exception:
                        pass  # If we can't parse, continue to review
                
                # Step 2: Self-review for correctness, algorithms, and validation
                # Only do review on first pass through models to balance quality vs speed
                if attempt < len(AGENT_MODELS):
                    review_messages = [
                        {"role": "system", "content": (
                            "You are a meticulous code reviewer who finds issues through careful analysis.\n"
                            "Focus on: completeness, algorithm correctness, input parsing, logic errors, "
                            "edge cases, and validation."
                        )},
                        {"role": "user", "content": (
                            f"# Problem Statement\n{problem_statement[:15000]}\n\n"
                            f"# Proposed Code\n```python\n{code}\n```\n\n"
                            "Review this code thoroughly:\n\n"
                            "1. COMPLETENESS: Are ALL methods fully implemented? No empty bodies or 'pass' stubs?\n"
                            "2. INPUT PARSING: Does it correctly parse the input format?\n"
                            "   - Handle whitespace, indentation, delimiters correctly?\n"
                            "   - Build appropriate data structures from input?\n"
                            "3. ALGORITHM: Is the core logic correct?\n"
                            "   - For graph/grid: Are neighbor relationships and traversal correct?\n"
                            "   - For state: Are state transitions handled properly?\n"
                            "   - For search: Is the search logic sound?\n"
                            "4. EXAMPLES: Trace through one example step-by-step:\n"
                            "   - Does the code produce the correct output?\n"
                            "   - Are there any logic errors in the trace?\n"
                            "5. EDGE CASES: Empty inputs, single elements, boundaries, etc.?\n"
                            "6. VALIDATION: Required validation and exceptions?\n\n"
                            "Respond with 'APPROVED' if the code is complete and correct, "
                            "or list specific issues with details."
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
                                        "Revise the code to fix ALL issues:\n"
                                        "- Implement any missing logic completely\n"
                                        "- Fix algorithm or parsing errors\n"
                                        "- Verify correctness against examples\n"
                                        "- Ensure all methods are fully implemented\n\n"
                                        "Format: ```python\\n# main.py\\n[revised complete code]\\n```"
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
