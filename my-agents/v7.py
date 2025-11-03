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

    # Balanced system prompt - concise but comprehensive
    system_msg = (
        "You are an expert Python engineer who writes flawless, production-ready code.\n\n"
        "APPROACH:\n"
        "1. Read the entire problem carefully, including all examples\n"
        "2. If examples exist, trace through them to understand the required logic\n"
        "3. Design your algorithm before coding:\n"
        "   - Choose appropriate data structures\n"
        "   - For graphs/grids: plan traversal strategy (BFS/DFS)\n"
        "   - For spatial problems: carefully model coordinates and neighbors\n"
        "   - For parsing: handle the exact input format\n"
        "4. Implement complete, working code with all methods fully implemented\n"
        "5. Mentally verify against examples to catch logic errors\n\n"
        "CRITICAL REQUIREMENTS:\n"
        "- Implement ALL methods completely (never leave empty or with just 'pass')\n"
        "- Parse input formats exactly as specified\n"
        "- For graph/grid problems: get neighbor relationships right, use proper BFS/DFS\n"
        "- For spatial problems: calculate coordinates carefully (watch for offsets, indexing)\n"
        "- Handle all edge cases (empty inputs, boundaries, single elements)\n"
        "- Implement validation rules and raise exceptions as needed\n"
        "- Verify your logic works for the provided examples\n\n"
        "COMMON PITFALLS TO AVOID:\n"
        "- Incorrect neighbor calculation in grids (especially hex/offset grids)\n"
        "- Off-by-one errors in coordinates or boundaries\n"
        "- Not exploring all valid neighbors in graph traversal\n"
        "- Incorrect parsing of formatted input (missing whitespace handling)\n"
        "- Missing edge cases\n\n"
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
        "- Handles all examples correctly\n"
        "- Implements all methods fully (no empty implementations)\n"
        "- Handles all edge cases\n"
        "- Uses appropriate algorithms and data structures"
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
                                    "The code has empty method bodies. Implement ALL methods with complete logic.\n"
                                    "Format: ```python\\n# main.py\\n[complete implementation]\\n```"
                                )
                            })
                            break
                except Exception:
                    pass
                
                # Step 2: Self-review (only on first pass)
                if attempt < len(AGENT_MODELS):
                    review_messages = [
                        {"role": "system", "content": (
                            "You are a code reviewer who finds bugs through careful analysis."
                        )},
                        {"role": "user", "content": (
                            f"# Problem\n{problem_statement[:15000]}\n\n"
                            f"# Code\n```python\n{code}\n```\n\n"
                            "Review for:\n"
                            "1. Are all methods fully implemented?\n"
                            "2. Does input parsing handle the format correctly?\n"
                            "3. For graphs/grids: Are neighbors calculated correctly? Is traversal sound?\n"
                            "4. Trace through an example: Does it produce correct output?\n"
                            "5. Are edge cases handled?\n\n"
                            "Reply 'APPROVED' if correct, or list specific issues."
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
                                        "Fix all issues and return corrected code.\n"
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
