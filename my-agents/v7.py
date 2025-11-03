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
    timeout_s: int = 300,
    temperature: float = 0.0
) -> str:
    """Call inference gateway with retry logic."""
    url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
    headers = {"Content-Type": "application/json"}
    model = AGENT_MODELS[attempt % len(AGENT_MODELS)]
    
    body = {
        "run_id": run_id,
        "messages": messages,
        "temperature": temperature,
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

    # Focused system prompt with clear algorithmic guidance
    system_msg = (
        "You are an expert Python engineer who writes correct, production-ready code.\n\n"
        "YOUR PROCESS:\n"
        "1. Read the problem statement completely, including ALL examples\n"
        "2. Understand what each example demonstrates - study the pattern\n"
        "3. Design your algorithm:\n"
        "   - For graph/grid connectivity: Use BFS or DFS to find paths\n"
        "   - For spatial/grid problems: Get coordinate system and neighbors RIGHT\n"
        "   - For parsing: Handle exact format (whitespace, structure, delimiters)\n"
        "4. Implement complete code with all methods fully working\n"
        "5. Mentally test against examples to verify correctness\n\n"
        "CRITICAL FOR GRID/GRAPH PROBLEMS:\n"
        "When implementing graph traversal or grid connectivity:\n"
        "- Parse the grid correctly, building a proper data structure\n"
        "- Define neighbor relationships PRECISELY (for hex grids, offset grids, etc.)\n"
        "- Start from ALL cells on the starting edge (not just one)\n"
        "- Use BFS/DFS to explore connected cells\n"
        "- Track visited cells to avoid infinite loops\n"
        "- Check if ANY path reaches the opposite edge\n"
        "- Handle edge cases: empty grids, single cells, no paths\n\n"
        "NEIGHBOR CALCULATION FOR IRREGULAR GRIDS:\n"
        "If the grid has indentation or offsets (like hex grids):\n"
        "- Rows may have different neighbor patterns based on row parity (even/odd)\n"
        "- Account for row offset when calculating column neighbors\n"
        "- Test neighbor calculation on paper with examples\n"
        "- Ensure boundary checks are correct\n\n"
        "IMPLEMENTATION RULES:\n"
        "- Implement ALL methods completely (no empty/pass-only bodies)\n"
        "- Use helper methods for complex operations\n"
        "- Add comments for non-obvious logic\n"
        "- Handle all edge cases\n"
        "- Validate inputs and raise exceptions as needed\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a Python code block starting with '# main.py'.\n"
        "Format: ```python\\n# main.py\\n[complete code]\\n```\n"
        "No explanations."
    )

    user_msg = (
        f"# Problem Statement\n{problem_statement[:15000]}\n\n"
        f"# Current Repository\n{repo_summary}\n\n"
        "Implement a complete, correct solution.\n"
        "Study the examples carefully to understand the required behavior.\n"
        "For grid/graph problems, get the neighbor relationships exactly right."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Extended attempts with temperature variation
    max_attempts = len(AGENT_MODELS) * 3  # Increased from 2x to 3x
    
    for attempt in range(max_attempts):
        try:
            # Temperature strategy: 0.0 for first pass, 0.3 for later attempts
            temp = 0.0 if attempt < len(AGENT_MODELS) else 0.3
            
            # Generate code
            response = _call_llm(messages, run_id, attempt, 300, temperature=temp)
            code_blocks = _extract_code_blocks(response)
            
            for code in code_blocks:
                # Validate syntax
                is_valid, error_msg = _validate_syntax(code)
                
                if not is_valid:
                    if attempt < max_attempts - 1:
                        messages.append({"role": "assistant", "content": response})
                        messages.append({
                            "role": "user",
                            "content": (
                                f"Syntax error: {error_msg}\n\n"
                                "Fix and return corrected code.\n"
                                "Format: ```python\\n# main.py\\n[code]\\n```"
                            )
                        })
                        break
                    continue
                
                # Check for incomplete implementations
                try:
                    tree = ast.parse(code)
                    has_empty = False
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                                has_empty = True
                                break
                    
                    if has_empty:
                        if attempt < max_attempts - 1:
                            messages.append({"role": "assistant", "content": response})
                            messages.append({
                                "role": "user",
                                "content": (
                                    "Incomplete implementation. All methods must have working logic.\n"
                                    "Format: ```python\\n# main.py\\n[complete code]\\n```"
                                )
                            })
                            break
                except Exception:
                    pass
                
                # Review phase (first 6 attempts only)
                if attempt < len(AGENT_MODELS) + 2:
                    review_messages = [
                        {"role": "system", "content": "You are a code reviewer focused on correctness."},
                        {"role": "user", "content": (
                            f"# Problem\n{problem_statement[:15000]}\n\n"
                            f"# Code\n```python\n{code}\n```\n\n"
                            "Review checklist:\n"
                            "1. All methods implemented?\n"
                            "2. Input parsing correct?\n"
                            "3. For grids: Neighbor calculation correct for the grid type?\n"
                            "4. For graphs: BFS/DFS logic sound?\n"
                            "5. Trace one simple and one complex example - correct outputs?\n"
                            "6. Edge cases handled?\n\n"
                            "Reply 'APPROVED' or list issues."
                        )}
                    ]
                    
                    try:
                        review = _call_llm(review_messages, run_id, attempt, 120)
                        
                        if review and "APPROVED" not in review.upper():
                            if attempt < max_attempts - 1:
                                messages.append({"role": "assistant", "content": response})
                                messages.append({
                                    "role": "user",
                                    "content": (
                                        f"Issues found:\n{review}\n\n"
                                        "Fix all issues. Pay special attention to neighbor "
                                        "calculations and algorithm correctness.\n"
                                        "Format: ```python\\n# main.py\\n[fixed code]\\n```"
                                    )
                                })
                                break
                    except Exception:
                        pass
                
                # Code passed validation
                return _build_single_file_patch("main.py", code)
            
        except Exception:
            continue

    # All attempts exhausted
    return ""
