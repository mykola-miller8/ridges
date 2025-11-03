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

    # Enhanced system prompt with extreme attention to detail for spatial/graph problems
    system_msg = (
        "You are an expert Python engineer who writes flawless, production-ready code.\n\n"
        "CRITICAL WORKFLOW - FOLLOW EVERY STEP METICULOUSLY:\n\n"
        "STEP 1: DEEP UNDERSTANDING\n"
        "- Read the COMPLETE problem statement multiple times\n"
        "- Identify the core task, requirements, constraints, and rules\n"
        "- If examples are provided, study EACH ONE in detail:\n"
        "  * What is the exact input format? How is it structured?\n"
        "  * What is the expected output?\n"
        "  * WHY does that input produce that output? What's the logic?\n"
        "  * What pattern connects inputs to outputs?\n"
        "- Study MULTIPLE examples to find patterns and edge cases\n\n"
        "STEP 2: PROBLEM DECOMPOSITION\n"
        "- Break the problem into clear sub-tasks\n"
        "- Plan to use helper methods for complex operations\n"
        "- Identify required data structures\n"
        "- Plan your algorithm approach\n\n"
        "STEP 3: ALGORITHM DESIGN (CRITICAL FOR CORRECTNESS)\n"
        "- Choose the RIGHT algorithm:\n"
        "  * Graph connectivity: BFS or DFS to find paths\n"
        "  * Grid problems: Model coordinates accurately\n"
        "  * Spatial/neighbor problems: Get adjacency relationships EXACTLY right\n"
        "  * Parsing: Handle format precisely (whitespace, delimiters, structure)\n"
        "- For grid/spatial problems with non-standard layouts:\n"
        "  * CAREFULLY determine neighbor relationships\n"
        "  * Consider offset patterns, hexagonal/diagonal adjacency\n"
        "  * Test neighbor calculation mentally on paper\n"
        "  * Double-check coordinate system and indexing\n"
        "- For graph traversal:\n"
        "  * Identify starting points (all cells on one edge)\n"
        "  * Identify goal condition (reaching opposite edge)\n"
        "  * Use proper visited tracking to avoid cycles\n"
        "  * Ensure you explore ALL reachable neighbors\n\n"
        "STEP 4: MENTAL VERIFICATION (MANDATORY)\n"
        "- Trace through MULTIPLE examples mentally:\n"
        "  * Start with a simple example first\n"
        "  * Then trace through a complex example\n"
        "  * For each example:\n"
        "    - Parse the input into your data structures\n"
        "    - Walk through your algorithm step-by-step\n"
        "    - Track visited nodes, current state, decisions\n"
        "    - Verify you get the expected output\n"
        "- If your trace reveals an error, STOP and redesign\n"
        "- Pay special attention to:\n"
        "  * Boundary conditions (edges of grids)\n"
        "  * Coordinate calculations (no off-by-one errors)\n"
        "  * Neighbor relationships (all valid neighbors included)\n"
        "  * Termination conditions (when to stop)\n\n"
        "STEP 5: IMPLEMENTATION\n"
        "- Write clean, well-structured code:\n"
        "  * Use helper methods for complex sub-tasks\n"
        "  * Use descriptive variable names\n"
        "  * Add comments for non-obvious logic\n"
        "  * Handle all edge cases explicitly\n"
        "- Implement ALL required methods completely (no 'pass' stubs)\n"
        "- For grid/graph problems:\n"
        "  * Implement neighbor calculation very carefully\n"
        "  * Double-check coordinate math and boundary checks\n"
        "  * Use proper BFS/DFS with visited tracking\n"
        "  * Test starting and ending conditions carefully\n"
        "- Parse input formats exactly as specified\n"
        "- Validate inputs and raise exceptions for violations\n"
        "- Return output in exact format required\n\n"
        "CRITICAL CORRECTNESS RULES:\n"
        "- NEVER leave method bodies empty or with just 'pass'\n"
        "- For spatial problems: Get neighbor relationships EXACTLY right\n"
        "- For grid problems: Test coordinate calculations carefully\n"
        "- For graph traversal: Explore ALL valid neighbors, not just some\n"
        "- For parsing: Handle whitespace and structure precisely\n"
        "- Trace through at least 2 examples mentally before finalizing\n"
        "- If a problem involves complex spatial relationships (hex grids, offset grids, etc.):\n"
        "  * Be EXTRA careful with neighbor calculation\n"
        "  * Consider how indentation/offset affects coordinates\n"
        "  * Draw it out mentally or on paper if needed\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a single Python code block with the complete main.py.\n"
        "Start with '# main.py' as the first line.\n"
        "Format: ```python\\n# main.py\\n[complete code]\\n```\n"
        "No explanations, no prose, no incomplete implementations."
    )

    user_msg = (
        f"# Problem Statement\n{problem_statement[:15000]}\n\n"
        f"# Current Repository\n{repo_summary}\n\n"
        "IMPLEMENT A COMPLETE, CORRECT SOLUTION:\n"
        "1. Study the problem and ALL examples thoroughly\n"
        "2. Design your algorithm carefully - get neighbor relationships right!\n"
        "3. Mentally trace through MULTIPLE examples to verify correctness\n"
        "4. Break complex logic into helper methods\n"
        "5. Implement ALL methods with complete, correct logic\n"
        "6. For spatial/grid problems: Be extra careful with coordinate calculations\n"
        "7. Handle all edge cases and validation\n\n"
        "Your solution must be COMPLETE and CORRECT for ALL cases."
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
                        pass
                
                # Step 2: Self-review with emphasis on correctness
                # Only do review on first pass through models
                if attempt < len(AGENT_MODELS):
                    review_messages = [
                        {"role": "system", "content": (
                            "You are a meticulous code reviewer who finds subtle bugs through careful analysis.\n"
                            "Focus on: algorithm correctness, coordinate calculations, neighbor relationships, "
                            "input parsing, edge cases, and logic errors."
                        )},
                        {"role": "user", "content": (
                            f"# Problem Statement\n{problem_statement[:15000]}\n\n"
                            f"# Proposed Code\n```python\n{code}\n```\n\n"
                            "Review this code with EXTREME attention to detail:\n\n"
                            "1. COMPLETENESS: Are ALL methods fully implemented?\n"
                            "2. INPUT PARSING: Does it correctly parse the input?\n"
                            "   - Handle whitespace, indentation, delimiters?\n"
                            "   - Build correct data structures?\n"
                            "3. SPATIAL/NEIGHBOR RELATIONSHIPS (CRITICAL):\n"
                            "   - For grid/graph problems: Are neighbor calculations EXACTLY right?\n"
                            "   - Are coordinates calculated correctly (no off-by-one)?\n"
                            "   - For offset/hex grids: Is the offset pattern handled correctly?\n"
                            "   - Are boundary checks correct?\n"
                            "4. ALGORITHM LOGIC:\n"
                            "   - For graph traversal: Does it explore ALL valid neighbors?\n"
                            "   - Are starting and ending conditions correct?\n"
                            "   - Is visited tracking correct?\n"
                            "5. EXAMPLE VERIFICATION: Trace through a complex example:\n"
                            "   - Parse the input step by step\n"
                            "   - Execute the algorithm step by step\n"
                            "   - Does it produce the correct output?\n"
                            "   - Are there any logic errors in the trace?\n"
                            "6. EDGE CASES: Empty inputs, single elements, boundaries?\n\n"
                            "Respond with 'APPROVED' if the code is complete and correct, "
                            "or list specific issues with line numbers and details."
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
                                        "- Fix any neighbor calculation or coordinate errors\n"
                                        "- Correct algorithm logic errors\n"
                                        "- Ensure parsing is accurate\n"
                                        "- Verify correctness against examples\n"
                                        "- Double-check spatial relationships and boundaries\n\n"
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
