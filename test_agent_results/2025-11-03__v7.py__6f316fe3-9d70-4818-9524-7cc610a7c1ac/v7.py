import os
import re
import json
import uuid
import time
import ast
import subprocess
from typing import Any, Dict, List

import requests


# v7 seed agent: generic code agent that proposes a full-file patch for main.py only.
# - Never embeds problem-specific constants or dataset names
# - Uses only the inference gateway exposed via INFERENCE_URL/SANDBOX_PROXY_URL
# - Returns a unified diff that replaces main.py entirely
# - Includes self-review refinement pass to catch edge cases


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
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


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


def _extract_main_py(response: str) -> str:
    """Extract Python code from LLM response with multiple fallback strategies."""
    if not response:
        return ""
    
    # Strategy 1: Explicitly headed block with main.py comment
    m = re.findall(r"```python\s*\n#\s*main\.py\n([\s\S]*?)\n```", response, re.DOTALL)
    if m and m[0].strip():
        return m[0].strip()
    
    # Strategy 2: First python block
    m2 = re.findall(r"```python\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    if m2 and m2[0].strip():
        return m2[0].strip()
    
    # Strategy 3: Any code block (no language tag)
    m3 = re.findall(r"```\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    if m3 and m3[0].strip():
        # Check if it looks like Python (has def, class, import, etc.)
        code = m3[0].strip()
        if any(kw in code for kw in ["def ", "class ", "import ", "from "]):
            return code
    
    return ""


def _is_valid_python(code: str) -> bool:
    """Check if code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def _is_substantial_code(code: str) -> bool:
    """Check if code appears to be a substantial implementation (not just a stub)."""
    if not code or len(code) < 50:
        return False
    # Count actual code constructs
    has_function = "def " in code
    has_class_or_logic = "class " in code or "if " in code or "for " in code or "while " in code
    lines_of_code = len([line for line in code.split('\n') if line.strip() and not line.strip().startswith('#')])
    return has_function and lines_of_code >= 5


def _call_llm(messages: List[Dict[str, str]], run_id: str, attempt: int, timeout_s: int = 240) -> str:
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
    last_err: Exception | None = None
    for r in range(3):
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
            time.sleep(1 + r)
    raise last_err if last_err else RuntimeError("LLM call failed")


def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False) -> str:
    """Entry point required by the evaluation harness.

    Returns a unified diff patch that fully replaces main.py.
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

    # Compact repository summary for context (generic, no problem-specific assumptions)
    parts: List[str] = []
    for name in ("main.py", "tests.py"):
        content = _read(name)
        if content:
            parts.append(f"### {name}\n```python\n{content[:8000]}\n```")
    summary = "\n\n".join(parts)

    system_msg = (
        "You are a senior Python engineer who writes correct, robust, COMPLETE code.\n"
        + ("Do not modify tests.py; only change main.py.\n" if mode == "tests_available" else "")
        + "CRITICAL REQUIREMENTS:\n"
        "- Implement ALL functions, classes, and exports mentioned in the problem statement\n"
        "- Identify and enforce ALL constraints mentioned in the problem:\n"
        "  * If it says 'must be unique', implement uniqueness checking and collision handling\n"
        "  * If it says 'must be random', ensure true randomness (not predictable sequences)\n"
        "  * If constraints conflict with randomness, implement retry/regeneration logic\n"
        "- Infer the complete API surface (methods, properties, attributes) from the problem description:\n"
        "  * If it mentions 'settable values', implement both getter and setter\n"
        "  * If it mentions 'registering callbacks', implement add/remove callback methods\n"
        "  * If it mentions 'computed values', implement automatic recomputation\n"
        "- Include ALL required custom exceptions with correct signatures\n"
        "- Ensure the code is COMPLETE and FUNCTIONAL, not a stub or placeholder\n"
        "- Follow the exact function/class signatures specified\n"
        "- For complex systems (reactive, event-driven, observer patterns), implement proper architecture\n"
        "- Use the exact exception types and error messages as specified\n\n"
        "Pay special attention to:\n"
        "- Detailed specifications and rules:\n"
        "  * If the problem lists multiple rules, implement ALL of them\n"
        "  * For parsing/formatting problems, edge cases in specifications are critical\n"
        "  * Escape sequences, special characters, and whitespace handling require exact implementation\n"
        "- Input format and characteristics:\n"
        "  * Does case sensitivity matter? (e.g., 'D' vs 'd' may mean different things)\n"
        "  * Are there lookup tables or rules that depend on input characteristics?\n"
        "  * Must you apply different logic based on input format?\n"
        "- Domain-specific rules and real-world expectations:\n"
        "  * What implicit rules exist in this domain? (e.g., debts offset, resources don't duplicate)\n"
        "  * How would this work in the real world?\n"
        "  * What consistency rules must be maintained across operations?\n"
        "- State interactions and consistency:\n"
        "  * When new operations interact with existing state, how should they combine?\n"
        "  * Should values net out, accumulate, or replace?\n"
        "- Edge cases and boundary conditions (minimum, maximum, base cases):\n"
        "  * Special values: zero, negative, empty, null - do these need special handling?\n"
        "  * Don't skip validation for special values unless explicitly allowed\n"
        "  * Edge cases often bypass normal logic paths - ensure validation still applies\n"
        "- Exact output specifications:\n"
        "  * What EXACTLY should be returned/included in output?\n"
        "  * Should results include boundary elements (e.g., start/end nodes in paths)?\n"
        "  * What is the exact format and ordering of output?\n"
        "  * Whitespace handling: Are trailing/leading spaces significant? Must they be preserved?\n"
        "  * Output formatting: Don't strip whitespace unless explicitly allowed\n"
        "- When examples are provided, extract ALL patterns including:\n"
        "  * The first/simplest/base case\n"
        "  * The last/final/terminal case\n"
        "  * The general/middle cases\n"
        "  * Any special behaviors for edge values\n"
        "  * EXACT output format from examples\n"
        "- Input validation and error handling with correct precedence:\n"
        "  * Check for invalid INPUT FORMAT/CHARACTERS first (e.g., letters, bad punctuation)\n"
        "  * Then validate structure/types (e.g., is it the right type?)\n"
        "  * Then check LENGTH/COMPLETENESS (e.g., too few/many items)\n"
        "  * Then check VALUE CONSTRAINTS (e.g., must start with X, can't be zero)\n"
        "  * Finally check SEMANTIC CORRECTNESS (e.g., relationships between fields)\n"
        "  * Use the exact exception types and messages specified\n"
        "  * When multiple error conditions are similar, check for SPECIFIC cases before GENERAL cases\n"
        "  * Each distinct error condition should have its own specific error message\n"
        "  * Distinguish semantic errors (unsupported operation/wrong domain) from syntax errors (malformed structure)\n"
        "- All validation rules mentioned in the problem statement\n\n"
        "Return ONLY one code block containing the complete main.py with a '# main.py' header.\n"
        "Format exactly as:\n```python\n# main.py\n[complete code]\n```\n"
        "No prose. Deterministic, production-quality code."
    )
    user_msg = (
        f"Problem Statement:\n{problem_statement[:12000]}\n\n"
        f"Repository Summary:\n{summary}\n\n"
        "Implement a COMPLETE, FUNCTIONAL solution:\n"
        "- Include ALL required functions, classes, and custom exceptions\n"
        "- Identify ALL explicit constraints in the problem statement:\n"
        "  * If it says 'must be unique', implement tracking and collision detection\n"
        "  * If it says 'must be random', implement proper randomization\n"
        "  * For conflicting requirements (random + unique), use retry loops\n"
        "- Carefully read the problem to infer the complete API:\n"
        "  * What properties/methods must each class have?\n"
        "  * What behaviors are implied (e.g., 'settable' means getter AND setter)?\n"
        "  * For complex systems, what architecture is needed?\n"
        "- Handle ALL cases mentioned in the problem statement\n"
        "- Read specifications carefully and implement ALL rules:\n"
        "  * If multiple rules are listed, each one must be implemented\n"
        "  * For parsing/string processing, pay special attention to edge cases\n"
        "  * Escape sequences and special character handling must match specs exactly\n"
        "- Consider domain conventions and implicit rules:\n"
        "  * Think about how this would work in the real world\n"
        "  * What consistency rules apply? (e.g., do debts offset? do resources net out?)\n"
        "  * When operations interact with existing state, how should they combine?\n"
        "- If examples are provided:\n"
        "  * Carefully analyze them to understand EXACT output format and content\n"
        "  * Note what elements are included/excluded (e.g., do paths include start node?)\n"
        "  * Check whitespace carefully: Are there trailing/leading spaces that must be preserved?\n"
        "  * Verify your logic would produce the exact same output for each example\n"
        "- Test your logic mentally with the smallest/first case and largest/last case\n"
        "- Test edge cases with special values:\n"
        "  * Zero, empty, null, negative - ensure validation applies to these too\n"
        "  * Don't skip validation checks for special values unless explicitly allowed\n"
        "  * Avoid early returns that bypass validation for edge cases\n"
        "- When validating input:\n"
        "  * Check format/character validity BEFORE checking length or value constraints\n"
        "  * For related error conditions, check SPECIFIC cases before GENERAL cases\n"
        "  * Use distinct, specific error messages for each type of validation failure\n"
        "  * Distinguish: semantic errors (wrong operation/domain) vs syntax errors (malformed input)\n"
        "- Ensure the code is ready to run, not a stub or partial implementation"
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    best_code = ""
    
    # Pass 1: Generate initial solution
    for attempt in range(len(AGENT_MODELS)):
        try:
            resp = _call_llm(messages, run_id, attempt, 300)
            main_src = _extract_main_py(resp)
            if main_src and _is_valid_python(main_src) and _is_substantial_code(main_src):
                best_code = main_src
                break
        except Exception:
            continue
    
    # Pass 2: Self-review and refinement (if we got valid code)
    if best_code:
        try:
            review_system = (
                "You are a code reviewer specializing in finding edge cases, bugs, and incompleteness.\n"
                "Review the provided solution and identify:\n"
                "- Missing required functions, classes, or custom exceptions\n"
                "- Unimplemented specification rules (e.g., escape sequence handling missing edge cases)\n"
                "- Incorrect parsing/formatting logic (especially for special characters and whitespace)\n"
                "- Unenforced constraints (e.g., 'must be unique' without collision detection)\n"
                "- Missing retry/regeneration logic for conflicting constraints (random + unique)\n"
                "- Incomplete API surface (missing methods, properties, or attributes)\n"
                "- Incorrect implementation of implied behaviors (e.g., 'settable' without setter)\n"
                "- Missing domain-specific rules (e.g., debts should offset, not duplicate)\n"
                "- Incorrect state interaction logic (e.g., operations not properly combining with existing state)\n"
                "- Missed edge cases, validation issues, or bugs\n"
                "- Incomplete implementations or stubs\n"
                "- Architectural issues in complex systems (reactive, event-driven, etc.)\n"
                "Then provide an improved version that fixes these issues.\n"
                + ("Do not modify tests.py; only improve main.py.\n" if mode == "tests_available" else "")
                + "Return ONLY one code block with the COMPLETE improved main.py.\n"
                "Format exactly as:\n```python\n# main.py\n[complete improved code]\n```"
            )
            review_user = (
                f"Problem Statement:\n{problem_statement[:8000]}\n\n"
                f"Current Implementation:\n```python\n{best_code}\n```\n\n"
                "Review this code carefully:\n"
                "0. Is the implementation COMPLETE and FUNCTIONAL?\n"
                "   - Are ALL required functions/classes/exceptions implemented?\n"
                "   - Are ALL constraints mentioned in the problem enforced?\n"
                "   - If uniqueness is required, is collision detection implemented?\n"
                "   - If random + unique, is there retry logic to handle collisions?\n"
                "   - Does each class have ALL required methods and properties?\n"
                "   - Are implied behaviors implemented (e.g., 'settable values' means both get and set)?\n"
                "   - For complex systems, is the architecture properly implemented?\n"
                "   - Is any part a stub or placeholder that needs full implementation?\n"
                "   - Will this code run successfully without import errors?\n"
                "1. Does it handle ALL edge cases mentioned in the problem?\n"
                "   - Are ALL specification rules implemented? (check each listed rule)\n"
                "   - For parsing/string processing: are escape sequences handled correctly?\n"
                "   - Special characters and whitespace: do they follow the exact specifications?\n"
                "   - Output whitespace: Are trailing/leading spaces preserved as required?\n"
                "   - Does the code inappropriately strip whitespace from output?\n"
                "   - Test special values (zero, empty, null): Is validation still applied?\n"
                "   - Does the code have early returns or shortcuts that bypass validation for edge cases?\n"
                "   - Test the FIRST/MINIMUM/BASE case\n"
                "   - Test the LAST/MAXIMUM/TERMINAL case\n"
                "   - Test middle/general cases\n"
                "   - Consider domain-specific rules: Are implicit conventions followed?\n"
                "   - Check state interactions: When operations modify existing state, do they behave correctly?\n"
                "   - Real-world sanity check: Would this make sense in the real world?\n"
                "2. If examples are provided, does the implementation match ALL examples exactly?\n"
                "   - Verify output format matches examples precisely\n"
                "   - Check if boundary elements are included (e.g., start/end of paths, ranges)\n"
                "   - Trace through each example step-by-step to confirm correctness\n"
                "   - Check the simplest example case\n"
                "   - Check special behaviors at boundaries\n"
                "3. Are ALL validation rules correctly implemented IN THE RIGHT ORDER?\n"
                "   The validation order MUST be:\n"
                "   a) Invalid input format/characters (e.g., letters where numbers expected)\n"
                "   b) Type/structure checks (e.g., is it a list/dict/string?)\n"
                "   c) Length/completeness (e.g., too few/many elements)\n"
                "   d) Value constraints (e.g., must start with specific value)\n"
                "   e) Semantic correctness (e.g., relationships between parts)\n"
                "   - For similar/related error conditions, check SPECIFIC cases before GENERAL ones\n"
                "   - Raise the correct exception type with the exact message specified\n"
                "   - Each distinct error condition needs its own specific error message\n"
                "   - Distinguish semantic errors (wrong operation/domain) from syntactic errors (malformed structure)\n"
                "4. Are error messages and exceptions exactly as specified in the problem?\n"
                "   - Is each error message matched to the specific condition that triggers it?\n"
                "   - Are specific error cases distinguished from general error cases?\n"
                "   - Are semantic errors (unsupported operations) distinguished from syntax errors (malformed input)?\n"
                "5. Are there any boundary conditions or special cases missed?\n\n"
                "Provide a COMPLETE improved version that fixes any issues found."
            )
            review_messages = [
                {"role": "system", "content": review_system},
                {"role": "user", "content": review_user},
            ]
            
            # Try review with a different model for fresh perspective
            review_attempt = 1 if attempt == 0 else 0
            review_resp = _call_llm(review_messages, run_id, review_attempt, 300)
            improved_src = _extract_main_py(review_resp)
            
            # Use improved version if it's valid, substantial, and not drastically shorter
            if (improved_src and _is_valid_python(improved_src) and 
                _is_substantial_code(improved_src) and len(improved_src) >= len(best_code) * 0.8):
                best_code = improved_src
        except Exception:
            # If review fails, stick with original
            pass
    
    if best_code:
        return _build_single_file_patch("main.py", best_code)
    
    return ""
