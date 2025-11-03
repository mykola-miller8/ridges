import os
import re
import ast
import json
import uuid
import time
from typing import Any, Dict, List, Optional

import requests


# v7 seed agent: generic code agent that proposes a full-file patch for main.py only.
# - Never embeds problem-specific constants or dataset names
# - Uses only the inference gateway exposed via INFERENCE_URL/SANDBOX_PROXY_URL
# - Returns a unified diff that replaces main.py entirely
# - Includes example-based validation and simplicity-focused approach


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
    """Read a file and return its contents, or empty string if not found."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _validate_syntax(code: str) -> Optional[str]:
    """
    Validate Python syntax of the given code.
    Returns None if valid, or an error message if invalid.
    """
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return f"Parse error: {str(e)}"


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
    """Extract Python code from LLM response, preferring blocks with '# main.py' header."""
    if not response:
        return ""
    # Prefer explicitly headed block
    m = re.findall(r"```python\s*\n#\s*main\.py\n([\s\S]*?)\n```", response, re.DOTALL)
    if m and m[0].strip():
        return m[0].strip()
    # Fallback: first python block
    m2 = re.findall(r"```python\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    return m2[0].strip() if m2 and m2[0].strip() else ""


def _call_llm(messages: List[Dict[str, str]], run_id: str, attempt: int, timeout_s: int = 240) -> str:
    """Call the inference gateway with retry logic."""
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


def _extract_examples_from_problem(problem_statement: str) -> str:
    """Extract example sections from problem statement for validation."""
    # Look for common example markers
    examples = []
    lines = problem_statement.split('\n')
    in_example = False
    current_example = []
    
    for i, line in enumerate(lines):
        lower = line.lower()
        # Start of example section
        if any(marker in lower for marker in ['example', 'for example', 'here is', 'consider']):
            in_example = True
            current_example = [line]
        elif in_example:
            # Continue collecting example lines
            current_example.append(line)
            # End example after some lines or when hitting new section
            if len(current_example) > 15 or (line and not line.startswith(' ') and ':' in line):
                examples.append('\n'.join(current_example))
                current_example = []
                in_example = False
    
    if current_example:
        examples.append('\n'.join(current_example))
    
    return '\n\n'.join(examples[:3]) if examples else ""  # Return up to 3 examples


def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False) -> str:
    """
    Entry point required by the evaluation harness.
    
    Generates a complete solution with example-based validation:
    1. Generate simple, correct solution emphasizing clarity
    2. Validate syntax
    3. Review by tracing through examples from problem statement
    4. Return unified diff patch that fully replaces main.py
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

    # Extract examples from problem for validation
    examples = _extract_examples_from_problem(problem_statement)

    # Initial system message emphasizing simplicity and correctness
    system_msg = (
        "You are a senior Python engineer who writes SIMPLE, CORRECT code.\n"
        + ("Do not modify tests.py; only change main.py.\n" if mode == "tests_available" else "")
        + "CRITICAL: Prefer simple, straightforward solutions over complex clever ones.\n"
        "Return ONLY one code block containing the complete main.py with a '# main.py' header.\n"
        "Format exactly as:\n```python\n# main.py\n[complete code]\n```\n"
        "No prose. Focus on correctness and clarity."
    )
    
    user_msg = (
        f"Problem Statement:\n{problem_statement[:12000]}\n\n"
        f"Repository Summary:\n{summary}\n\n"
        "Implement a complete, correct solution.\n"
        "- Use the SIMPLEST approach that correctly solves the problem\n"
        "- Handle all edge cases and boundary conditions\n"
        "- Write clear, readable code that obviously works"
    )

    # Try multiple models to generate initial solution
    best_code = ""
    best_attempt = 0
    
    for attempt in range(len(AGENT_MODELS)):
        try:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            resp = _call_llm(messages, run_id, attempt, 300)
            main_src = _extract_main_py(resp)
            
            if not main_src:
                continue
                
            # Validate syntax
            syntax_err = _validate_syntax(main_src)
            if syntax_err:
                continue  # Try next model
                
            best_code = main_src
            best_attempt = attempt
            break
            
        except Exception:
            continue

    if not best_code:
        return ""

    # Example-based validation review
    try:
        review_system = (
            "You are an expert code reviewer who validates correctness by tracing examples.\n"
            "Your task: manually trace through the code with concrete examples to find bugs.\n"
            "If the code is CORRECT for all examples, respond ONLY with: APPROVED\n"
            "If you find bugs, provide CORRECTED code:\n```python\n# main.py\n[corrected code]\n```"
        )
        
        examples_section = f"\n\nExamples from problem:\n{examples}\n" if examples else ""
        
        review_user = (
            f"Problem:\n{problem_statement[:8000]}\n{examples_section}\n"
            f"Code to validate:\n```python\n{best_code}\n```\n\n"
            "Manually trace through the code with the examples above:\n"
            "1. Walk through the logic step-by-step with a concrete example\n"
            "2. Check if the output matches the expected result\n"
            "3. Verify edge cases: empty input, single element, boundary values\n"
            "4. Look for off-by-one errors, incorrect loop conditions, wrong comparisons\n\n"
            "Does the code produce correct results for ALL examples?"
        )
        
        review_messages = [
            {"role": "system", "content": review_system},
            {"role": "user", "content": review_user},
        ]
        
        review_resp = _call_llm(review_messages, run_id, best_attempt, 300)
        
        # Check if reviewer approved or provided corrections
        if "APPROVED" not in review_resp.upper():
            refined_code = _extract_main_py(review_resp)
            if refined_code:
                # Validate refined code syntax
                syntax_err = _validate_syntax(refined_code)
                if not syntax_err:
                    best_code = refined_code
    
    except Exception:
        # If review fails, continue with best_code from initial generation
        pass

    return _build_single_file_patch("main.py", best_code)
