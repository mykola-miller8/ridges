import os
import re
import json
import uuid
import time
import subprocess
import tempfile
import ast
from typing import Any, Dict, List, Optional, Tuple

import requests


# v7 agent: generic code agent with test-driven iteration
# - Never embeds problem-specific constants or dataset names
# - Uses only the inference gateway exposed via INFERENCE_URL/SANDBOX_PROXY_URL
# - Returns a unified diff that replaces main.py entirely
# - When tests are available, uses test failures as feedback for refinement


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
    """Extract Python code from LLM response with multiple strategies."""
    if not response:
        return ""
    
    # Strategy 1: Explicitly labeled main.py block
    m = re.findall(r"```python\s*\n#\s*main\.py\n([\s\S]*?)\n```", response, re.DOTALL)
    if m and m[0].strip():
        return m[0].strip()
    
    # Strategy 2: First python block
    m2 = re.findall(r"```python\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    if m2 and m2[0].strip():
        return m2[0].strip()
    
    # Strategy 3: Code block without language tag
    m3 = re.findall(r"```\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    if m3 and m3[0].strip() and ("def " in m3[0] or "class " in m3[0]):
        return m3[0].strip()
    
    return ""


def _validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Check if Python code has valid syntax. Returns (is_valid, error_msg)."""
    if not code or not code.strip():
        return False, "Empty code"
    
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Parse error: {str(e)}"


def _run_tests(test_file: str = "tests.py", timeout: int = 30) -> Tuple[bool, str]:
    """Run test file and return (success, output). Returns (False, error) if tests fail."""
    if not os.path.exists(test_file):
        return False, f"Test file {test_file} not found"
    
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", test_file, "-v", "--tb=short", "--no-header", "-x"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        output = result.stdout + result.stderr
        
        if result.returncode == 0:
            return True, output
        else:
            # Extract failure information
            return False, output[:3000]  # Limit output length
            
    except subprocess.TimeoutExpired:
        return False, "Tests timed out"
    except Exception as e:
        return False, f"Test execution error: {str(e)}"


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

    # Compact repository summary for context
    parts: List[str] = []
    for name in ("main.py", "tests.py"):
        content = _read(name)
        if content:
            parts.append(f"### {name}\n```python\n{content[:8000]}\n```")
    summary = "\n\n".join(parts)

    # Build initial system message
    system_msg = (
        "You are a senior Python engineer who writes robust, correct code.\n"
        + ("Do not modify tests.py; only change main.py.\n" if mode == "tests_available" else "")
        + "Return ONLY one code block containing the complete main.py with a '# main.py' header.\n"
        "Format exactly as:\n```python\n# main.py\n[complete code]\n```\n\n"
        "CRITICAL: Pay careful attention to:\n"
        "- Edge cases and boundary conditions\n"
        "- Input validation and error handling\n"
        "- State management and tracking\n"
        "- All requirements in the problem statement\n"
        "No prose. Production-quality, deterministic code."
    )
    
    user_msg = (
        f"Problem Statement:\n{problem_statement[:12000]}\n\n"
        f"Repository Summary:\n{summary}\n\n"
        "Implement the solution carefully, ensuring all edge cases are handled correctly."
    )
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    best_code = None
    best_patch = None
    
    # Phase 1: Initial generation attempts across models
    for attempt in range(len(AGENT_MODELS)):
        try:
            resp = _call_llm(messages, run_id, attempt, 300)
            main_src = _extract_main_py(resp)
            
            if not main_src:
                continue
            
            # Validate syntax
            is_valid, error_msg = _validate_python_syntax(main_src)
            if not is_valid:
                continue
            
            # If we have valid code, save it as candidate
            if not best_code:
                best_code = main_src
                best_patch = _build_single_file_patch("main.py", main_src)
            
            # If tests available, try running them
            if mode == "tests_available" and os.path.exists("tests.py"):
                # Write the code to main.py temporarily
                original_main = _read("main.py")
                try:
                    with open("main.py", "w", encoding="utf-8") as f:
                        f.write(main_src)
                    
                    # Run tests
                    test_passed, test_output = _run_tests()
                    
                    if test_passed:
                        # Tests passed! Return immediately
                        return _build_single_file_patch("main.py", main_src)
                    
                    # Tests failed, save output for feedback
                    # Continue to next attempt
                    
                finally:
                    # Restore original main.py
                    with open("main.py", "w", encoding="utf-8") as f:
                        f.write(original_main)
            else:
                # No tests to run, return first valid code
                return best_patch
                
        except Exception:
            continue
    
    # Phase 2: If we have tests and initial attempts didn't pass, try refinement with feedback
    if mode == "tests_available" and os.path.exists("tests.py") and best_code:
        # Get test failure feedback
        original_main = _read("main.py")
        try:
            with open("main.py", "w", encoding="utf-8") as f:
                f.write(best_code)
            
            test_passed, test_output = _run_tests()
            
            if not test_passed:
                # Add test failure feedback to conversation
                refinement_msg = (
                    f"The previous implementation has test failures:\n\n"
                    f"```\n{test_output}\n```\n\n"
                    f"Please fix the implementation to pass all tests. "
                    f"Focus on the specific errors shown above. "
                    f"Return the complete corrected main.py in the same format."
                )
                
                messages.append({"role": "assistant", "content": f"```python\n# main.py\n{best_code}\n```"})
                messages.append({"role": "user", "content": refinement_msg})
                
                # Try refinement attempts with different models
                for attempt in range(len(AGENT_MODELS)):
                    try:
                        resp = _call_llm(messages, run_id, attempt + len(AGENT_MODELS), 300)
                        refined_src = _extract_main_py(resp)
                        
                        if not refined_src:
                            continue
                        
                        is_valid, _ = _validate_python_syntax(refined_src)
                        if not is_valid:
                            continue
                        
                        # Test refined code
                        with open("main.py", "w", encoding="utf-8") as f:
                            f.write(refined_src)
                        
                        test_passed, _ = _run_tests()
                        
                        if test_passed:
                            return _build_single_file_patch("main.py", refined_src)
                        
                        # Update best code if it's different
                        if refined_src != best_code:
                            best_code = refined_src
                            best_patch = _build_single_file_patch("main.py", refined_src)
                            
                    except Exception:
                        continue
                        
        finally:
            # Restore original main.py
            with open("main.py", "w", encoding="utf-8") as f:
                f.write(original_main)
    
    # Return best code found, or empty if nothing worked
    return best_patch or ""
