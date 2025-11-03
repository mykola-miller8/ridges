import os
import re
import json
import uuid
import time
import ast
import subprocess
import difflib
from typing import Any, Dict, List, Optional, Tuple


# v7 agent: Robust generic code agent with improved error handling and iteration
# - Generic solving mechanisms only (no problem-specific constants)
# - Uses inference gateway via INFERENCE_URL/SANDBOX_PROXY_URL
# - Returns unified diff patches with proper format
# - Syntax validation, iterative refinement, and comprehensive error handling


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

MAX_ITERATIONS = 3
MAX_MODEL_ATTEMPTS = 4


def _read(path: str) -> str:
    """Safely read file contents."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Validate Python syntax and return (is_valid, error_message)."""
    if not code or not code.strip():
        return False, "Empty code"
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Parse error: {str(e)}"


def _build_unified_diff(filename: str, old_content: str, new_content: str) -> str:
    """Build a proper unified diff using difflib."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    
    # Generate unified diff
    diff_lines = list(difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm=""
    ))
    
    if not diff_lines:
        # No changes, but still return a valid patch
        return ""
    
    return "".join(line if line.endswith("\n") else line + "\n" for line in diff_lines)


def _extract_code_from_response(response: str, target_file: str = "main.py") -> Optional[str]:
    """Extract code from LLM response using multiple strategies."""
    if not response or not response.strip():
        return None
    
    strategies = [
        # Strategy 1: Code block with file header comment
        lambda r: _extract_with_pattern(
            r, rf"```python\s*\n#\s*{re.escape(target_file)}\s*\n([\s\S]*?)\n```"
        ),
        # Strategy 2: Code block with filename in fence
        lambda r: _extract_with_pattern(
            r, rf"```python.*{re.escape(target_file)}.*\n([\s\S]*?)\n```"
        ),
        # Strategy 3: Any Python code block
        lambda r: _extract_with_pattern(
            r, r"```python\s*\n([\s\S]*?)\n```"
        ),
        # Strategy 4: Code block without language tag
        lambda r: _extract_with_pattern(
            r, r"```\s*\n([\s\S]*?)\n```"
        ),
        # Strategy 5: XML-style tags
        lambda r: _extract_with_pattern(
            r, r"<code>([\s\S]*?)</code>"
        ),
        # Strategy 6: Look for Python-like content (imports, def, class)
        lambda r: _extract_python_heuristic(r),
    ]
    
    for strategy in strategies:
        try:
            code = strategy(response)
            if code:
                code = code.strip()
                # Validate it's Python
                is_valid, _ = _validate_python_syntax(code)
                if is_valid:
                    return code
        except Exception:
            continue
    
    return None


def _extract_with_pattern(text: str, pattern: str) -> Optional[str]:
    """Extract code using regex pattern."""
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        # Return the longest match (likely the most complete)
        return max(matches, key=len) if matches else None
    return None


def _extract_python_heuristic(text: str) -> Optional[str]:
    """Try to extract Python code using heuristics."""
    # Look for lines that start typical Python files
    lines = text.split("\n")
    start_idx = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "def ", "class ", '"""', "'''")):
            start_idx = i
            break
    
    if start_idx is not None:
        # Take from start_idx to end, or until we hit non-code markers
        code_lines = []
        for line in lines[start_idx:]:
            if line.strip().startswith(("```", "Note:", "Here", "This code")):
                break
            code_lines.append(line)
        
        return "\n".join(code_lines).strip() if code_lines else None
    
    return None


def _gather_context(repo_dir: str, max_file_size: int = 12000) -> str:
    """Gather comprehensive repository context."""
    parts = []
    
    # Primary target file
    for filename in ["main.py", "tests.py", "test_main.py", "test.py"]:
        content = _read(filename)
        if content:
            truncated = content[:max_file_size]
            if len(content) > max_file_size:
                truncated += f"\n# ... ({len(content) - max_file_size} more chars truncated)"
            parts.append(f"### {filename}\n```python\n{truncated}\n```")
    
    # Look for additional context files
    context_files = ["README.md", "requirements.txt", "setup.py", "pyproject.toml"]
    for filename in context_files:
        if os.path.exists(filename):
            content = _read(filename)
            if content:
                truncated = content[:2000]
                parts.append(f"### {filename}\n```\n{truncated}\n```")
    
    # Directory structure (for context)
    try:
        tree_output = subprocess.check_output(
            ["find", ".", "-type", "f", "-name", "*.py", "-not", "-path", "*/.*"],
            stderr=subprocess.DEVNULL,
            timeout=5,
            text=True
        ).strip()
        if tree_output:
            py_files = tree_output.split("\n")[:20]  # Limit to first 20
            parts.append(f"### Python files in repository:\n{chr(10).join(py_files)}")
    except Exception:
        pass
    
    return "\n\n".join(parts)


def _run_tests(timeout: int = 30) -> Tuple[bool, str]:
    """Run tests and return (success, output)."""
    test_files = ["tests.py", "test_main.py", "test.py"]
    test_file = None
    
    for tf in test_files:
        if os.path.exists(tf):
            test_file = tf
            break
    
    if not test_file:
        return True, "No test file found"
    
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        return success, output
    except subprocess.TimeoutExpired:
        return False, "Tests timed out"
    except FileNotFoundError:
        # pytest not available, try unittest
        try:
            result = subprocess.run(
                ["python", "-m", "unittest", test_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            success = result.returncode == 0
            output = result.stdout + result.stderr
            return success, output
        except Exception as e:
            return False, f"Test execution failed: {str(e)}"
    except Exception as e:
        return False, f"Test execution failed: {str(e)}"


def _call_llm(
    messages: List[Dict[str, str]],
    run_id: str,
    model: str,
    timeout_s: int = 300,
    temperature: float = 0.0
) -> Optional[str]:
    """Call LLM with retry logic and exponential backoff."""
    url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
    headers = {"Content-Type": "application/json"}
    body = {
        "run_id": run_id,
        "messages": messages,
        "temperature": temperature,
        "agent_id": "agent-v7-robust",
        "model": model,
    }
    
    max_retries = 3
    for retry in range(max_retries):
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            
            # Extract content from various response formats
            if isinstance(data, dict):
                if "choices" in data and data["choices"]:
                    choice = data["choices"][0]
                    if isinstance(choice, dict):
                        message = choice.get("message", {})
                        if isinstance(message, dict):
                            content = message.get("content")
                            if content:
                                return str(content)
                        # Alternative structure
                        text = choice.get("text")
                        if text:
                            return str(text)
                # Direct content field
                if "content" in data:
                    return str(data["content"])
            
            if isinstance(data, str):
                return data
            
            # Last resort: stringify
            return json.dumps(data)
            
        except requests.exceptions.Timeout:
            if retry < max_retries - 1:
                time.sleep(2 ** retry)
                continue
            return None
        except requests.exceptions.RequestException as e:
            if retry < max_retries - 1:
                time.sleep(2 ** retry)
                continue
            return None
        except Exception as e:
            if retry < max_retries - 1:
                time.sleep(1)
                continue
            return None
    
    return None


def _generate_solution(
    problem_statement: str,
    context: str,
    run_id: str,
    model: str,
    has_tests: bool,
    previous_attempt: Optional[str] = None,
    test_output: Optional[str] = None
) -> Optional[str]:
    """Generate a solution using the LLM."""
    
    system_prompt = (
        "You are an expert Python engineer. Your task is to write clean, correct, working Python code.\n"
        "Critical requirements:\n"
        "- Return ONLY a complete, working Python code file\n"
        "- Format your response as a Python code block with '# main.py' as the first line\n"
        "- Use this exact format:\n"
        "```python\n"
        "# main.py\n"
        "[your complete implementation]\n"
        "```\n"
        "- Ensure the code is syntactically correct and will execute without errors\n"
        "- Do not include any explanatory text outside the code block\n"
    )
    
    if has_tests:
        system_prompt += "- The code must pass all provided tests\n"
        system_prompt += "- Do not modify any test files\n"
    
    system_prompt += "- Write deterministic, production-quality code\n"
    
    user_parts = [
        "# Problem Statement",
        problem_statement[:15000],  # Larger limit
        "",
        "# Repository Context",
        context,
    ]
    
    if previous_attempt and test_output:
        user_parts.extend([
            "",
            "# Previous Attempt (FAILED)",
            f"```python\n{previous_attempt[:3000]}\n```",
            "",
            "# Test Output (ERRORS)",
            test_output[:2000],
            "",
            "# Instructions",
            "The previous implementation failed. Analyze the errors and provide a corrected implementation.",
            "Focus on fixing the specific issues shown in the test output.",
        ])
    else:
        user_parts.extend([
            "",
            "# Instructions",
            "Implement a complete solution that satisfies the problem statement.",
        ])
        if has_tests:
            user_parts.append("Ensure all tests pass.")
    
    user_prompt = "\n".join(user_parts)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    # Use slightly higher temperature for retries to get different solutions
    temperature = 0.1 if previous_attempt else 0.0
    
    return _call_llm(messages, run_id, model, timeout_s=300, temperature=temperature)


def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False) -> str:
    """Entry point required by the evaluation harness.

    Args:
        input_dict: Contains problem_statement, run_id, and optional problem_category
        repo_dir: Directory containing the repository to work on
        test_mode: Whether running in test mode

    Returns:
        A unified diff patch string
    """
    run_id = (input_dict or {}).get("run_id", os.getenv("RUN_ID", str(uuid.uuid4())))
    
    # Change to repo directory
    if repo_dir and os.path.exists(repo_dir):
        try:
            os.chdir(repo_dir)
        except Exception:
            pass
    
    problem_statement = (input_dict or {}).get("problem_statement", "") or ""
    if not problem_statement:
        return ""
    
    # Detect mode based on test file presence
    has_tests = any(os.path.exists(f) for f in ["tests.py", "test_main.py", "test.py"])
    
    # Gather comprehensive context
    context = _gather_context(repo_dir)
    
    # Read original main.py for diff generation
    original_main = _read("main.py")
    
    # Try multiple models and iterations
    best_solution = None
    best_is_valid = False
    
    for model_idx in range(MAX_MODEL_ATTEMPTS):
        model = AGENT_MODELS[model_idx % len(AGENT_MODELS)]
        
        previous_attempt = None
        test_output = None
        
        for iteration in range(MAX_ITERATIONS):
            # Generate solution
            response = _generate_solution(
                problem_statement,
                context,
                run_id,
                model,
                has_tests,
                previous_attempt,
                test_output
            )
            
            if not response:
                break  # Try next model
            
            # Extract code from response
            code = _extract_code_from_response(response, "main.py")
            if not code:
                break  # Try next model
            
            # Validate syntax
            is_valid, error_msg = _validate_python_syntax(code)
            if not is_valid:
                previous_attempt = code
                test_output = f"Syntax validation failed: {error_msg}"
                continue  # Try again with error feedback
            
            # Store as potential solution
            if not best_is_valid:
                best_solution = code
                best_is_valid = True
            
            # If we have tests, run them
            if has_tests:
                # Write code to main.py temporarily
                try:
                    with open("main.py", "w", encoding="utf-8") as f:
                        f.write(code)
                    
                    # Run tests
                    tests_passed, test_out = _run_tests(timeout=30)
                    
                    if tests_passed:
                        # Success! Return immediately
                        diff = _build_unified_diff("main.py", original_main, code)
                        return diff if diff else _fallback_patch("main.py", code)
                    else:
                        # Tests failed, prepare for retry
                        previous_attempt = code
                        test_output = test_out
                        continue
                        
                except Exception as e:
                    # File writing or test execution failed
                    previous_attempt = code
                    test_output = f"Execution error: {str(e)}"
                    continue
                finally:
                    # Restore original if we modified it
                    if original_main:
                        try:
                            with open("main.py", "w", encoding="utf-8") as f:
                                f.write(original_main)
                        except Exception:
                            pass
            else:
                # No tests, return valid solution
                diff = _build_unified_diff("main.py", original_main, code)
                return diff if diff else _fallback_patch("main.py", code)
        
        # If we have a valid solution from this model, it's better than nothing
        if best_is_valid and best_solution:
            # Try next model to see if we can do better
            continue
    
    # Return best effort solution if we have one
    if best_solution:
        diff = _build_unified_diff("main.py", original_main, best_solution)
        return diff if diff else _fallback_patch("main.py", best_solution)
    
    return ""


def _fallback_patch(filename: str, new_content: str) -> str:
    """Generate a fallback patch when difflib fails."""
    old_content = _read(filename)
    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()
    
    lines = [
        f"diff --git a/{filename} b/{filename}",
        f"--- a/{filename}",
        f"+++ b/{filename}",
        f"@@ -1,{len(old_lines)} +1,{len(new_lines)} @@",
    ]
    
    # Remove all old lines
    for line in old_lines:
        lines.append(f"-{line}")
    
    # Add all new lines
    for line in new_lines:
        lines.append(f"+{line}")
    
    return "\n".join(lines) + "\n"
