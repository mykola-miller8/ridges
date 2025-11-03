import os
import re
import json
import uuid
import time
import ast
from typing import Any, Dict, List, Optional

import requests


# v7 agent: generic code agent with enhanced reasoning and validation
# - Never embeds problem-specific constants or dataset names
# - Uses only the inference gateway exposed via INFERENCE_URL/SANDBOX_PROXY_URL
# - Returns a unified diff that replaces main.py entirely
# - Includes syntax validation and self-review mechanisms


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


def _validate_syntax(code: str) -> Optional[str]:
    """Check if Python code is syntactically valid. Returns error message or None."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"
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
    if not response:
        return ""
    # Prefer explicitly headed block
    m = re.findall(r"```python\s*\n#\s*main\.py\n([\s\S]*?)\n```", response, re.DOTALL)
    if m and m[0].strip():
        return m[0].strip()
    # Try with optional whitespace around header
    m = re.findall(r"```python\s*\n#\s*main\.py\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    if m and m[0].strip():
        return m[0].strip()
    # Fallback: first python block
    m2 = re.findall(r"```python\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    return m2[0].strip() if m2 and m2[0].strip() else ""


def _call_llm(messages: List[Dict[str, str]], run_id: str, model_index: int, timeout_s: int = 300, temperature: float = 0.0) -> str:
    url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
    headers = {"Content-Type": "application/json"}
    model = AGENT_MODELS[model_index % len(AGENT_MODELS)]
    body = {
        "run_id": run_id,
        "messages": messages,
        "temperature": temperature,
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
    
    # Read main.py skeleton (only file available besides problem statement)
    main_py_skeleton = _read("main.py")
    
    # Enhanced system prompt emphasizing careful analysis and optimization
    system_msg = (
        "You are a senior software engineer specializing in algorithmic problem-solving.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Read the problem statement VERY carefully - every detail matters\n"
        "2. If the problem asks for 'optimal', 'best', 'maximum', or 'minimum' solutions, you MUST implement algorithms that consider multiple strategies\n"
        "3. Pay special attention to edge cases and examples in the problem statement\n"
        "4. For optimization problems, greedy algorithms often fail - consider dynamic programming, exhaustive search with pruning, or mathematical optimization\n"
        "5. Implement complete, working, syntactically correct Python code\n"
        "6. Include all necessary imports\n"
        "7. Return ONLY one code block with format:\n"
        "```python\n# main.py\n[complete code]\n```\n\n"
        "No explanations, just code."
    )
    
    # Step 1: Problem analysis and reasoning
    analysis_prompt = (
        f"Problem Statement:\n{problem_statement[:16000]}\n\n"
        f"Current main.py skeleton:\n```python\n{main_py_skeleton[:4000]}\n```\n\n"
        "Before implementing, analyze:\n"
        "1. What is the core problem asking for?\n"
        "2. Are there optimization requirements (best, optimal, minimum, maximum)?\n"
        "3. What algorithm approach is needed (greedy, dynamic programming, exhaustive search, etc.)?\n"
        "4. What are the edge cases?\n"
        "5. What examples are provided and what do they teach us?\n\n"
        "Provide a brief analysis (2-3 paragraphs) focusing on the algorithmic approach."
    )
    
    analysis_messages = [
        {"role": "system", "content": "You are an expert algorithm designer. Analyze problems carefully before implementing."},
        {"role": "user", "content": analysis_prompt}
    ]
    
    # Step 2: Implementation with reasoning context
    best_code = None
    best_model = 0
    
    # Try multiple models with reasoning
    for attempt in range(len(AGENT_MODELS)):
        try:
            # Get analysis (helps prime the model's reasoning)
            analysis = ""
            try:
                analysis = _call_llm(analysis_messages, run_id, attempt, 180, 0.3)
            except Exception:
                analysis = ""  # Continue without analysis if it fails
            
            # Generate implementation
            impl_prompt = (
                f"Problem Statement:\n{problem_statement[:16000]}\n\n"
                f"Current main.py skeleton:\n```python\n{main_py_skeleton[:4000]}\n```\n\n"
            )
            
            if analysis:
                impl_prompt += f"Analysis:\n{analysis[:2000]}\n\n"
            
            impl_prompt += (
                "Now implement the complete solution in main.py.\n"
                "Remember:\n"
                "- If the problem requires optimization, implement a thorough algorithm\n"
                "- Handle all edge cases mentioned in the problem\n"
                "- Ensure the solution is complete and syntactically correct\n"
                "- Include all necessary imports\n\n"
                "Return ONLY the code block in this exact format:\n"
                "```python\n# main.py\n[complete implementation]\n```"
            )
            
            impl_messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": impl_prompt}
            ]
            
            resp = _call_llm(impl_messages, run_id, attempt, 300, 0.0)
            main_src = _extract_main_py(resp)
            
            if not main_src:
                continue
            
            # Validate syntax
            syntax_error = _validate_syntax(main_src)
            if syntax_error:
                # Try to fix with feedback
                fix_prompt = (
                    f"The previous implementation has a syntax error:\n{syntax_error}\n\n"
                    f"Previous code:\n```python\n{main_src[:4000]}\n```\n\n"
                    "Fix the syntax error and return the corrected complete main.py:\n"
                    "```python\n# main.py\n[corrected code]\n```"
                )
                
                impl_messages.append({"role": "assistant", "content": resp})
                impl_messages.append({"role": "user", "content": fix_prompt})
                
                fix_resp = _call_llm(impl_messages, run_id, attempt, 240, 0.0)
                fixed_src = _extract_main_py(fix_resp)
                
                if fixed_src and _validate_syntax(fixed_src) is None:
                    main_src = fixed_src
                else:
                    continue  # Skip this attempt if unfixable
            
            # We have valid code
            if best_code is None:
                best_code = main_src
                best_model = attempt
                # Don't break - try other models too for better solutions
            
            # For the first valid solution, also try a review pass
            if attempt == 0:
                review_prompt = (
                    f"Review this implementation for the problem:\n\n"
                    f"Problem: {problem_statement[:8000]}\n\n"
                    f"Implementation:\n```python\n{main_src[:6000]}\n```\n\n"
                    "Check:\n"
                    "1. Does it handle ALL cases including edge cases?\n"
                    "2. For optimization problems, does it find the truly optimal solution?\n"
                    "3. Are there any logical errors?\n\n"
                    "If improvements are needed, return the improved complete code:\n"
                    "```python\n# main.py\n[improved code]\n```\n\n"
                    "If it's already correct, return the same code."
                )
                
                review_messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": review_prompt}
                ]
                
                try:
                    review_resp = _call_llm(review_messages, run_id, attempt, 240, 0.1)
                    reviewed_src = _extract_main_py(review_resp)
                    
                    if reviewed_src and _validate_syntax(reviewed_src) is None:
                        best_code = reviewed_src
                except Exception:
                    pass  # Keep original if review fails
            
        except Exception:
            continue
    
    # Return best valid code found
    if best_code:
        return _build_single_file_patch("main.py", best_code)
    
    return ""


