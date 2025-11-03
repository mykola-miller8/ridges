import os
import re
import ast
import json
import uuid
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

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


def _verbose_log(step: str, details: Any = None, level: str = "INFO") -> None:
    """Log verbose information for evaluation and evolution tracking."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_line = f"[V7_VERBOSE:{level}] [{timestamp}] {step}"
    if details is not None:
        if isinstance(details, (dict, list)):
            details_str = json.dumps(details, indent=2, default=str)
            # Truncate very long details
            if len(details_str) > 2000:
                details_str = details_str[:2000] + "... [truncated]"
            log_line += f"\n{details_str}"
        else:
            details_str = str(details)
            if len(details_str) > 1000:
                details_str = details_str[:1000] + "... [truncated]"
            log_line += f": {details_str}"
    print(log_line, flush=True)


def _read(path: str) -> str:
    """Read file contents, returning empty string on error."""
    _verbose_log("FILE_READ: Attempting to read file", {"path": path})
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            _verbose_log("FILE_READ: Successfully read file", {
                "path": path,
                "size_bytes": len(content),
                "size_lines": len(content.splitlines()),
                "preview": content[:200] + "..." if len(content) > 200 else content
            })
            return content
    except Exception as e:
        _verbose_log("FILE_READ: Failed to read file", {
            "path": path,
            "error": str(e)
        }, level="WARN")
        return ""


def _validate_syntax(code: str) -> Tuple[bool, str]:
    """
    Check if code has valid Python syntax.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    _verbose_log("SYNTAX_VALIDATION: Starting syntax validation", {
        "code_length": len(code),
        "code_lines": len(code.splitlines())
    })
    try:
        ast.parse(code)
        _verbose_log("SYNTAX_VALIDATION: Syntax is valid")
        return True, ""
    except SyntaxError as e:
        error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
        _verbose_log("SYNTAX_VALIDATION: Syntax error found", {
            "line": e.lineno,
            "message": e.msg,
            "error": error_msg
        }, level="ERROR")
        return False, error_msg
    except Exception as e:
        error_msg = str(e)
        _verbose_log("SYNTAX_VALIDATION: Validation exception", {
            "error_type": type(e).__name__,
            "error": error_msg
        }, level="ERROR")
        return False, error_msg


def _build_single_file_patch(filename: str, new_content: str) -> str:
    """Build unified diff patch replacing file contents."""
    _verbose_log("PATCH_BUILD: Building patch", {"filename": filename})
    old = _read(filename)
    old_lines = old.splitlines()
    new_lines = new_content.splitlines()
    _verbose_log("PATCH_BUILD: File comparison", {
        "old_lines": len(old_lines),
        "new_lines": len(new_lines),
        "old_size": len(old),
        "new_size": len(new_content)
    })
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
    patch = "\n".join(header + body) + "\n"
    _verbose_log("PATCH_BUILD: Patch built successfully", {
        "patch_size": len(patch),
        "patch_lines": len(patch.splitlines()),
        "preview": patch[:500] + "..." if len(patch) > 500 else patch
    })
    return patch


def _extract_main_py(response: str) -> str:
    """Extract Python code from LLM response using multiple strategies."""
    _verbose_log("CODE_EXTRACTION: Starting code extraction", {
        "response_length": len(response),
        "response_preview": response[:300] + "..." if len(response) > 300 else response
    })
    if not response:
        _verbose_log("CODE_EXTRACTION: Empty response, returning empty", level="WARN")
        return ""
    
    # Strategy 1: Explicitly headed block with main.py comment
    m = re.findall(r"```python\s*\n#\s*main\.py\n([\s\S]*?)\n```", response, re.DOTALL)
    if m and m[0].strip():
        extracted = m[0].strip()
        _verbose_log("CODE_EXTRACTION: Strategy 1 succeeded (explicit main.py)", {
            "extracted_length": len(extracted),
            "extracted_lines": len(extracted.splitlines()),
            "preview": extracted[:200] + "..." if len(extracted) > 200 else extracted
        })
        return extracted
    
    # Strategy 2: Any python code block
    m2 = re.findall(r"```python\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    if m2:
        for block in m2:
            if block.strip():
                extracted = block.strip()
                _verbose_log("CODE_EXTRACTION: Strategy 2 succeeded (python block)", {
                    "extracted_length": len(extracted),
                    "extracted_lines": len(extracted.splitlines()),
                    "preview": extracted[:200] + "..." if len(extracted) > 200 else extracted
                })
                return extracted
    
    # Strategy 3: Code block without language specifier
    m3 = re.findall(r"```\n([\s\S]*?)\n```", response, re.DOTALL)
    if m3:
        for block in m3:
            # Check if it looks like Python code (has def, class, or import)
            if block.strip() and any(keyword in block for keyword in ['def ', 'class ', 'import ']):
                extracted = block.strip()
                _verbose_log("CODE_EXTRACTION: Strategy 3 succeeded (generic code block)", {
                    "extracted_length": len(extracted),
                    "extracted_lines": len(extracted.splitlines()),
                    "preview": extracted[:200] + "..." if len(extracted) > 200 else extracted
                })
                return extracted
    
    _verbose_log("CODE_EXTRACTION: All strategies failed, no code extracted", {
        "response_length": len(response),
        "strategies_tried": 3
    }, level="WARN")
    return ""


def _call_llm(
    messages: List[Dict[str, str]], 
    run_id: str, 
    model_idx: int,
    temperature: float = 0.0
) -> str:
    """Call inference gateway with specified model and parameters."""
    model = AGENT_MODELS[model_idx % len(AGENT_MODELS)]
    url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
    _verbose_log("LLM_CALL: Preparing LLM request", {
        "model": model,
        "model_idx": model_idx,
        "run_id": run_id,
        "temperature": temperature,
        "url": url,
        "message_count": len(messages),
        "total_chars": sum(len(m.get("content", "")) for m in messages)
    })
    
    headers = {"Content-Type": "application/json"}
    body = {
        "run_id": run_id,
        "messages": messages,
        "temperature": temperature,
        "agent_id": "agent-v7",
        "model": model,
    }
    
    for retry in range(3):
        _verbose_log("LLM_CALL: Attempting request", {
            "retry": retry + 1,
            "max_retries": 3,
            "model": model
        })
        call_start_time = time.time()
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=300)
            call_duration = time.time() - call_start_time
            resp.raise_for_status()
            data = resp.json()
            
            _verbose_log("LLM_CALL: Request successful", {
                "retry": retry + 1,
                "duration_seconds": round(call_duration, 2),
                "status_code": resp.status_code,
                "response_type": type(data).__name__
            })
            
            if isinstance(data, dict) and data.get("choices"):
                content = (data["choices"][0].get("message", {}) or {}).get("content") or ""
                _verbose_log("LLM_CALL: Response extracted from choices", {
                    "content_length": len(content),
                    "content_preview": content[:300] + "..." if len(content) > 300 else content
                })
                return content
            if isinstance(data, str):
                _verbose_log("LLM_CALL: Response is string", {
                    "content_length": len(data),
                    "content_preview": data[:300] + "..." if len(data) > 300 else data
                })
                return data
            _verbose_log("LLM_CALL: Converting response to JSON string", {
                "data_keys": list(data.keys()) if isinstance(data, dict) else "non-dict"
            })
            return json.dumps(data)
        except Exception as e:
            call_duration = time.time() - call_start_time
            _verbose_log("LLM_CALL: Request failed", {
                "retry": retry + 1,
                "duration_seconds": round(call_duration, 2),
                "error_type": type(e).__name__,
                "error": str(e)
            }, level="ERROR")
            if retry == 2:
                _verbose_log("LLM_CALL: Max retries reached, raising exception", level="ERROR")
                raise
            sleep_time = 1 + retry
            _verbose_log("LLM_CALL: Retrying after delay", {"sleep_seconds": sleep_time})
            time.sleep(sleep_time)
    _verbose_log("LLM_CALL: All retries exhausted, returning empty", level="ERROR")
    return ""


def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False) -> str:
    """
    Main entry point for the evaluation harness.
    Generates a solution and returns a unified diff patch for main.py.
    
    Args:
        input_dict: Dictionary containing problem_statement, run_id, and other metadata
        repo_dir: Directory containing the problem files (main.py, tests.py, etc.)
        test_mode: Whether running in test mode (unused, for compatibility)
    
    Returns:
        Unified diff patch string for main.py, or empty string on failure
    """
    agent_start_time = time.time()
    _verbose_log("AGENT_MAIN: Starting agent execution", {
        "repo_dir": repo_dir,
        "test_mode": test_mode,
        "input_dict_keys": list(input_dict.keys()) if input_dict else []
    })
    
    run_id = (input_dict or {}).get("run_id", os.getenv("RUN_ID", str(uuid.uuid4())))
    _verbose_log("AGENT_MAIN: Run ID determined", {"run_id": run_id})

    if repo_dir and os.path.exists(repo_dir):
        try:
            old_cwd = os.getcwd()
            os.chdir(repo_dir)
            new_cwd = os.getcwd()
            _verbose_log("AGENT_MAIN: Changed working directory", {
                "old_cwd": old_cwd,
                "new_cwd": new_cwd,
                "repo_dir": repo_dir
            })
        except Exception as e:
            _verbose_log("AGENT_MAIN: Failed to change directory", {
                "repo_dir": repo_dir,
                "error": str(e)
            }, level="WARN")

    problem_statement = (input_dict or {}).get("problem_statement", "") or ""
    _verbose_log("AGENT_MAIN: Problem statement loaded", {
        "statement_length": len(problem_statement),
        "statement_preview": problem_statement[:300] + "..." if len(problem_statement) > 300 else problem_statement
    })
    
    mode = (input_dict or {}).get("problem_category", None)
    if mode not in ("spec_only", "tests_available"):
        mode = "tests_available" if os.path.exists("tests.py") else "spec_only"
    _verbose_log("AGENT_MAIN: Mode determined", {
        "mode": mode,
        "tests_available": os.path.exists("tests.py") if mode == "tests_available" else False
    })

    # Build repository context with larger context window
    _verbose_log("AGENT_MAIN: Building repository context")
    parts: List[str] = []
    for name in ("main.py", "tests.py"):
        content = _read(name)
        if content:
            # Increased from 8000 to 15000 for better context
            truncated = content[:15000]
            parts.append(f"### {name}\n```python\n{truncated}\n```")
            _verbose_log("AGENT_MAIN: File added to context", {
                "file": name,
                "original_size": len(content),
                "truncated_size": len(truncated),
                "was_truncated": len(content) > 15000
            })
        else:
            _verbose_log("AGENT_MAIN: File not found or empty", {"file": name}, level="WARN")
    repo_summary = "\n\n".join(parts)
    _verbose_log("AGENT_MAIN: Repository context built", {
        "summary_length": len(repo_summary),
        "files_included": len(parts)
    })

    system_msg = (
        "You are an expert Python engineer. Your task is to write production-quality, "
        "bug-free Python code that correctly implements the given specification.\n\n"
        "CRITICAL REQUIREMENTS:\n"
        "- Follow the specification EXACTLY - do not deviate or add assumptions\n"
        "- If error messages are specified, use them VERBATIM (exact wording)\n"
        "- If validation is required, implement ALL validation cases completely\n"
        "- Pay attention to type signatures, constants, and structure definitions\n"
        + ("- IMPORTANT: Only modify main.py. Do not change tests.py.\n" if mode == "tests_available" else "")
        + "\nReturn your solution in this exact format:\n"
        "```python\n# main.py\n<your complete implementation here>\n```"
    )
    
    user_msg = f"""# Problem Statement

{problem_statement[:15000]}

# Repository Files

{repo_summary}

# IMPLEMENTATION GUIDELINES

## 0. READ THE SPECIFICATION CAREFULLY - EXACT OUTPUT MATTERS
**Many problems require EXACT string matching or specific output format.**

### When the spec shows example output:
1. **Study the examples CHARACTER BY CHARACTER**
   - Exact punctuation: commas, periods, colons, quotes
   - Exact capitalization: "Ten" not "ten", "The" not "the"
   - Exact spacing: single spaces, no trailing spaces
   - Line structure: where do lines break?

2. **Identify patterns in the examples**
   - Pluralization rules: "bottle" vs "bottles", "is" vs "are"
   - Number representations: numeric (1, 2, 3) vs words ("one", "two", "three")
   - Special cases in wording: "no items" vs "zero items", "a" vs "an"
   - Formatting patterns: separators between items, empty lines between blocks

3. **Match the return type format PRECISELY**
   - `list[str]`: Each element is ONE complete line (no embedded \\n characters)
   - `str`: May contain newlines (\\n) to separate lines
   - Empty strings in a list often represent blank lines

4. **For string generation tasks:**
   - Build helper functions/dictionaries for conversions (numbers to words, etc.)
   - Handle singular/plural forms explicitly
   - Test edge cases: 0, 1, 2, boundary values
   - Don't guess at formatting - use the examples as ground truth

5. **CRITICAL: EXACT ERROR MESSAGES**
   - **If the spec explicitly provides error/exception messages, COPY THEM EXACTLY**
   - Look for sections like "Exception messages", "Error handling", "raise statement"
   - Common patterns: `raise TypeError("exact message here")` or `raise ValueError("exact message")`
   - **DO NOT paraphrase or reword specified error messages**
   - Tests often verify both the exception type AND the exact message text
   - Example: If spec says `raise ValueError("Edge is malformed")`, use that EXACT string
   - Search for keywords: "raise", "exception", "error message", "message text"

## 1. SIMPLICITY IS CRITICAL
**The #1 cause of bugs is unnecessary complexity.**
- Use the MINIMUM state variables needed (each extra variable = exponentially more bugs)
- Prefer simple, direct logic over clever abstractions
- Before coding, ask: "What's the SIMPLEST approach that could work?"
- For stateful classes: Track only what you absolutely need
- Avoid redundant or derived state (don't store what you can compute)
- **Input parsing**: Don't over-engineer! Whitespace/formatting is often just visual
  - Try the simplest approach: remove spaces, split lines, strip
  - Complex parsing logic = more bugs
  - If simple parsing works, don't build elaborate parsers

## 2. TRACE YOUR LOGIC BEFORE CODING
**Test your approach mentally with concrete examples BEFORE writing code.**
- Pick 2-3 test cases from the spec and trace through your planned logic
- Check: start state ? middle state ? end state
- Verify boundary conditions: first item, last item, empty input
- Watch for off-by-one errors in loops and indices
- If you can't trace it easily in your head, it's too complex!

## 3. IDENTIFY SPECIAL CASES
**Scan the ENTIRE specification for exceptions to general rules:**
- Keywords: "special", "except", "however", "note that", "but", "otherwise"
- Positional: "first", "last", "final", "10th", specific indices
- Conditional: "if X then Y", "only when", "unless"
- Extra handling: "bonus", "fill", "additional"

**Implement special cases EXPLICITLY - don't assume a uniform loop will handle them!**

Examples:
- "The 10th frame is special" ? Handle frame 10 separately with different logic
- "Except for the last element" ? Process n-1 items in loop, then handle last specially
- "Bonus rolls if spare/strike" ? Add conditional logic after main processing

## 4. INPUT VALIDATION AND ERROR HANDLING
**Many problems require strict input validation with specific error types and messages.**

### Validation strategy:
1. **Check the specification for validation requirements**
   - Look for: "raise", "exception", "error", "invalid", "malformed"
   - Note EXACT error types: TypeError, ValueError, etc.
   - Note EXACT error messages if provided
   - **TypeError vs ValueError distinction**:
     - TypeError: Wrong input type or structural problems (not a list, empty tuple, missing elements)
     - ValueError: Wrong values or content problems (invalid constant, wrong field types, out of range)

2. **Validate in the correct order (fail fast principle)**
   - Type checks first (is it the right type? list vs dict vs str vs int)
   - Structure checks next (right length? right format?)
   - Content checks last (valid values? constraints met?)

3. **Common validation patterns**
   - **Type validation**: `if not isinstance(data, expected_type): raise TypeError("...")`
   - **Structure validation**: Check tuple length, dict keys, list elements
     - **For DSLs with tuple-based syntax**: Each tuple type has an expected length
     - Example: If spec shows `(TYPE, arg1, arg2)`, tuples must have exactly 3 elements
     - Check: `if len(item) != expected_length: raise ...("... incomplete/malformed")`
   - **Content validation**: Check value ranges, string formats, relationships
   - **Completeness validation**: Check for missing required fields

4. **Be explicit about what's wrong**
   - If spec provides exact messages, use them verbatim
   - Otherwise, make messages descriptive but consistent with the spec's tone

5. **DSL and parser problems - special attention**
   - **Look for constant definitions** at the top of main.py (e.g., `NODE, EDGE, ATTR = range(3)`)
   - **These constants identify different data types** in the input
   - **Each type usually has a specific structure**: (TYPE_CONSTANT, ...required args...)
   - **Validation must check**:
     1. Is the type constant valid/recognized?
     2. Does the tuple have the right number of elements for that type?
     3. Are the element types correct (str, dict, int, etc.)?
   - **Read all examples in the spec** to understand the expected structure for each type

### State management for classes:
- **Validate preconditions** in all state-modifying methods
  - Check: Is this operation allowed in the current state?
  - **CRITICAL**: Before accepting input, verify the operation is still valid
  - Example: game.roll() must check if game is complete and reject if so
  - Raise clear exceptions when preconditions fail
- **Track completion for fixed-length games/processes**
  - If there's a fixed number of rounds/frames/steps, track progress
  - Check "are we done?" before accepting more input
  - **Variable-length final rounds**: Some games have special last rounds (e.g., bonus balls)
    - Track the round number AND what makes that round complete
    - Don't just count rolls - check logical completion conditions
- **Test state transitions**: mentally trace start ? middle ? end
- **Avoid state redundancy**: Don't track the same information multiple ways

## 5. COMMON PITFALLS TO AVOID
- **Off-by-one errors**: Double-check loop bounds and index arithmetic
  - Is it `range(n)` or `range(n+1)`?
  - Is frame 10 at index 9 or 10?
- **Loop safety**: In `while` loops with `continue`, increment BEFORE continue
  - Wrong: `while i < n: if cond: continue` (infinite loop!)
  - Right: `while i < n: if cond: i += 1; continue`
- **Validation**: Use correct checks for edge cases
  - `len(item) < 2` catches empty AND single-element
  - `len(item) < 1` only catches empty
- **Return types**: If returning `list[str]`, each element is ONE line (not multi-line with \\n)
- **String formatting**: Don't add extra spaces, newlines, or punctuation not in the spec
- **Grid/graph adjacency and connectivity problems**:
  - **Don't assume standard adjacency**: Not all grids use 4-way or 8-way adjacency
  - **Parse visual layout carefully**: Indentation and spacing often encode the structure
    - If rows are indented differently, this usually indicates a hex/diamond grid
    - The visual spacing shows which cells are actually adjacent
  - **Verify adjacency with examples**: Trace through the examples to understand which cells connect
  - **For hex grids**: 6 neighbors (not 4 or 8), and diagonals are often NOT valid connections
  - **Parsing strategy**: Often the simplest approach is to normalize the input
    - Remove leading spaces from each row to get the logical grid
    - The column index in the parsed grid corresponds to the cell's position
    - Don't overthink row offset patterns - test with examples first
  - **For connectivity/path-finding**: Use proper BFS/DFS, but FIRST ensure adjacency is correct
  - **Test your adjacency function**: Mentally verify neighbors for a cell in the middle of the grid

## 6. EDGE CASES CHECKLIST
Always handle:
- Empty inputs (empty strings, empty lists, zero values)
- Single-element collections (often requires singular forms)
- First element in a sequence
- Last element in a sequence  
- Boundary values (min, max)
- Initial state before any operations
- **Zero vs "no"**: Check if 0 should be "zero", "no", or something else

## 7. FINAL VERIFICATION
Before submitting, mentally trace through your code:
1. Does it handle the basic/normal case?
2. Does it handle ALL special cases mentioned in the spec?
3. Did I match the example output EXACTLY (punctuation, capitalization, spacing)?
4. **If error messages are specified**: Did I use the EXACT error messages verbatim?
5. **If validation is required**: Did I validate input type, structure, and content correctly?
6. Did I use the minimum state needed?
7. Can I explain the logic in 2-3 simple sentences?
8. Are there off-by-one errors in my indexing?
9. Do all state-modifying methods validate preconditions?
10. **For games/processes with fixed length**: Does it prevent operations after completion?
11. **For string generation**: Did I test singular/plural forms and edge cases?

**If you can't clearly trace the logic, simplify it!**

Provide your complete implementation now."""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    
    _verbose_log("AGENT_MAIN: Prompt messages built", {
        "system_msg_length": len(system_msg),
        "user_msg_length": len(user_msg),
        "total_length": len(system_msg) + len(user_msg),
        "message_count": len(messages)
    })

    # Try each model with iterative refinement
    _verbose_log("AGENT_MAIN: Starting model attempts", {
        "total_models": len(AGENT_MODELS),
        "models": AGENT_MODELS
    })
    
    for model_idx in range(len(AGENT_MODELS)):
        model = AGENT_MODELS[model_idx]
        _verbose_log("AGENT_MAIN: Starting model attempt", {
            "model_idx": model_idx,
            "model": model,
            "attempt": model_idx + 1,
            "total": len(AGENT_MODELS)
        })
        model_start_time = time.time()
        try:
            # First attempt
            _verbose_log("AGENT_MAIN: First LLM call", {"model": model})
            response = _call_llm(messages, run_id, model_idx)
            response_duration = time.time() - model_start_time
            _verbose_log("AGENT_MAIN: First LLM response received", {
                "model": model,
                "response_length": len(response),
                "duration_seconds": round(response_duration, 2)
            })
            
            code = _extract_main_py(response)
            
            if not code:
                _verbose_log("AGENT_MAIN: No code extracted, trying next model", {
                    "model": model,
                    "response_length": len(response)
                }, level="WARN")
                continue
            
            _verbose_log("AGENT_MAIN: Code extracted, validating syntax", {
                "model": model,
                "code_length": len(code),
                "code_lines": len(code.splitlines())
            })
            
            # Validate syntax
            is_valid, error_msg = _validate_syntax(code)
            
            if is_valid:
                _verbose_log("AGENT_MAIN: Syntax valid, building patch", {
                    "model": model,
                    "code_length": len(code)
                })
                # Success - return the patch
                patch = _build_single_file_patch("main.py", code)
                total_duration = time.time() - agent_start_time
                _verbose_log("AGENT_MAIN: Agent execution successful", {
                    "model": model,
                    "model_idx": model_idx,
                    "total_duration_seconds": round(total_duration, 2),
                    "patch_size": len(patch)
                })
                return patch
            
            _verbose_log("AGENT_MAIN: Syntax error detected, attempting refinement", {
                "model": model,
                "error": error_msg,
                "code_length": len(code)
            }, level="WARN")
            
            # If syntax error, try one refinement attempt with this model
            refinement_msg = {
                "role": "user",
                "content": f"""The code has a syntax error:

{error_msg}

Please fix the error and provide the corrected complete implementation of main.py.
Return it in the same format:
```python
# main.py
<corrected implementation>
```"""
            }
            
            refined_messages = messages + [
                {"role": "assistant", "content": response},
                refinement_msg
            ]
            
            _verbose_log("AGENT_MAIN: Calling LLM for refinement", {
                "model": model,
                "refinement_context_length": len(error_msg)
            })
            refinement_start_time = time.time()
            refined_response = _call_llm(refined_messages, run_id, model_idx)
            refinement_duration = time.time() - refinement_start_time
            _verbose_log("AGENT_MAIN: Refinement response received", {
                "model": model,
                "response_length": len(refined_response),
                "duration_seconds": round(refinement_duration, 2)
            })
            
            refined_code = _extract_main_py(refined_response)
            
            if refined_code:
                _verbose_log("AGENT_MAIN: Refined code extracted, validating", {
                    "model": model,
                    "refined_code_length": len(refined_code)
                })
                is_valid_refined, refined_error_msg = _validate_syntax(refined_code)
                if is_valid_refined:
                    _verbose_log("AGENT_MAIN: Refined code syntax valid, building patch", {
                        "model": model,
                        "refined_code_length": len(refined_code)
                    })
                    patch = _build_single_file_patch("main.py", refined_code)
                    total_duration = time.time() - agent_start_time
                    _verbose_log("AGENT_MAIN: Agent execution successful after refinement", {
                        "model": model,
                        "model_idx": model_idx,
                        "total_duration_seconds": round(total_duration, 2),
                        "patch_size": len(patch)
                    })
                    return patch
                else:
                    _verbose_log("AGENT_MAIN: Refined code still has syntax errors", {
                        "model": model,
                        "error": refined_error_msg
                    }, level="ERROR")
            else:
                _verbose_log("AGENT_MAIN: No code extracted from refinement", {
                    "model": model
                }, level="WARN")
                    
        except Exception as e:
            model_duration = time.time() - model_start_time
            _verbose_log("AGENT_MAIN: Model attempt failed with exception", {
                "model": model,
                "model_idx": model_idx,
                "duration_seconds": round(model_duration, 2),
                "error_type": type(e).__name__,
                "error": str(e)
            }, level="ERROR")
            # Move to next model on exception
            continue
    
    # All models failed
    total_duration = time.time() - agent_start_time
    _verbose_log("AGENT_MAIN: All model attempts failed", {
        "total_models_tried": len(AGENT_MODELS),
        "total_duration_seconds": round(total_duration, 2)
    }, level="ERROR")
    return ""
