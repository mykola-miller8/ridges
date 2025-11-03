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
    Check if code has valid Python syntax and can be compiled without runtime errors.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    _verbose_log("SYNTAX_VALIDATION: Starting syntax validation", {
        "code_length": len(code),
        "code_lines": len(code.splitlines())
    })
    try:
        # Step 1: Parse syntax
        tree = ast.parse(code)
        _verbose_log("SYNTAX_VALIDATION: Syntax is valid")
        
        # Step 2: Check imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        if imports:
            _verbose_log("SYNTAX_VALIDATION: Found imports", {"imports": imports})
            safe_imports = {
                'collections', 're', 'math', 'itertools', 'functools',
                'typing', 'dataclasses', 'enum', 'heapq', 'bisect',
                'string', 'operator', 'copy', 'decimal', 'fractions'
            }
            unsafe = [imp for imp in imports if imp.split('.')[0] not in safe_imports]
            if unsafe:
                _verbose_log("SYNTAX_VALIDATION: Unsafe imports detected", {
                    "unsafe_imports": unsafe
                }, level="WARN")
        
        # Step 3: Try to compile (catches more errors than parse)
        try:
            compile(code, '<string>', 'exec')
            _verbose_log("SYNTAX_VALIDATION: Code compiles successfully")
        except Exception as compile_err:
            error_msg = f"Compilation error: {compile_err}"
            _verbose_log("SYNTAX_VALIDATION: Compilation failed", {
                "error": str(compile_err),
                "error_type": type(compile_err).__name__
            }, level="ERROR")
            return False, error_msg
        
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
    
    # Strategy 1: Python block with main.py comment (flexible newlines)
    m = re.findall(r"```python\s*\n\s*#\s*main\.py\s*\n([\s\S]*?)```", response, re.DOTALL)
    if m and m[0].strip():
        extracted = m[0].strip()
        _verbose_log("CODE_EXTRACTION: Strategy 1 succeeded (explicit main.py)", {
            "extracted_length": len(extracted),
            "extracted_lines": len(extracted.splitlines()),
            "preview": extracted[:200] + "..." if len(extracted) > 200 else extracted
        })
        return extracted
    
    # Strategy 2: Any python code block (flexible newlines)
    m2 = re.findall(r"```python\s*\n([\s\S]*?)```", response, re.DOTALL)
    if m2:
        for block in m2:
            stripped = block.strip()
            if stripped:
                # Remove leading "# main.py" comment if present
                if stripped.startswith("# main.py"):
                    lines = stripped.split('\n', 1)
                    if len(lines) > 1:
                        stripped = lines[1].strip()
                
                _verbose_log("CODE_EXTRACTION: Strategy 2 succeeded (python block)", {
                    "extracted_length": len(stripped),
                    "extracted_lines": len(stripped.splitlines()),
                    "preview": stripped[:200] + "..." if len(stripped) > 200 else stripped
                })
                return stripped
    
    # Strategy 3: Generic code block (flexible newlines)
    m3 = re.findall(r"```\s*\n([\s\S]*?)```", response, re.DOTALL)
    if m3:
        for block in m3:
            stripped = block.strip()
            # Check if it looks like Python code
            if stripped and any(keyword in stripped for keyword in ['def ', 'class ', 'import ', 'from ']):
                # Remove leading "# main.py" comment if present
                if stripped.startswith("# main.py"):
                    lines = stripped.split('\n', 1)
                    if len(lines) > 1:
                        stripped = lines[1].strip()
                
                _verbose_log("CODE_EXTRACTION: Strategy 3 succeeded (generic code block)", {
                    "extracted_length": len(stripped),
                    "extracted_lines": len(stripped.splitlines()),
                    "preview": stripped[:200] + "..." if len(stripped) > 200 else stripped
                })
                return stripped
    
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
        "CRITICAL PRIORITIES:\n"
        "1. GET THE CORE LOGIC CORRECT FIRST - focus on basic functionality working properly\n"
        "2. THEN add validation - don't over-validate or make up restrictions not in the spec\n"
        "3. Follow the specification EXACTLY - do not deviate or add assumptions\n"
        "4. If error messages are specified, use them VERBATIM (exact wording)\n"
        "5. !!! DSL VALIDATION: Check `len(item) < min` BEFORE accessing `item[0]` - see top of guidelines\n"
        "6. For sequential inputs: watch for phase boundaries where constraints RESET ('new', 'fresh', 'bonus')\n"
        + ("7. IMPORTANT: Only modify main.py. Do not change tests.py.\n" if mode == "tests_available" else "")
        + "\nReturn your solution in this exact format:\n"
        "```python\n# main.py\n<your complete implementation here>\n```"
    )
    
    user_msg = f"""# Problem Statement

{problem_statement[:15000]}

# Repository Files

{repo_summary}

# IMPLEMENTATION GUIDELINES

## !!! CRITICAL: DSL TUPLE VALIDATION MUST BE IN THIS ORDER !!!
**If you see constants like `NODE, EDGE, ATTR = range(3)`, YOU MUST validate in this EXACT order:**

```python
# WRONG ORDER CAUSES WRONG ERRORS! Check len FIRST before accessing item[0]
for item in data:
    # Step 1: FIRST check completeness (TypeError)
    if len(item) < 2:  # Can't even access item[0] if len < 1
        raise TypeError("Graph item incomplete")  # NOT "X is malformed"!
    
    # Step 2: Then check type is valid (ValueError) 
    if item[0] not in [NODE, EDGE, ATTR]:
        raise ValueError("Unknown item")
    
    # Step 3: Then check type-specific length (ValueError)
    if item[0] == ATTR and len(item) != 3:
        raise ValueError("Attribute is malformed")
    # ... etc for NODE, EDGE
```

**WHY:** Empty `()` has len=0, can't check `item[0]`. Incomplete `(ATTR,)` has len=1, should be "incomplete" NOT "malformed".

## 0. APPROACH: CORRECT LOGIC FIRST, THEN VALIDATION
**The #1 priority is implementing the CORRECT core logic. Validation comes second.**

**WARNING: Don't over-engineer or over-validate!** Many failures come from:
- Implementing overly strict/incorrect completion logic
- Making up restrictions not in the spec
- Getting distracted by edge cases before basic logic works

### Three-step approach:
1. **Understand the problem**: Read the spec 2-3 times to understand what the code should DO
2. **Implement correct core logic**: Focus on getting the basic behavior right
3. **Add validation**: Then implement error checking, edge cases, and constraints

### For exact output matching (strings, lists, etc.):
- **Study examples CHARACTER BY CHARACTER**: punctuation, capitalization, spacing
- **Match return type format**: `list[str]` = one line per element; `str` = may have `\n`
- **Identify patterns**: pluralization, number words, special cases
- **Use examples as ground truth** - don't guess
- **For cumulative/repetitive text**: Study which parts repeat and which don't
  - Don't assume uniformity - check EVERY line in examples
  - Some elements may include extra info, some may not
  - The pattern may not be as simple as it first appears

### For error messages and validation:
- **If spec provides exact error messages, COPY THEM VERBATIM**
- Look for keywords: "raise", "exception", "error message", "must", "cannot"
- Tests verify both exception type AND exact message text

### For custom exceptions:
- **CRITICAL**: Custom exception classes must call `super().__init__(message)` in `__init__`
- Without this, `exception.args[0]` will be undefined and cause IndexError in tests
- **Example of CORRECT custom exception**:
  ```python
  class CustomError(Exception):
      def __init__(self, message):
          super().__init__(message)  # REQUIRED!
          # Optional: self.message = message
  ```
- **Simpler alternative**: Just use `class CustomError(Exception): pass` (no custom __init__)
- **DO NOT** implement `__init__` without calling `super().__init__(message)`
- Even if spec shows broken exception code, fix it to call super().__init__()

## 1. SIMPLICITY + CORRECTNESS = SUCCESS
**Get the basic logic working correctly with minimum complexity.**

- Use MINIMUM state variables needed
- Prefer simple, direct logic over clever abstractions
- For stateful classes: Track only what you absolutely need
- Don't over-engineer - try the simplest approach first
- **Keep code self-contained**: Only import from Python standard library (collections, re, math, itertools, etc.)
  - Avoid obscure imports that might not be available
  - Prefer built-in functions over imports when possible

### Mental testing (CRITICAL):
- Pick 2-3 examples from spec and trace through your logic
- Check: start → middle → end states
- Verify boundary conditions: first, last, empty inputs
- If you can't trace it in your head, simplify!

### For observer/reactive patterns (listeners, callbacks, dependencies):
- **Avoid circular dependencies**: If A depends on B and B depends on A, you'll get infinite loops
- **Initialize before registering**: Create objects fully before adding observers/callbacks
- **Store callbacks in lists, not sets**: Order matters, callbacks should be called in registration order
- **Keep it simple**: Don't over-engineer - basic list of observers/callbacks usually sufficient
- **Propagation pattern**: When value changes → notify observers → observers update themselves
- **Callback removal**: Store callbacks by identity (use list, support removal by reference)

### For sparse/vague specifications:
- **If spec is very short (<500 chars) or lacks detail, be EXTRA careful**
- **Think through edge cases explicitly**: What happens with ties→ Empty input→ Boundary conditions→
- **Don't make assumptions** - implement only what's clearly specified
- **For external references** (e.g., "see Wikipedia"), use common sense but stick to basics

### For comparison/ranking problems:
- **Tie-breaking is CRITICAL**: "Pick the best X" means if multiple items tie for best, return ALL tied items
- **Test tie scenarios mentally**: What if all items are equal→ What if 2 items tie for first→
- **Common pattern**: Calculate scores, find max score, return ALL items with max score
- **Example**:
  ```python
  scores = [score_item(item) for item in items]
  max_score = max(scores)
  return [items[i] for i, s in enumerate(scores) if s == max_score]
  ```

### For grid/board/coordinate problems (CRITICAL):
- **Clarify coordinate system FIRST**: Is `x` the row or column→ Is `y` the row or column→
- **Common conventions**:
  - `board[y][x]` or `board[row][col]` means y=row (vertical), x=col (horizontal)
  - If given `(x, y)` coordinates, determine which maps to rows and which to columns
- **Test with examples**: Pick a coordinate from the spec, verify your indexing gives the right cell
- **Rectangular boards expose errors**: Non-square boards make row/col confusion obvious
- **Trace a specific example**: "If spec says coordinate (2, 3), which cell is that in my board→"

### Special cases:
- Look for keywords: "special", "except", "however", "but", "otherwise", "bonus", "final"
- **Implement special cases EXPLICITLY** - don't assume a loop handles them
- Example: "The 10th frame is special" → implement frame 10 separately
- **Visual layouts with indentation/spacing**: Often encode structure (e.g., hex grids, trees)
  - Don't ignore the visual formatting - it's usually meaningful
  - If examples show increasing indentation per row → likely a hex/offset grid
- **Repetitive/cumulative patterns**: Don't assume uniformity - check examples line-by-line
  - What repeats exactly vs what varies
  - Some lines may have extra elements, others may not
- **Self-referential/shadowing definitions** (interpreters, parsers, DSLs):
  - **CRITICAL**: When redefining something that uses its own name, capture the OLD definition first
  - Example: `: foo 10 ;` then `: foo foo 1 + ;` should call the OLD foo (10), not recurse infinitely
  - **Implementation pattern**:
    ```python
    # When parsing ": foo foo 1 + ;" and foo already exists
    # WRONG: definitions['foo'] = ['foo', '1', '+']  # 'foo' will recurse!
    # RIGHT: Expand 'foo' to its current definition while parsing
    old_foo = definitions.get('foo', [])  # Save old definition
    new_body = []
    for token in [' foo', '1', '+']:
        if token in definitions:
            new_body.extend(definitions[token])  # Expand using OLD definition
        else:
            new_body.append(token)
    definitions['foo'] = new_body  # Now safe to overwrite
    ```
  - **Key insight**: Resolve references at definition time, not execution time

## 2. VALIDATION (AFTER CORE LOGIC WORKS)
**Add validation AFTER the basic functionality is correct.**

### Validation order (fail fast):
1. Type checks first (TypeError: wrong type, structural problems)
2. Structure checks (right length/format→)
3. Content checks (valid values, ranges)
4. Dependent checks last (constraints based on previous inputs - see below)

### Key points:
- **Use EXACT error messages if spec provides them** (copy verbatim)
- TypeError vs ValueError: TypeError = wrong type/structure; ValueError = wrong values
- For DSLs with tuples: Check tuple length matches expected (e.g., `len(item) != 3` → malformed)

### Dependent/Sequential Validation:
**For sequential inputs (methods called multiple times), later inputs may depend on earlier ones.**

**Key concept: Constraint resets vs. cumulative constraints**
- **Cumulative**: "If you knocked down 6, you can't knock down more than 4 remaining" (same phase)
- **Resets**: "After a strike, you get fresh pins" (new phase - constraints reset!)
- **Keywords**: Look for "new", "fresh", "reset", "bonus", "extra" → indicates phase boundary

**Implementation tip**:
```python
# Only apply cumulative constraints WITHIN the same phase
if self.in_same_phase() and self.previous_value:
    if value + self.previous_value > 10:
        raise Exception("Exceeds limit")
```

**How to find rules**:
- Read spec 2-3 times looking for: "cannot", "must not", "only if", "depends on"
- Look for phase boundaries: "new", "fresh", "bonus", "independent"
- Don't assume simple rules - check for special cases

### For DSL/parser problems with tuple-based syntax:
**DSLs with constants like `NODE, EDGE, ATTR = range(3)` require STRICT validation ordering.**

**MUST CHECK IN THIS ORDER (wrong order = wrong error type/message):**

```python
# Example: Validating graph DSL items
for item in data:
    # Step 1: Check completeness FIRST (TypeError)
    if len(item) < 2:  # Minimum: (TYPE, ...)
        raise TypeError("Graph item incomplete")
    
    # Step 2: Check type constant is valid (ValueError)
    if item[0] not in [NODE, EDGE, ATTR]:
        raise ValueError("Unknown item")
    
    # Step 3: Check type-specific length (ValueError)
    if item[0] == NODE and len(item) != 3:
        raise ValueError("Node is malformed")
    elif item[0] == EDGE and len(item) != 4:
        raise ValueError("Edge is malformed")
    elif item[0] == ATTR and len(item) != 3:
        raise ValueError("Attribute is malformed")
    
    # Step 4: Check element types (ValueError)
    if item[0] == NODE:
        if not isinstance(item[1], str) or not isinstance(item[2], dict):
            raise ValueError("Node is malformed")
    # ... similar for EDGE, ATTR
```

**WHY THIS ORDER MATTERS:**
- `()` has len=0, can't even check `item[0]` → must check length FIRST
- `(ATTR,)` has len=1, `item[0]` exists but missing args → "incomplete" not "malformed"
- Only after confirming completeness can you safely check type-specific requirements

### State management for stateful classes:
**CRITICAL: Get completion logic RIGHT - don't over-validate!**
- Understand what "complete" means by reading the spec carefully
- Example: 10 frames with 2 rolls each = 20 rolls (unless bonus rounds)
- Test your completion check mentally with examples from the spec
- Only reject input when truly invalid (don't make up extra restrictions)
- Store previous values if needed for dependent validation

## 3. COMMON PITFALLS
- **Circular dependencies**: In reactive/observer patterns, avoid A→B→A dependency chains
- **Initialization order**: Initialize all attributes in __init__ before registering observers
- **Over-complicating completion logic**: Understand what "complete" means, don't make up restrictions
- **DSL tuple validation - WRONG ORDER = WRONG ERROR**:
  - **CRITICAL**: Check tuple completeness (length) BEFORE checking type-specific validation
  - Empty `()` or incomplete `(TYPE,)` → TypeError: "incomplete" (NOT ValueError: "malformed")
  - Must check `len(item) < min_length` FIRST, before accessing `item[0]`
  - See Section 2 for complete code example showing correct validation order
- **Cumulative constraints across phases**: Don't apply unless in same phase (watch for "new", "fresh", "bonus")
- **Off-by-one errors**: Check loop bounds and index arithmetic
- **Loop safety**: In `while` loops with `continue`, increment BEFORE continue
- **Return types**: `list[str]` = one line per element (no embedded `\n`)
- **Hex/offset grids - CRITICAL for connectivity problems**:
  - **Indentation encodes adjacency**: If rows have increasing indentation, it's likely a hex grid
  - **Hex grids have 6 neighbors** (not 4 or 8) - diagonals typically NOT valid
  - **Parsing strategy**: Remove leading spaces from each row, track the offset per row
  - **Adjacency depends on row offset**: For offset grids, neighbors differ by row (even vs odd)
  - **MUST test adjacency with examples**: Pick a cell in the middle, manually verify its neighbors match expected
  - **Common mistake**: Using standard 4-way or 8-way adjacency on hex grids → wrong connectivity

## 4. FINAL VERIFICATION
Mentally trace through your code:
1. Does it handle the basic case correctly→
2. Does it handle ALL special cases from the spec→
3. For output: Did I match examples EXACTLY (punctuation, capitalization, spacing)→
4. For validation: Did I use EXACT error messages if provided→
5. **For DSL/tuple validation**: Did I validate in the correct order (completeness → type valid → length → element types)→
6. For sequential inputs: Did I identify phase boundaries where constraints reset→
7. **For grids with visual formatting**: Did I handle indentation/offset adjacency correctly→
8. **For interpreters/DSLs with definitions**: When redefining `X` to use `X`, did I capture the old definition first→
9. **For coordinate systems**: Did I verify x/y map correctly to rows/columns using spec examples→
10. **For comparison/ranking problems**: Did I handle ties correctly (return ALL items with best score)→
11. **For observer/reactive patterns**: Did I avoid circular dependencies and initialize before registering→
12. **If spec is sparse**: Did I think through edge cases explicitly→
13. Can I explain the logic in 2-3 simple sentences→
14. Did I test the logic with examples from the spec→

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
