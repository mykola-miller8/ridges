# Bug Fix: Synthetic Test Generation Issues

## Errors Identified

### Run 6e3b96 and f753ee (dot-dsl)

**Error 1**: Synthetic tests unable to import main module
```
[SYNTH_TEST] Failed to load synthetic tests: No module named 'main'
```

**Root Cause**: Using `importlib.util.spec_from_file_location` didn't properly register the module in `sys.modules` for subsequent imports.

**Error 2**: Empty diff during refinement
```
[DIFF_GEN] Generated diff: 0 chars, 0 lines
```

**Root Cause**: `generate_diff_from_implementations` copied ALL files from current directory (including synthetic_tests.py), then wrote new content to the same files. If content was identical, git saw no changes.

## Fixes Applied

### Fix 1: Use `importlib.import_module` for Module Loading

**File**: `my-agents/v4.py` (lines 731-758)

**Change**: Switched from `importlib.util` to proper `importlib.import_module`:

**Before**:
```python
main_spec = importlib.util.spec_from_file_location("main", "main.py")
main_module = importlib.util.module_from_spec(main_spec)
main_spec.loader.exec_module(main_module)
```

**After**:
```python
import sys
import importlib

if "." not in sys.path:
    sys.path.insert(0, ".")

main_module = importlib.import_module("main")  # Properly registers in sys.modules
tests_module = importlib.import_module("synthetic_tests")  # Can now import from main
```

**Impact**: Modules are properly registered in `sys.modules`, allowing synthetic tests to import from `main`.

### Fix 2: Exclude Modified Files from Base Copy

**File**: `my-agents/v4.py` (lines 162-166)

**Change**: Skip copying files that will be modified:

**Before**:
```python
# Copy all .py files
if file.endswith('.py'):
    # ... copy file ...
```

**After**:
```python
# Only copy files that are NOT being modified
if rel_path not in file_implementations:
    # ... copy file ...
```

**Impact**: Prevents copying and overwriting the same file, which caused empty diffs.

### Fix 3: Clarify Import Instructions

**File**: `my-agents/v4.py` (lines 391-402)

**Change**: Made test generation prompt explicitly require importing from `main`:

```python
test_generation_prompt = (
    ...
    f"4. Import from 'main' module: use 'from main import ClassName, function_name'\n"
    f"5. Follow the exact error messages and exception types mentioned in instructions\n"
    f"6. CRITICAL: Always import from 'main', never from other module names\n\n"
    ...
)
```

**Impact**: Synthetic tests will now correctly import from `main.py`.

### Fix 4: Constrain Refinement to Original Files

**File**: `my-agents/v4.py` (lines 576-616)

**Changes**:
1. Store valid file names from initial generation (line 580):
   ```python
   valid_file_names = list(file_implementations.keys())
   ```

2. Include file names in refinement prompt (lines 586-588):
   ```python
   f"Please return corrected complete Python implementations for the same files you modified before: {valid_file_names}\n"
   f"CRITICAL: Use the exact same file names: {valid_file_names}\n"
   "Do not create new files or use different file names."
   ```

3. Filter out unexpected file names (lines 607-616):
   ```python
   # Validate that extracted files match expected file names
   unexpected_files = [f for f in new_impl.keys() if f not in valid_file_names]
   if unexpected_files:
       print(f"[AGENT]   Warning: Unexpected file names extracted: {unexpected_files}")
       print(f"[AGENT]   Expected: {valid_file_names}, filtering to match...")
       new_impl = {k: v for k, v in new_impl.items() if k in valid_file_names}
   
   if not new_impl:
       print(f"[AGENT]   No valid implementations after filtering, stopping refinement")
       break
   ```

**Impact**: Refinement will only work with the original file names, preventing empty diffs.

## Expected Behavior After Fix

1. **Synthetic Tests**: Will import from `main` module correctly
2. **Refinement**: Will maintain original file names, preventing empty diffs
3. **Logging**: Will warn when unexpected file names are detected

## Status

✅ Fixes applied and syntax validated
⏳ Ready for re-evaluation on dot-dsl

## Key Lesson

When doing iterative refinement with LLMs:
- Always constrain outputs to expected file names
- Be explicit about module import paths
- Add validation/filtering to prevent hallucinations
- Log warnings when LLM diverges from expectations

