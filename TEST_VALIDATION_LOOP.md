# Test-Running Validation Loop - Detailed Design

## Overview

The test-running validation loop is a critical feature that enables the agent to iteratively improve its code by executing actual tests and receiving concrete feedback on failures.

## Problem Statement

**Before**: The agent only validated:
- Syntax correctness (compile)
- Patch applicability (dry-run)
- Basic sanity checks

**Gap**: No functional validation - the agent couldn't know if its code actually satisfied test requirements.

**Impact**: 
- 25% of failures (like dot-dsl) had correct syntax but wrong logic
- No feedback loop for iterative improvement
- Agent had to guess what tests expected

## Solution Architecture

### Key Components

1. **`run_tests_locally(patch)` Function**
   - **Purpose**: Execute tests on patched code and return structured results
   - **Input**: Unified diff patch string
   - **Output**: (all_passed: bool, feedback_message: str, test_results: dict)
   - **Mechanism**:
     - Apply patch to working directory
     - Import and execute main.py and tests.py
     - Run each test method in sequence
     - Stop on first failure
     - Capture error message and traceback
     - Reset working directory to clean state
     - Return structured feedback

2. **Integration Points**

   **Initial Generation** (line 506-517):
   ```python
   if self.mode == "tests_available":
       print(f"[AGENT]   Running initial tests for validation...")
       test_ok, test_feedback, test_details = run_tests_locally(patch)
       if test_ok:
           print(f"[AGENT] ✓ All tests passed on first try!")
           return patch
       else:
           print(f"[AGENT]   Tests failed: {test_feedback}")
           err = test_feedback  # Use test feedback for refinement
   ```

   **Iterative Refinement** (line 577-589):
   ```python
   if self.mode == "tests_available":
       print(f"[AGENT]   Running tests for validation...")
       test_ok, test_feedback, test_details = run_tests_locally(new_patch)
       if test_ok:
           print(f"[AGENT] ✓ All tests passed in refinement {refinement_round + 1}!")
           return new_patch
       else:
           print(f"[AGENT]   Tests failed: {test_feedback}")
           err = test_feedback  # Feed back to LLM for next round
   ```

### Execution Flow

```
Generate Code → Syntax Check → Patch Generation → Dry Run → Test Execution
                                                                      ↓
                                             All tests pass? → YES → Return Success
                                                                      ↓
                                                                    NO
                                                                      ↓
Extract failing test name and error → Feed to LLM → Refinement Round → Repeat
```

### Error Feedback Format

The LLM receives structured feedback:
```
Test 'test_malformed_graph_item' failed.
Error: Attribute is malformed
The test expects something different from what your code does.
```

This is much more actionable than generic syntax errors.

## Safety Guarantees

1. **State Isolation**: Working directory always reset to clean state before and after test execution
2. **Timeout Protection**: All subprocess calls have timeouts
3. **Error Recovery**: Try/finally blocks ensure cleanup even on exceptions
4. **Mode-Aware**: Only runs tests in `tests_available` mode; skips for `spec_only` mode

## Benefits

1. **Early Detection**: Catches logic errors before returning patch
2. **Concrete Feedback**: Knows exactly which test failed and why
3. **Iterative Improvement**: Up to 3 refinement rounds with test feedback
4. **Higher Success Rate**: Expected to increase from 48.5% toward 70% target

## Example: dot-dsl

**Before**:
- Agent generated code with `self.nodes = {}` (dict instead of list)
- Syntax valid ✓
- Patch applicable ✓
- **But**: `test_empty_graph` failed with `{} != []`

**After** (with test validation):
```
[TEST_RUN] Running test_malformed_graph_item...
[TEST_RUN] test_malformed_graph_item: FAILED - Attribute is malformed
[TEST_RUN] Results: 9/10 passed, 1 failed
[AGENT]   Tests failed: Test 'test_malformed_graph_item' failed...
[AGENT]   Feedback to LLM: Test 'test_malformed_graph_item' failed...
[AGENT]   Refinement round 1/3...
```

The agent now knows:
1. Which specific test failed
2. What error message it got
3. That it needs to fix the logic (not syntax)

LLM can then generate corrected code based on this feedback.

## Performance Considerations

- **Overhead**: +2-5 seconds per test execution (negligible compared to LLM calls)
- **Frequency**: Tests run once on initial patch, once per refinement round (max 3)
- **Optimization**: Stop on first failure to minimize overhead
- **Mode-Specific**: Only enabled for `tests_available` mode to avoid wasted cycles

## Future Enhancements

1. **Test Selection**: Run only impacted tests instead of full suite
2. **Parallel Execution**: Run all tests in parallel for faster feedback
3. **Test Coverage**: Track which parts of code tests exercise
4. **Incremental Testing**: Re-run only failing tests between rounds

## Conclusion

The test-running validation loop transforms the agent from "syntactically correct guesser" to "functionally validated problem solver". This is the key feature that will help us reach the 70% success rate target.

