# Spec-Only Architecture for Polyglot Problems

## Problem

Polyglot problems (exercism-style) don't provide `tests.py` to the agent at runtime. The agent only receives:
- `main.py` (with `pass` placeholders)
- `instructions.md` (specification)
- No external tests

This makes validation challenging because the agent can't run real tests to verify correctness.

## Solution: LLM-Generated Synthetic Tests

### Architecture Flow

```
Instructions → Parse Requirements → Generate Code → Generate Synthetic Tests → Validate → Refine
```

### Key Components

#### 1. `_generate_synthetic_tests()` Method

**Location**: `MinimalCursorAgent` class (line 387-421)

**Purpose**: Ask LLM to generate test cases based on the problem specification

**Input**: `run_id` (for LLM call tracking)

**Output**: Python test code as string

**Mechanism**:
```python
1. Create test generation prompt with instructions
2. Call LLM with system message: "You are a test generation expert"
3. Extract test code from LLM response
4. Return the test code
```

**Prompt Structure**:
- System: "You are a test generation expert. Create comprehensive test cases based on specifications."
- User: Problem statement + test requirements

**Requirements for LLM**:
- Generate 3-5 test cases using unittest.TestCase
- Test specific functionality from instructions
- Use assertions like self.assertEqual
- Test edge cases and error conditions
- Follow exact error messages from instructions

#### 2. `run_synthetic_tests()` Function

**Location**: Top-level function (line 653-784)

**Purpose**: Execute synthetic test code and return validation results

**Input**: 
- `test_code`: Generated test code string
- `patch`: Unified diff patch

**Output**: `(all_passed: bool, feedback_message: str, test_results: dict)`

**Execution Flow**:
```
1. Reset git state
2. Apply patch to codebase
3. Write synthetic test code to synthetic_tests.py
4. Load main.py and synthetic_tests.py
5. Find test class in synthetic_tests module
6. Run each test method
7. Stop on first failure
8. Reset state and cleanup
9. Return results
```

**Validation**:
- Syntax check: Ensure synthetic tests are valid Python
- Execution: Run tests and capture failures
- Feedback: Return specific error messages for LLM refinement

#### 3. Integration Points

**Initial Generation** (line 533-572):
```python
# Generate synthetic tests for spec_only mode
if self.mode == "spec_only":
    synthetic_test_code = self._generate_synthetic_tests(run_id)

# After syntax check passes
elif self.mode == "spec_only" and synthetic_test_code:
    test_ok, test_feedback, test_details = run_synthetic_tests(synthetic_test_code, patch)
    if test_ok:
        return patch  # Success!
    else:
        err = test_feedback  # Feed to refinement
```

**Iterative Refinement** (line 641-653):
```python
# In refinement loop
elif self.mode == "spec_only" and synthetic_test_code:
    test_ok, test_feedback, test_details = run_synthetic_tests(synthetic_test_code, new_patch)
    if test_ok:
        return new_patch  # Success after refinement!
    else:
        err = test_feedback  # Continue refining
```

### Mode Detection

The agent detects mode based on available files:

```python
if problem_category not in ("spec_only", "tests_available"):
    # Best-guess mode detection
    problem_category = "tests_available" if os.path.exists("tests.py") else "spec_only"
```

For polyglot problems:
- `tests.py` doesn't exist
- Mode defaults to `spec_only`
- Synthetic test generation is triggered

### Benefits

1. **Self-Validation**: Agent validates its own code against specs
2. **Early Detection**: Catch logic errors before evaluation
3. **Iterative Improvement**: Get feedback and refine
4. **Spec-Driven**: Works with any problem that has clear instructions
5. **Graceful Degradation**: Falls back to syntax-only if test generation fails

### Limitations

1. **LLM Quality**: Synthetic tests depend on LLM's understanding of requirements
2. **Coverage**: May not cover all edge cases that real tests do
3. **False Positives**: Synthetic tests might pass while real tests fail
4. **Token Cost**: Additional LLM call for test generation

### Example Flow: dot-dsl

**Problem**: Build Graph DSL class

**Instructions**: Handle attributes, nodes, edges with specific error messages

**Agent Action**:
1. Generates Graph class implementation
2. Generates synthetic tests: `test_empty_graph`, `test_with_attributes`, `test_malformed_item`
3. Runs synthetic tests on implementation
4. Gets feedback: "ValueError instead of TypeError for empty tuple"
5. Refines implementation
6. Re-runs tests
7. All pass → Return patch

**Real Tests**: Still run in eval sandbox, but agent has higher confidence

### Status

✅ **Implemented**: Synthetic test generation and validation integrated
✅ **Integrated**: Works in both initial generation and refinement loops
✅ **Tested**: Syntax validation passed
⏳ **Pending**: Real-world evaluation on polyglot problems

### Future Enhancements

1. **Test Quality Scoring**: Assess synthetic test quality before using
2. **Example Extraction**: Parse I/O examples from instructions.md
3. **Test Templates**: Use domain-specific test templates for common patterns
4. **Multi-Pass Testing**: Generate tests, validate, generate more targeted tests
5. **Test Augmentation**: Add edge cases based on initial failures

## Conclusion

The spec-only architecture enables the agent to self-validate implementations even without external tests. This is crucial for polyglot problems where tests aren't available at runtime. The approach is inspired by Cursor AI's spec-driven development philosophy where the agent creates and runs its own validation.

