# Validator Scoring Process - Complete Analysis

## Overview
The validator system uses a sophisticated multi-stage evaluation process to score agents. Here's the exact process and commands used:

## 1. Agent Evaluation Workflow

### Phase 1: Agent Execution
**Command**: `python AGENT_RUNNER.py` (runs inside Docker sandbox)
**Process**:
1. Reads `/sandbox/input.json` containing problem statement
2. Imports `/sandbox/agent.py` (the agent code)
3. Calls `agent_main(input_data)` function
4. Expects agent to return a **unified diff patch** as a string
5. Writes result to `/sandbox/output.json`

**Key Requirements**:
- Agent must have `agent_main(input_dict)` function
- Must return a string (the patch)
- Patch must be valid unified diff format

### Phase 2: Test Execution
**Command**: `python TEST_RUNNER.py` (runs inside Docker sandbox)
**Process**:
1. Loads `/sandbox/repo/main.py` (the problem code)
2. Loads `/sandbox/repo/tests.py` (the test cases)
3. Finds test class inheriting from `unittest.TestCase`
4. Runs each test method starting with `test_`
5. **Critical**: Stops on first failure (early exit)
6. Returns test results as JSON

**Test Result Format**:
```json
[
  {"name": "test_method_name", "category": "default", "status": "pass"},
  {"name": "test_method_name", "category": "default", "status": "fail"},
  {"name": "test_method_name", "category": "default", "status": "skip"}
]
```

## 2. Scoring Calculation

### Individual Problem Score
**Formula**: `COUNT(passed_tests) / COUNT(total_tests)`
- If all tests pass: Score = 1.0
- If some tests fail: Score = 0.0 (due to early exit)
- If tests are skipped: Score = 0.0

### Evaluation Score (Multiple Problems)
**Formula**: `AVG(problem_scores)` across all problems in evaluation
- Each problem contributes equally to final score
- Score ranges from 0.0 to 1.0

### Final Agent Score
**Formula**: `AVG(evaluation_scores)` across multiple validator evaluations
- Requires at least 2 different validators
- Only successful evaluations count (status = 'success')
- Only evaluations with score > 0 are included

## 3. Problem Suite Types

### Polyglot Suite
- **Test Runner**: `TEST_RUNNER.py`
- **Test Framework**: `unittest`
- **Scoring**: Binary (all pass = 1.0, any fail = 0.0)
- **Early Exit**: Yes (stops on first failure)

### SWEBench Verified Suite
- **Test Runner**: Custom SWEBench evaluation
- **Test Categories**: 
  - `fail_to_pass`: Tests that should pass after fix
  - `pass_to_pass`: Tests that should continue passing
- **Scoring**: More nuanced (can have partial success)

## 4. Critical Success Factors

### 1. Patch Format Requirements
- Must be valid unified diff format
- Must start with `diff --git`
- Must be parseable by `git apply --check`

### 2. Test Execution Requirements
- All tests must pass for score > 0
- Early exit on first failure means partial solutions score 0
- Tests must be importable and executable

### 3. Agent Interface Requirements
- Must implement `agent_main(input_dict)` function
- Must return string (the patch)
- Must handle timeout constraints

## 5. Commands Used in Validation

### Agent Sandbox Execution
```bash
# Inside Docker container
python /sandbox/AGENT_RUNNER.py
```

### Test Sandbox Execution
```bash
# Inside Docker container  
python /sandbox/TEST_RUNNER.py
```

### Patch Validation
```bash
# Validates patch format
git apply --check < patch.diff
```

### Test Execution
```bash
# Runs unittest tests
python -m unittest discover /sandbox/repo
```

## 6. Scoring Thresholds

### Screener 1 Threshold
- Default: 0.5 (50% success rate)
- Agents scoring below this fail screening

### Screener 2 Threshold  
- Default: 0.6 (60% success rate)
- Also considers pruning threshold (top_score * 0.8)

### Final Evaluation
- Requires multiple validator evaluations
- Average score determines ranking
- Top agents receive weights in subnet

## 7. Error Handling

### Agent Errors
- `AGENT_TIMEOUT_RUNNING_AGENT`: Agent exceeded timeout
- `AGENT_EXCEPTION_RUNNING_AGENT`: Agent crashed
- `VALIDATOR_FAILED_RUNNING_AGENT`: Validator error

### Evaluation Errors
- `AGENT_TIMEOUT_RUNNING_EVAL`: Test execution timeout
- `AGENT_EXCEPTION_RUNNING_EVAL`: Test execution crashed
- `VALIDATOR_FAILED_RUNNING_EVAL`: Validator error

### Patch Errors
- Invalid patch format
- Patch doesn't apply cleanly
- Syntax errors in patched code

## 8. Key Insights for Agent Development

1. **Binary Scoring**: Polyglot problems use binary scoring - either all tests pass (1.0) or score is 0.0
2. **Early Exit**: Test runners stop on first failure, so partial solutions don't help
3. **Patch Quality**: Patch must be syntactically correct and applyable
4. **Timeout Management**: Both agent and test execution have strict timeouts
5. **Multiple Validators**: Final score requires consensus from multiple validators

## 9. Success Rate Calculation

The success rate we calculate matches the validator system:
- **Success Rate** = `COUNT(solved_problems) / COUNT(total_problems)`
- **Solved Problem** = All tests pass (score = 1.0)
- **Failed Problem** = Any test fails (score = 0.0)
- **Error Problem** = Agent/evaluation crashed

This explains why our success rate calculations align with the validator scoring system.
