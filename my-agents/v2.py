# v2.py - Adversarial Test Generator Agent
# Generic code agent using adversarial architecture
# Test generator challenges solver until all tests pass

from __future__ import annotations

import os
import re
import json
import time
import subprocess
import tempfile
import hashlib
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import requests


# ============================================================================
# Configuration and Constants
# ============================================================================

DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
DEBUG_MODE = os.getenv("V2_DEBUG", "0") == "1"

# Adversarial Configuration
NUM_TESTS_PER_ITERATION = 10  # Easy to change
MAX_ITERATIONS = 10
TEST_GENERATION_TIMEOUT = 60
SOLVING_TIMEOUT = 120
TEST_EXECUTION_TIMEOUT = 60

# Whitelisted models (production requirement)
GLM_MODEL_NAME = "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"

AGENT_MODELS = [GLM_MODEL_NAME, QWEN_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME]
AGENT_ID = "adversarial-agent/v2.0"


# ============================================================================
# Data Structures
# ============================================================================

class TestResult:
    """Result of test execution"""
    def __init__(self, passed: List[str], failed: List[str], error_details: Dict[str, str], all_passed: bool):
        self.passed = passed
        self.failed = failed
        self.error_details = error_details
        self.all_passed = all_passed
    
    @property
    def total_count(self) -> int:
        return len(self.passed) + len(self.failed)
    
    @property
    def passed_count(self) -> int:
        return len(self.passed)
    
    @property
    def failed_count(self) -> int:
        return len(self.failed)


class ModelConfig:
    """Configuration for LLM model usage"""
    def __init__(self, name: str, endpoint: str, timeout: int = 120, max_retries: int = 2):
        self.name = name
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries


# ============================================================================
# Logging System
# ============================================================================

class Logger:
    """Simple logging system for the adversarial agent"""
    
    def __init__(self):
        self.logs = []
    
    def log(self, component: str, message: str, run_id: str = ""):
        """Log a message with timestamp and component"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{component}]"
        if run_id:
            log_entry += f" [{run_id}]"
        log_entry += f" {message}"
        
        self.logs.append(log_entry)
        print(log_entry)  # Always print for debugging
    
    def get_logs(self) -> List[str]:
        """Get all log entries"""
        return self.logs.copy()

# Global logger instance
_global_logger = Logger()


# ============================================================================
# LLM Client
# ============================================================================

class LLMClient:
    """Client for interacting with LLM models"""
    
    def __init__(self, proxy_url: str = DEFAULT_PROXY_URL):
        self.proxy_url = proxy_url
        self.logger = Logger()
    
    def call_model(self, model_name: str, prompt: str, timeout: int = 120) -> Optional[str]:
        """Call a specific model with the given prompt"""
        try:
            self.logger.log("LLM_CALL", f"Calling {model_name} with timeout {timeout}s")
            
            response = requests.post(
                f"{self.proxy_url}/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4000,
                    "temperature": 0.1
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                self.logger.log("LLM_CALL", f"Successfully called {model_name}, got {len(content)} chars")
                return content
            else:
                self.logger.log("ERROR", f"LLM call failed: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.log("ERROR", f"LLM call timed out for {model_name}")
            return None
        except Exception as e:
            self.logger.log("ERROR", f"LLM call failed for {model_name}: {e}")
            return None


# ============================================================================
# Test Generator Component
# ============================================================================

class AdversarialTestGenerator:
    """Generates adversarial tests based on problem statement and previous failures"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.logger = Logger()
    
    def generate_tests(self, problem_statement: str, previous_solution: Optional[str] = None, 
                      test_failures: Optional[TestResult] = None) -> List[str]:
        """Generate test cases based on problem statement and previous failures"""
        
        iteration = 1 if previous_solution is None else 2
        self.logger.log("TEST_GEN", f"Generating {NUM_TESTS_PER_ITERATION} tests for iteration {iteration}")
        
        # Build prompt based on iteration
        if iteration == 1:
            prompt = self._build_initial_test_prompt(problem_statement)
        else:
            prompt = self._build_adversarial_test_prompt(problem_statement, previous_solution, test_failures)
        
        # Try different models
        for model_name in AGENT_MODELS:
            response = self.llm_client.call_model(model_name, prompt, TEST_GENERATION_TIMEOUT)
            if response:
                tests = self._parse_test_response(response)
                if tests:
                    self.logger.log("TEST_GEN", f"Generated {len(tests)} tests using {model_name}")
                    return tests
                else:
                    self.logger.log("TEST_GEN", f"Failed to parse tests from {model_name}")
            else:
                self.logger.log("TEST_GEN", f"Failed to get response from {model_name}")
        
        self.logger.log("ERROR", "Failed to generate tests from any model")
        return []
    
    def _build_initial_test_prompt(self, problem_statement: str) -> str:
        """Build prompt for initial test generation"""
        return f"""You are a test generator for a generic code agent. Analyze the problem statement and generate {NUM_TESTS_PER_ITERATION} comprehensive test cases.

PROBLEM STATEMENT:
{problem_statement}

REQUIREMENTS:
1. Generate exactly {NUM_TESTS_PER_ITERATION} test cases
2. Each test should be in pytest format: def test_name(): assert function_call() == expected_result
3. Cover diverse scenarios: edge cases, boundary conditions, empty inputs, type validation
4. Test both success cases and error conditions
5. Use generic assertions and avoid problem-specific terminology
6. Focus on function behavior, input/output types, and error handling

GENERIC TEST PATTERNS TO CONSIDER:
- Empty/null input handling
- Type validation (string vs list vs other)
- Boundary value testing
- Error message validation
- Output format verification
- Edge case scenarios

OUTPUT FORMAT:
Return only the test functions, one per line, in this exact format:
def test_case_1(): assert function_name(input) == expected_output
def test_case_2(): assert function_name(input) == expected_output
...

Do not include any explanations or additional text."""

    def _build_adversarial_test_prompt(self, problem_statement: str, previous_solution: str, 
                                     test_failures: TestResult) -> str:
        """Build prompt for adversarial test generation based on previous failures"""
        failed_tests = "\n".join(test_failures.failed)
        error_details = "\n".join([f"{test}: {error}" for test, error in test_failures.error_details.items()])
        
        return f"""You are an adversarial test generator. The previous solution failed some tests. Generate {NUM_TESTS_PER_ITERATION} new test cases that will challenge the solver more effectively.

PROBLEM STATEMENT:
{problem_statement}

PREVIOUS SOLUTION:
{previous_solution}

FAILED TESTS:
{failed_tests}

ERROR DETAILS:
{error_details}

ADVERSARIAL STRATEGY:
1. Analyze what the previous solution got wrong
2. Generate tests that target those specific weaknesses
3. Create more challenging edge cases
4. Test scenarios the previous solution didn't handle
5. Focus on the types of errors that occurred

REQUIREMENTS:
1. Generate exactly {NUM_TESTS_PER_ITERATION} test cases
2. Each test should be in pytest format: def test_name(): assert function_call() == expected_result
3. Make tests more challenging than the previous batch
4. Target the specific failure patterns observed
5. Use generic assertions and avoid problem-specific terminology

OUTPUT FORMAT:
Return only the test functions, one per line, in this exact format:
def test_case_1(): assert function_name(input) == expected_output
def test_case_2(): assert function_name(input) == expected_output
...

Do not include any explanations or additional text."""

    def _parse_test_response(self, response: str) -> List[str]:
        """Parse test functions from LLM response"""
        tests = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('def test_') and 'assert' in line:
                tests.append(line)
        
        return tests


# ============================================================================
# Test Executor Component
# ============================================================================

class TestExecutor:
    """Executes tests on generated code"""
    
    def __init__(self):
        self.logger = Logger()
    
    def execute_tests(self, code: str, test_cases: List[str]) -> TestResult:
        """Execute test cases on the generated code"""
        self.logger.log("TEST_EXEC", f"Executing {len(test_cases)} test cases")
        
        temp_dir = None
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            self.logger.log("TEST_EXEC", f"Created temp directory: {temp_dir}")
            
            # Write code to main.py
            main_file = os.path.join(temp_dir, "main.py")
            with open(main_file, 'w') as f:
                f.write(code)
            
            # Write tests to tests.py
            tests_file = os.path.join(temp_dir, "tests.py")
            with open(tests_file, 'w') as f:
                f.write("import unittest\n")
                f.write("from main import *\n\n")
                f.write("class TestMain(unittest.TestCase):\n")
                for i, test_case in enumerate(test_cases):
                    f.write(f"    def test_{i}(self):\n")
                    # Convert assert to unittest format
                    if 'assert' in test_case:
                        assertion = test_case.split('assert')[1].strip()
                        f.write(f"        self.assertEqual({assertion.split('==')[0].strip()}, {assertion.split('==')[1].strip()})\n")
                    else:
                        f.write(f"        {test_case}\n")
            
            # Run tests
            result = subprocess.run(
                ['python', '-m', 'unittest', 'tests.py', '-v'],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=TEST_EXECUTION_TIMEOUT
            )
            
            # Parse results
            return self._parse_test_results(result.stdout, result.stderr)
            
        except Exception as e:
            self.logger.log("ERROR", f"Test execution failed: {e}")
            return TestResult([], [], {"execution_error": str(e)}, False)
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def _parse_test_results(self, stdout: str, stderr: str) -> TestResult:
        """Parse test execution results"""
        passed = []
        failed = []
        error_details = {}
        
        lines = stdout.split('\n')
        for line in lines:
            if 'test_' in line:
                if 'ok' in line:
                    test_name = line.split('(')[0].strip()
                    passed.append(test_name)
                elif 'FAIL' in line:
                    test_name = line.split('(')[0].strip()
                    failed.append(test_name)
                    error_details[test_name] = "Test failed"
        
        all_passed = len(failed) == 0
        self.logger.log("TEST_EXEC", f"Results: {len(passed)} passed, {len(failed)} failed")
        
        return TestResult(passed, failed, error_details, all_passed)


# ============================================================================
# Solver Component
# ============================================================================

class AdversarialSolver:
    """Solves problems based on test cases"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.logger = Logger()
    
    def solve(self, problem_statement: str, test_cases: List[str], 
              previous_attempts: List[str] = []) -> str:
        """Generate code to pass all test cases"""
        
        self.logger.log("SOLVER", f"Solving with {len(test_cases)} test cases")
        
        prompt = self._build_solve_prompt(problem_statement, test_cases, previous_attempts)
        
        # Try different models
        for model_name in AGENT_MODELS:
            response = self.llm_client.call_model(model_name, prompt, SOLVING_TIMEOUT)
            if response:
                code = self._extract_code(response)
                if code:
                    self.logger.log("SOLVER", f"Generated solution using {model_name}")
                    return code
                else:
                    self.logger.log("SOLVER", f"Failed to extract code from {model_name}")
            else:
                self.logger.log("SOLVER", f"Failed to get response from {model_name}")
        
        self.logger.log("ERROR", "Failed to generate solution from any model")
        return ""
    
    def _build_solve_prompt(self, problem_statement: str, test_cases: List[str], 
                           previous_attempts: List[str]) -> str:
        """Build prompt for code generation"""
        test_cases_text = "\n".join(test_cases)
        
        previous_attempts_text = ""
        if previous_attempts:
            previous_attempts_text = f"\nPREVIOUS FAILED ATTEMPTS:\n" + "\n".join(previous_attempts)
        
        return f"""You are a generic code agent. Generate Python code that solves the problem and passes ALL test cases.

PROBLEM STATEMENT:
{problem_statement}

TEST CASES TO PASS:
{test_cases_text}
{previous_attempts_text}

REQUIREMENTS:
1. Generate complete Python code that implements the required functionality
2. The code must pass ALL test cases exactly as written
3. Pay attention to exact output formats, data types, and error messages
4. Handle edge cases and boundary conditions properly
5. Use generic, clean code without problem-specific hardcoding
6. Focus on correctness over optimization

GENERIC GUIDANCE:
- Ensure correct return types (string, list, dict, etc.)
- Handle empty/null inputs appropriately
- Validate input types and formats
- Return exact error messages as specified
- Follow Python best practices

OUTPUT FORMAT:
Return only the Python code, starting with function definitions. Do not include explanations or markdown formatting."""

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response"""
        lines = response.strip().split('\n')
        code_lines = []
        
        in_code = False
        for line in lines:
            if line.strip().startswith('def ') or line.strip().startswith('import ') or line.strip().startswith('from '):
                in_code = True
            if in_code:
                code_lines.append(line)
        
        return '\n'.join(code_lines)


# ============================================================================
# Main Agent Loop
# ============================================================================

def agent_main(problem_statement: str) -> str:
    """Main agent function implementing adversarial test-solve-refine loop"""
    
    _global_logger.log("AGENT", f"Starting adversarial test generator agent with problem length: {len(problem_statement)}")
    
    try:
        llm_client = LLMClient()
        test_generator = AdversarialTestGenerator(llm_client)
        test_executor = TestExecutor()
        solver = AdversarialSolver(llm_client)
        
        _global_logger.log("AGENT", "Components initialized successfully")
        
        iteration = 0
        test_cases = []
        previous_solution = None
        test_failures = None
        previous_attempts = []
        
        while iteration < MAX_ITERATIONS:
            iteration += 1
            _global_logger.log("ITERATION", f"Starting iteration {iteration}/{MAX_ITERATIONS}")
            
            # Generate tests
            _global_logger.log("ITERATION", f"Generating {NUM_TESTS_PER_ITERATION} tests")
            test_cases = test_generator.generate_tests(problem_statement, previous_solution, test_failures)
            
            if not test_cases:
                _global_logger.log("ERROR", "Failed to generate test cases")
                break
            
            _global_logger.log("ITERATION", f"Generated {len(test_cases)} test cases")
            
            # Solve
            _global_logger.log("ITERATION", "Generating solution")
            solution = solver.solve(problem_statement, test_cases, previous_attempts)
            
            if not solution:
                _global_logger.log("ERROR", "Failed to generate solution")
                break
            
            _global_logger.log("ITERATION", f"Generated solution ({len(solution)} chars)")
            
            # Execute tests
            _global_logger.log("ITERATION", "Executing tests")
            results = test_executor.execute_tests(solution, test_cases)
            
            # Check success
            if results.all_passed:
                _global_logger.log("ITERATION", f"All tests passed! Success in iteration {iteration}")
                return generate_patch(solution)
            
            # Record failure and iterate
            _global_logger.log("ITERATION", f"Tests failed: {results.failed_count}/{results.total_count}")
            previous_solution = solution
            test_failures = results
            previous_attempts.append(solution)
        
        _global_logger.log("AGENT", f"Failed after {MAX_ITERATIONS} iterations")
        return ""
        
    except Exception as e:
        _global_logger.log("ERROR", f"Agent failed with error: {e}")
        import traceback
        _global_logger.log("ERROR", f"Traceback: {traceback.format_exc()}")
        return ""


# ============================================================================
# Diff Generation
# ============================================================================

def generate_patch(solution_code: str) -> str:
    """Generate git diff patch from solution code"""
    try:
        # Create temporary directory for git operations
        temp_dir = tempfile.mkdtemp()
        
        # Initialize git repo
        subprocess.run(['git', 'init'], cwd=temp_dir, check=True)
        subprocess.run(['git', 'config', 'user.email', 'agent@example.com'], cwd=temp_dir, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Agent'], cwd=temp_dir, check=True)
        
        # Create original main.py (empty or with pass)
        original_file = os.path.join(temp_dir, 'main.py')
        with open(original_file, 'w') as f:
            f.write('def main():\n    pass\n')
        
        # Commit original
        subprocess.run(['git', 'add', 'main.py'], cwd=temp_dir, check=True)
        subprocess.run(['git', 'commit', '-m', 'Original'], cwd=temp_dir, check=True)
        
        # Write solution
        with open(original_file, 'w') as f:
            f.write(solution_code)
        
        # Generate diff
        result = subprocess.run(['git', 'diff'], cwd=temp_dir, capture_output=True, text=True, check=True)
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return result.stdout
        
    except Exception as e:
        return f"# Error generating patch: {e}\n{solution_code}"


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # Read problem statement from input.json
    try:
        with open('/sandbox/input.json', 'r') as f:
            input_data = json.load(f)
            problem_statement = input_data.get('problem_statement', '')
        
        if not problem_statement:
            print("No problem statement found in input.json")
            sys.exit(1)
        
        # Run agent
        patch = agent_main(problem_statement)
        print(patch)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
