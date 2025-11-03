#!/usr/bin/env python3
"""
Test script to validate the code-first approach on a few simple problems
"""

import os
import sys
import asyncio
import pathlib
import tempfile
import shutil

# Add the ridges directory to the path
sys.path.append('/root/62/ridges')

# Import the test framework
from test_agent import run_local_evaluation_run
from evaluator.sandbox.sandbox_manager import SandboxManager
from evaluator.problem_suites.polyglot.polyglot_suite import PolyglotSuite

async def test_simple_problems():
    """Test the code-first approach on simple polyglot problems"""
    
    # Create a temporary directory for the agent
    with tempfile.TemporaryDirectory() as temp_dir:
        agent_path = os.path.join(temp_dir, "agent.py")
        
        # Copy the new agent code
        shutil.copy2("/root/62/ridges/new-agent.py", agent_path)
        
        # Read the agent code
        with open(agent_path, "r") as f:
            agent_code = f.read()
        
        print(f"Testing code-first agent on simple problems...")
        print(f"Agent code length: {len(agent_code)} characters")
        
        # Initialize sandbox manager
        sandbox_manager = SandboxManager()
        
        # Initialize problem suite
        polyglot_suite = PolyglotSuite()
        
        # Test problems (simple ones first)
        test_problems = [
            "proverb",
            "robot-name", 
            "pig-latin",
            "affine-cipher"
        ]
        
        results = {}
        
        for problem_name in test_problems:
            print(f"\n{'='*50}")
            print(f"Testing problem: {problem_name}")
            print(f"{'='*50}")
            
            try:
                # Run the evaluation
                evaluation_run = await run_local_evaluation_run(
                    sandbox_manager, 
                    [polyglot_suite], 
                    problem_name
                )
                
                # Check results
                if evaluation_run.test_results:
                    passed_tests = sum(1 for result in evaluation_run.test_results if result.status.value == "passed")
                    total_tests = len(evaluation_run.test_results)
                    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
                    
                    results[problem_name] = {
                        'passed': passed_tests,
                        'total': total_tests,
                        'success_rate': success_rate,
                        'status': evaluation_run.status.value,
                        'patch_length': len(evaluation_run.patch) if evaluation_run.patch else 0
                    }
                    
                    print(f"✓ {problem_name}: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
                    print(f"  Status: {evaluation_run.status.value}")
                    print(f"  Patch length: {len(evaluation_run.patch) if evaluation_run.patch else 0} characters")
                else:
                    results[problem_name] = {
                        'passed': 0,
                        'total': 0,
                        'success_rate': 0,
                        'status': evaluation_run.status.value,
                        'patch_length': 0
                    }
                    print(f"✗ {problem_name}: No test results")
                    
            except Exception as e:
                print(f"✗ {problem_name}: Error - {e}")
                results[problem_name] = {
                    'passed': 0,
                    'total': 0,
                    'success_rate': 0,
                    'status': 'error',
                    'patch_length': 0,
                    'error': str(e)
                }
        
        # Print summary
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        
        total_passed = 0
        total_tests = 0
        
        for problem_name, result in results.items():
            print(f"{problem_name:15} | {result['passed']:2}/{result['total']:2} | {result['success_rate']:5.1f}% | {result['status']:15} | {result['patch_length']:4} chars")
            total_passed += result['passed']
            total_tests += result['total']
        
        overall_success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
        print(f"{'='*60}")
        print(f"OVERALL: {total_passed}/{total_tests} tests passed ({overall_success_rate:.1f}%)")
        
        return results

if __name__ == "__main__":
    print("Testing code-first approach on simple polyglot problems...")
    results = asyncio.run(test_simple_problems())
    
    # Check if we're meeting the target
    total_passed = sum(r['passed'] for r in results.values())
    total_tests = sum(r['total'] for r in results.values())
    success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\nTarget: 45% success rate")
    print(f"Actual: {success_rate:.1f}% success rate")
    
    if success_rate >= 45:
        print("🎉 SUCCESS: Target achieved!")
    else:
        print("❌ Target not yet achieved, but code-first approach is working")
