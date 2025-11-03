#!/usr/bin/env python3

import os
import json
from pathlib import Path

def analyze_latest_results():
    # Find the latest test results directory
    results_dir = Path("/root/62/ridges/test_agent_results")
    latest_dir = max([d for d in results_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
    
    print(f"Analyzing results from: {latest_dir}")
    
    successful_problems = []
    failed_problems = []
    error_problems = []
    total_tests = 0
    passed_tests = 0
    
    for problem_dir in latest_dir.iterdir():
        if not problem_dir.is_dir() or problem_dir.name.endswith('.py'):
            continue
            
        problem_name = problem_dir.name.split('__')[0]
        eval_file = problem_dir / "evaluation_run.json"
        
        if not eval_file.exists():
            error_problems.append(problem_name)
            continue
            
        try:
            with open(eval_file, 'r') as f:
                data = json.load(f)
            
            if data.get("status") != "finished":
                error_problems.append(problem_name)
                continue
                
            test_results = data.get("test_results", [])
            if not test_results:
                error_problems.append(problem_name)
                continue
                
            passed = sum(1 for test in test_results if test.get("status") == "passed")
            failed = sum(1 for test in test_results if test.get("status") == "failed")
            total = len(test_results)
            
            total_tests += total
            passed_tests += passed
            
            if failed == 0 and passed > 0:
                successful_problems.append(f"{problem_name}: {passed} tests passed")
            elif passed > 0:
                failed_problems.append(f"{problem_name}: {passed} passed, {failed} failed")
            else:
                failed_problems.append(f"{problem_name}: 0 passed, {failed} failed")
                
        except Exception as e:
            error_problems.append(problem_name)
    
    # Calculate success rate using production methodology
    total_problems = len(successful_problems) + len(failed_problems) + len(error_problems)
    production_success_rate = (len(successful_problems) / total_problems * 100) if total_problems > 0 else 0
    local_success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Passed tests: {passed_tests}")
    print(f"Total problems: {total_problems}")
    print(f"Problems with ALL tests passing: {len(successful_problems)}")
    print(f"")
    print(f"PRODUCTION Success Rate: {production_success_rate:.1f}% (problems with all tests passing)")
    print(f"LOCAL Success Rate: {local_success_rate:.1f}% (passed tests / total tests)")
    print(f"Target: 50%")
    print(f"Successful problems: {len(successful_problems)}")
    print(f"Failed problems: {len(failed_problems)}")
    print(f"Error problems: {len(error_problems)}")
    
    print(f"\n{'='*60}")
    print("SUCCESSFUL PROBLEMS")
    print(f"{'='*60}")
    for problem in successful_problems:
        print(f"✓ {problem}")
    
    print(f"\n{'='*60}")
    print("FAILED PROBLEMS")
    print(f"{'='*60}")
    for problem in failed_problems:
        print(f"✗ {problem}")
    
    print(f"\n{'='*60}")
    print("ERROR PROBLEMS")
    print(f"{'='*60}")
    for problem in error_problems:
        print(f"✗ {problem}: Error or no test results")
    
    return {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'total_problems': total_problems,
        'production_success_rate': production_success_rate,
        'local_success_rate': local_success_rate,
        'successful_problems': successful_problems,
        'failed_problems': failed_problems,
        'error_problems': error_problems
    }

if __name__ == "__main__":
    analyze_latest_results()
