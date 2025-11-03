#!/usr/bin/env python3
"""
Analyze the latest test results to calculate success rate
"""

import json
import os
import glob

def analyze_latest_results():
    """Analyze the latest test results"""
    
    # Find the latest results directory
    results_base = "/root/62/ridges/test_agent_results"
    latest_dir = None
    
    for item in sorted(os.listdir(results_base), reverse=True):
        if os.path.isdir(os.path.join(results_base, item)):
            latest_dir = os.path.join(results_base, item)
            break
    
    if not latest_dir:
        print("No results directory found")
        return
    
    print(f"Analyzing results from: {latest_dir}")
    
    # Find all evaluation_run.json files
    pattern = os.path.join(latest_dir, "*", "evaluation_run.json")
    result_files = glob.glob(pattern)
    
    print(f"Found {len(result_files)} result files")
    
    total_tests = 0
    passed_tests = 0
    successful_problems = []
    failed_problems = []
    error_problems = []
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            problem_name = data.get('problem_name', 'unknown')
            test_results = data.get('test_results', [])
            status = data.get('status', 'unknown')
            
            if status == 'finished' and test_results:
                problem_passed = 0
                problem_failed = 0
                
                for test in test_results:
                    test_status = test.get('status', 'unknown')
                    if test_status == 'pass':
                        problem_passed += 1
                        passed_tests += 1
                    elif test_status == 'fail':
                        problem_failed += 1
                    total_tests += 1
                
                if problem_passed > 0 and problem_failed == 0:
                    successful_problems.append((problem_name, problem_passed))
                    print(f"✓ {problem_name}: {problem_passed} tests passed")
                else:
                    failed_problems.append((problem_name, problem_passed, problem_failed))
                    print(f"✗ {problem_name}: {problem_passed} passed, {problem_failed} failed")
            else:
                error_problems.append(problem_name)
                print(f"✗ {problem_name}: Error or no test results")
                
        except Exception as e:
            print(f"Error reading {result_file}: {e}")
    
    # Calculate success rate using production methodology
    # Production: problems_with_all_tests_passing / total_problems
    total_problems = len(successful_problems) + len(failed_problems) + len(error_problems)
    production_success_rate = (len(successful_problems) / total_problems * 100) if total_problems > 0 else 0
    
    # Also calculate local methodology for comparison
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
    print(f"Target: 45%")
    print(f"Successful problems: {len(successful_problems)}")
    print(f"Failed problems: {len(failed_problems)}")
    print(f"Error problems: {len(error_problems)}")
    
    print(f"\n{'='*60}")
    print("SUCCESSFUL PROBLEMS")
    print(f"{'='*60}")
    for problem_name, passed_count in successful_problems:
        print(f"✓ {problem_name}: {passed_count} tests passed")
    
    print(f"\n{'='*60}")
    print("FAILED PROBLEMS")
    print(f"{'='*60}")
    for problem_name, passed_count, failed_count in failed_problems:
        print(f"✗ {problem_name}: {passed_count} passed, {failed_count} failed")
    
    print(f"\n{'='*60}")
    print("ERROR PROBLEMS")
    print(f"{'='*60}")
    for problem_name in error_problems:
        print(f"✗ {problem_name}: Error or no test results")
    
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
