#!/usr/bin/env python3

import os
import json
from pathlib import Path

def analyze_latest_results():
    # Find the latest directory
    test_results_dir = Path('test_agent_results')
    latest_dir = max(test_results_dir.glob('*'), key=lambda x: x.stat().st_mtime)
    print(f'Analyzing results from: {latest_dir}')

    # Analyze results
    total_problems = 0
    successful_problems = 0
    failed_problems = 0
    error_problems = 0
    total_tests = 0
    passed_tests = 0

    successful_list = []
    failed_list = []
    error_list = []

    for problem_dir in latest_dir.iterdir():
        if not problem_dir.is_dir():
            continue
        
        problem_name = problem_dir.name.split('__')[0]
        total_problems += 1
        
        eval_file = problem_dir / 'evaluation_run.json'
        if not eval_file.exists():
            error_problems += 1
            error_list.append(problem_name)
            continue
        
        try:
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
            
            test_results = eval_data.get('test_results', [])
            if not test_results:
                error_problems += 1
                error_list.append(problem_name)
                continue
            
            problem_passed = 0
            problem_failed = 0
            
            for test in test_results:
                total_tests += 1
                status = test.get('status', 'unknown')
                if status == 'pass':
                    passed_tests += 1
                    problem_passed += 1
                elif status == 'fail':
                    problem_failed += 1
            
            if problem_failed == 0 and problem_passed > 0:
                successful_problems += 1
                successful_list.append(f'{problem_name}: {problem_passed} passed, {problem_failed} failed')
            elif problem_passed > 0 or problem_failed > 0:
                failed_problems += 1
                failed_list.append(f'{problem_name}: {problem_passed} passed, {problem_failed} failed')
            else:
                error_problems += 1
                error_list.append(problem_name)
        
        except Exception as e:
            print(f'Error processing {problem_name}: {e}')
            error_problems += 1
            error_list.append(problem_name)

    # Calculate success rates
    production_success_rate = (successful_problems / total_problems * 100) if total_problems > 0 else 0
    local_success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

    print('\n' + '='*60)
    print('FINAL RESULTS')
    print('='*60)
    print(f'Total tests: {total_tests}')
    print(f'Passed tests: {passed_tests}')
    print(f'Total problems: {total_problems}')
    print(f'Problems with ALL tests passing: {successful_problems}')
    print()
    print(f'PRODUCTION Success Rate: {production_success_rate:.1f}% (problems with all tests passing)')
    print(f'LOCAL Success Rate: {local_success_rate:.1f}% (passed tests / total tests)')
    print(f'Target: 50%')
    print(f'Successful problems: {successful_problems}')
    print(f'Failed problems: {failed_problems}')
    print(f'Error problems: {error_problems}')

    print('\n' + '='*60)
    print('SUCCESSFUL PROBLEMS')
    print('='*60)
    for item in successful_list:
        print(f'✓ {item}')

    print('\n' + '='*60)
    print('FAILED PROBLEMS')
    print('='*60)
    for item in failed_list:
        print(f'✗ {item}')

    print('\n' + '='*60)
    print('ERROR PROBLEMS')
    print('='*60)
    for item in error_list:
        print(f'✗ {item}: Error or no test results')

if __name__ == '__main__':
    analyze_latest_results()
