#!/usr/bin/env python3
"""
Analyze test results for the enabled problem type (ALGORITHM_IMPLEMENTATION)
"""

import json
import os
import glob
from pathlib import Path

def analyze_test_results():
    results_dir = "/root/62/ridges/test_agent_results/2025-10-28__v1.py__843b939a-ff47-43a4-8ab9-93fc643a813d"
    
    # Find all problem directories
    problem_dirs = glob.glob(f"{results_dir}/*")
    
    algorithm_problems = []
    other_problems = []
    
    for problem_dir in problem_dirs:
        problem_name = os.path.basename(problem_dir)
        
        # Check if this is an algorithm implementation problem
        # Based on the problem names, these are typically algorithm problems
        algorithm_keywords = [
            'affine-cipher', 'beer-song', 'book-store', 'grep', 'hangman', 
            'list-ops', 'phone-number', 'pig-latin', 'poker', 'pov', 
            'proverb', 'react', 'rest-api', 'robot-name', 'scale-generator'
        ]
        
        is_algorithm = any(keyword in problem_name for keyword in algorithm_keywords)
        
        if is_algorithm:
            algorithm_problems.append(problem_dir)
        else:
            other_problems.append(problem_dir)
    
    print(f"=== ANALYSIS REPORT ===")
    print(f"Total problems: {len(problem_dirs)}")
    print(f"Algorithm Implementation problems: {len(algorithm_problems)}")
    print(f"Other problem types: {len(other_problems)}")
    print()
    
    # Analyze algorithm implementation problems
    print("=== ALGORITHM IMPLEMENTATION PROBLEMS ANALYSIS ===")
    
    successful_problems = []
    failed_problems = []
    empty_patch_problems = []
    
    for problem_dir in algorithm_problems:
        problem_name = os.path.basename(problem_dir)
        
        # Check for evaluation_run.json
        eval_file = os.path.join(problem_dir, "evaluation_run.json")
        agent_logs = os.path.join(problem_dir, "agent_logs.txt")
        
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
            
            patch = eval_data.get('patch', '')
            test_results = eval_data.get('test_results', [])
            
            if not patch.strip():
                empty_patch_problems.append(problem_name)
                print(f"❌ {problem_name}: Empty patch")
            else:
                # Count test results (handle None case)
                if test_results is None:
                    test_results = []
                passed = sum(1 for test in test_results if test.get('status') == 'pass')
                failed = sum(1 for test in test_results if test.get('status') == 'fail')
                skipped = sum(1 for test in test_results if test.get('status') == 'skip')
                
                if passed > 0:
                    successful_problems.append((problem_name, passed, failed, skipped))
                    print(f"✅ {problem_name}: {passed} passed, {failed} failed, {skipped} skipped")
                else:
                    failed_problems.append((problem_name, passed, failed, skipped))
                    print(f"❌ {problem_name}: {passed} passed, {failed} failed, {skipped} skipped")
        else:
            # Check agent logs for empty patch
            if os.path.exists(agent_logs):
                with open(agent_logs, 'r') as f:
                    logs = f.read()
                if "No solution generated" in logs or "0 line(s) of patch" in logs:
                    empty_patch_problems.append(problem_name)
                    print(f"❌ {problem_name}: No solution generated")
                else:
                    print(f"⚠️  {problem_name}: No evaluation file found")
    
    print()
    print("=== SUMMARY STATISTICS ===")
    print(f"Total Algorithm Problems: {len(algorithm_problems)}")
    print(f"Successful (at least 1 test passed): {len(successful_problems)}")
    print(f"Failed (0 tests passed): {len(failed_problems)}")
    print(f"Empty patches: {len(empty_patch_problems)}")
    
    if len(algorithm_problems) > 0:
        success_rate = len(successful_problems) / len(algorithm_problems) * 100
        print(f"Success Rate: {success_rate:.1f}%")
    
    print()
    print("=== SUCCESSFUL PROBLEMS ===")
    for name, passed, failed, skipped in successful_problems:
        print(f"✅ {name}: {passed} passed, {failed} failed, {skipped} skipped")
    
    print()
    print("=== FAILED PROBLEMS ===")
    for name, passed, failed, skipped in failed_problems:
        print(f"❌ {name}: {passed} passed, {failed} failed, {skipped} skipped")
    
    print()
    print("=== EMPTY PATCH PROBLEMS ===")
    for name in empty_patch_problems:
        print(f"❌ {name}: No patch generated")
    
    return {
        'total': len(algorithm_problems),
        'successful': len(successful_problems),
        'failed': len(failed_problems),
        'empty_patches': len(empty_patch_problems),
        'success_rate': len(successful_problems) / len(algorithm_problems) * 100 if len(algorithm_problems) > 0 else 0,
        'successful_problems': successful_problems,
        'failed_problems': failed_problems,
        'empty_patch_problems': empty_patch_problems
    }

if __name__ == "__main__":
    results = analyze_test_results()