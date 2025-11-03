#!/usr/bin/env python3
"""
Check the current status of the improvement pipeline.
Shows latest results and available versions.
"""

import json
from pathlib import Path
from datetime import datetime
from analyze_results import ResultAnalyzer

def main():
    print("=== AGENT IMPROVEMENT PIPELINE STATUS ===\n")
    
    # Check directories
    base_dir = Path("/root/62/ridges")
    versions_dir = base_dir / "agent_versions"
    logs_dir = base_dir / "improvement_logs"
    results_dir = base_dir / "test_agent_results"
    
    print(f"Base directory: {base_dir}")
    print(f"Versions directory: {versions_dir} ({'exists' if versions_dir.exists() else 'missing'})")
    print(f"Logs directory: {logs_dir} ({'exists' if logs_dir.exists() else 'missing'})")
    print(f"Results directory: {results_dir} ({'exists' if results_dir.exists() else 'missing'})")
    
    # Check agent versions
    if versions_dir.exists():
        versions = list(versions_dir.glob("v*.py"))
        print(f"\nAgent versions available: {len(versions)}")
        for v in sorted(versions):
            size = v.stat().st_size
            mtime = datetime.fromtimestamp(v.stat().st_mtime)
            print(f"  {v.name} ({size:,} bytes, {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
    
    # Check improvement logs
    if logs_dir.exists():
        logs = list(logs_dir.glob("*.json"))
        print(f"\nAnalysis logs available: {len(logs)}")
        for log in sorted(logs)[-3:]:  # Show last 3
            size = log.stat().st_size
            mtime = datetime.fromtimestamp(log.stat().st_mtime)
            print(f"  {log.name} ({size:,} bytes, {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
    
    # Check latest results
    if results_dir.exists():
        eval_dirs = [d for d in results_dir.iterdir() if d.is_dir() and '__' in d.name]
        if eval_dirs:
            latest_eval = sorted(eval_dirs, key=lambda x: x.name)[-1]
            print(f"\nLatest evaluation: {latest_eval.name}")
            
            # Count problems
            problem_dirs = [d for d in latest_eval.iterdir() if d.is_dir()]
            print(f"Problems in latest evaluation: {len(problem_dirs)}")
            
            # Quick analysis
            try:
                analyzer = ResultAnalyzer(str(results_dir))
                analysis = analyzer.analyze_evaluation(latest_eval)
                
                print(f"\nLatest Results:")
                print(f"  Total problems: {analysis.get('total_problems', 0)}")
                print(f"  Passed: {analysis.get('summary', {}).get('passed', 0)}")
                print(f"  Failed: {analysis.get('summary', {}).get('failed', 0)}")
                print(f"  Timeout: {analysis.get('summary', {}).get('timeout', 0)}")
                print(f"  Error: {analysis.get('summary', {}).get('error', 0)}")
                print(f"  Pass rate: {analysis.get('pass_rate', 0):.2%}")
                
                # Show improvement suggestions
                suggestions = analyzer.generate_improvement_suggestions(analysis)
                if suggestions:
                    print(f"\nTop improvement suggestions:")
                    for i, suggestion in enumerate(suggestions[:3], 1):
                        print(f"  {i}. {suggestion}")
                        
            except Exception as e:
                print(f"Error analyzing latest results: {e}")
    
    print(f"\n=== QUICK COMMANDS ===")
    print(f"Run analysis: python analyze_results.py")
    print(f"Start improvement: python start_improvement.py")
    print(f"Full pipeline: python improvement_pipeline.py")

if __name__ == "__main__":
    main()
