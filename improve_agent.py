#!/usr/bin/env python3
"""
Main entry point for the agent improvement system.
Provides a simple interface to run different improvement tasks.
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Agent Improvement System")
    parser.add_argument("command", choices=["status", "analyze", "test", "run", "full"], 
                       help="Command to run")
    parser.add_argument("--iterations", type=int, default=3, 
                       help="Number of iterations for 'run' command")
    parser.add_argument("--max-iterations", type=int, default=20, 
                       help="Maximum iterations for 'full' command")
    
    args = parser.parse_args()
    
    if args.command == "status":
        print("Checking system status...")
        import subprocess
        subprocess.run([sys.executable, "check_status.py"])
        
    elif args.command == "analyze":
        print("Analyzing current results...")
        import subprocess
        subprocess.run([sys.executable, "analyze_results.py"])
        
    elif args.command == "test":
        print("Running limited test (3 iterations)...")
        import subprocess
        subprocess.run([sys.executable, "start_improvement.py"])
        
    elif args.command == "run":
        print(f"Running improvement pipeline ({args.iterations} iterations)...")
        from improvement_pipeline import ImprovementPipeline
        pipeline = ImprovementPipeline()
        pipeline.run_full_pipeline(max_iterations=args.iterations)
        
    elif args.command == "full":
        print(f"Running full improvement pipeline (up to {args.max_iterations} iterations)...")
        from improvement_pipeline import ImprovementPipeline
        pipeline = ImprovementPipeline()
        pipeline.run_full_pipeline(max_iterations=args.max_iterations)

if __name__ == "__main__":
    main()
