#!/usr/bin/env python3
"""
Quick start script for the improvement pipeline.
This runs a limited number of iterations for testing.
"""

import sys
from improvement_pipeline import ImprovementPipeline

def main():
    print("Starting Agent Improvement Pipeline (Limited Run)")
    print("This will run 3 iterations to test the system")
    
    pipeline = ImprovementPipeline()
    
    # Run a limited test with 3 iterations
    pipeline.run_full_pipeline(max_iterations=3)

if __name__ == "__main__":
    main()
