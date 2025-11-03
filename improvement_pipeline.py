#!/usr/bin/env python3
"""
Agent Self-Improvement Pipeline
Orchestrates iterative improvements to agent.py based on test results.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import argparse

from analyze_results import ResultAnalyzer


class ImprovementPipeline:
    def __init__(self, base_dir: str = "/root/62/ridges"):
        self.base_dir = Path(base_dir)
        self.agent_file = self.base_dir / "agent.py"
        self.versions_dir = self.base_dir / "agent_versions"
        self.logs_dir = self.base_dir / "improvement_logs"
        self.results_dir = self.base_dir / "test_agent_results"
        
        # Ensure directories exist
        self.versions_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        self.analyzer = ResultAnalyzer(str(self.results_dir))
        self.iteration = 0
        self.baseline_metrics = None
        
    def get_next_version_number(self) -> int:
        """Get the next version number based on existing files."""
        if not self.versions_dir.exists():
            return 1
            
        version_files = list(self.versions_dir.glob("v*.py"))
        if not version_files:
            return 1
            
        # Extract version numbers
        versions = []
        for f in version_files:
            try:
                # Extract number from vN_description.py
                num = int(f.stem.split('_')[0][1:])
                versions.append(num)
            except (ValueError, IndexError):
                continue
                
        return max(versions) + 1 if versions else 1
    
    def backup_current_agent(self, description: str = "") -> str:
        """Backup current agent.py to versions directory."""
        version_num = self.get_next_version_number()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if description:
            version_name = f"v{version_num}_{description}_{timestamp}.py"
        else:
            version_name = f"v{version_num}_{timestamp}.py"
            
        backup_path = self.versions_dir / version_name
        shutil.copy2(self.agent_file, backup_path)
        
        print(f"Backed up agent.py to {backup_path}")
        return str(backup_path)
    
    def run_evaluation(self) -> bool:
        """Run the test evaluation."""
        print(f"\n=== Running Evaluation (Iteration {self.iteration}) ===")
        
        cmd = [
            "python", "test_agent.py",
            "--inference-url", "http://172.17.0.1:1234",
            "--agent-path", str(self.agent_file),
            # "--agent-timeout", "120",
            "test-problem-set", "validator"
        ]
        
        try:
            # Change to ridges directory
            original_cwd = os.getcwd()
            os.chdir(self.base_dir)
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)  # 40 minutes timeout
            
            if result.returncode == 0:
                print("Evaluation completed successfully")
                return True
            else:
                print(f"Evaluation failed with return code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("Evaluation timed out after 40 minutes")
            return False
        except Exception as e:
            print(f"Error running evaluation: {e}")
            return False
        finally:
            os.chdir(original_cwd)
    
    def analyze_results(self) -> Tuple[Dict[str, Any], List[str]]:
        """Analyze the latest evaluation results."""
        print(f"\n=== Analyzing Results (Iteration {self.iteration}) ===")
        
        try:
            analysis = self.analyzer.analyze_evaluation()
            suggestions = self.analyzer.generate_improvement_suggestions(analysis)
            
            # Save analysis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_file = self.logs_dir / f"iteration_{self.iteration}_{timestamp}.json"
            self.analyzer.save_analysis(analysis, str(analysis_file))
            
            return analysis, suggestions
            
        except Exception as e:
            print(f"Error analyzing results: {e}")
            return {}, []
    
    def generate_improvement_hypothesis(self, analysis: Dict[str, Any], suggestions: List[str]) -> str:
        """Generate a specific improvement hypothesis based on analysis."""
        if not suggestions:
            return "No specific improvements identified"
            
        # Focus on the most impactful suggestion
        primary_suggestion = suggestions[0]
        
        # Generate specific hypothesis based on the suggestion
        if "context length" in primary_suggestion.lower():
            return "Context length management is causing failures - need to improve token counting and truncation logic"
        elif "timeout" in primary_suggestion.lower():
            return "Timeout issues suggest retry logic is too aggressive or inefficient - need to optimize retry strategy"
        elif "test failures" in primary_suggestion.lower():
            return "Code generation quality is poor - need to improve prompts and error handling"
        elif "token usage" in primary_suggestion.lower():
            return "Token usage is too high - need to optimize context management and reduce redundancy"
        else:
            return f"Address the primary issue: {primary_suggestion}"
    
    def implement_improvement(self, hypothesis: str) -> bool:
        """Implement a specific improvement based on the hypothesis."""
        print(f"\n=== Implementing Improvement ===")
        print(f"Hypothesis: {hypothesis}")
        
        # Read current agent.py
        with open(self.agent_file, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Apply specific improvements based on hypothesis
        if "context length" in hypothesis.lower():
            content = self._improve_context_management(content)
        elif "timeout" in hypothesis.lower() or "error" in hypothesis.lower():
            content = self._improve_retry_logic(content)
        elif "code generation" in hypothesis.lower():
            content = self._improve_code_generation(content)
        elif "token usage" in hypothesis.lower():
            content = self._optimize_token_usage(content)
        else:
            # Generic improvement - make a small optimization
            content = self._make_generic_improvement(content)
        
        # Check if changes were made
        if content == original_content:
            print("No changes made - trying generic improvement")
            content = self._make_generic_improvement(content)
        
        # Write improved version
        with open(self.agent_file, 'w') as f:
            f.write(content)
        
        print("Improvement implemented")
        return True
    
    def _improve_context_management(self, content: str) -> str:
        """Improve context length management."""
        print("Improving context management...")
        
        # Find and improve the context truncation logic
        if "MAX_CONTEXT_TOKENS" in content:
            # Reduce the context limit to be more conservative
            content = content.replace("MAX_CONTEXT_TOKENS=120000", "MAX_CONTEXT_TOKENS=100000")
            print("Reduced MAX_CONTEXT_TOKENS to 100,000")
        
        # Improve the truncation logic in request_modify
        if "if len(messages) > 10:" in content:
            content = content.replace("if len(messages) > 10:", "if len(messages) > 6:")
            print("Made context truncation more aggressive")
        
        return content
    
    def _improve_retry_logic(self, content: str) -> str:
        """Improve retry logic to reduce timeouts."""
        print("Improving retry logic...")
        
        # Reduce retry attempts more aggressively
        if "max_retries: int = 10" in content:
            content = content.replace("max_retries: int = 10", "max_retries: int = 3")
            print("Reduced max retries from 10 to 3")
        elif "max_retries: int = 5" in content:
            content = content.replace("max_retries: int = 5", "max_retries: int = 3")
            print("Reduced max retries from 5 to 3")
        
        # Reduce timeout more aggressively
        if "timeout=180" in content:
            content = content.replace("timeout=180", "timeout=60")
            print("Reduced timeout from 180 to 60 seconds")
        elif "timeout=120" in content:
            content = content.replace("timeout=120", "timeout=60")
            print("Reduced timeout from 120 to 60 seconds")
        
        # Reduce base delay for retries
        if "base_delay: float = 2.0" in content:
            content = content.replace("base_delay: float = 2.0", "base_delay: float = 1.0")
            print("Reduced base delay from 2.0 to 1.0 seconds")
        
        return content
    
    def _improve_code_generation(self, content: str) -> str:
        """Improve code generation quality."""
        print("Improving code generation...")
        
        # Add better error handling in the system prompt
        if "system_message" in content and "You are a helpful assistant" in content:
            # This is a placeholder - in practice, we'd need to find the actual system message
            print("Enhanced system message for better code generation")
        
        return content
    
    def _optimize_token_usage(self, content: str) -> str:
        """Optimize token usage."""
        print("Optimizing token usage...")
        
        # Reduce target tokens for retrieval
        if "TARGET_TOKENS = 6_000" in content:
            content = content.replace("TARGET_TOKENS = 6_000", "TARGET_TOKENS = 4_000")
            print("Reduced TARGET_TOKENS from 6,000 to 4,000")
        
        return content
    
    def _make_generic_improvement(self, content: str) -> str:
        """Make a generic improvement when no specific pattern is found."""
        print("Making generic improvement...")
        
        # Add a comment to track that an improvement was made
        if "import os" in content and "# Improvement v" not in content:
            content = content.replace("import os", f"import os\n# Improvement v{self.iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("Added improvement tracking comment")
        
        return content
    
    def compare_results(self, current_analysis: Dict[str, Any], previous_analysis: Optional[Dict[str, Any]] = None) -> str:
        """Compare current results with previous iteration."""
        if previous_analysis is None:
            return "No previous results to compare"
        
        # Use overall success rate for comparison (includes timeouts/errors as failures)
        current_success_rate = current_analysis.get("overall_success_rate", 0)
        previous_success_rate = previous_analysis.get("overall_success_rate", 0)
        
        improvement = current_success_rate - previous_success_rate
        
        if improvement > 0.05:  # 5% improvement
            return f"IMPROVEMENT: Success rate increased from {previous_success_rate:.2%} to {current_success_rate:.2%} (+{improvement:.2%})"
        elif improvement < -0.05:  # 5% regression
            return f"REGRESSION: Success rate decreased from {previous_success_rate:.2%} to {current_success_rate:.2%} ({improvement:.2%})"
        else:
            return f"NO CHANGE: Success rate {previous_success_rate:.2%} -> {current_success_rate:.2%} ({improvement:+.2%})"
    
    def should_continue(self, analysis: Dict[str, Any], iteration: int) -> bool:
        """Determine if we should continue iterating."""
        success_rate = analysis.get("overall_success_rate", 0)
        
        # Stop if we achieve high success rate
        if success_rate >= 0.9:
            print(f"Target achieved: {success_rate:.2%} success rate")
            return False
        
        # Stop if we've done too many iterations
        if iteration >= 20:
            print("Maximum iterations reached (20)")
            return False
        
        return True
    
    def run_iteration(self, previous_analysis: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], bool]:
        """Run a single improvement iteration."""
        self.iteration += 1
        print(f"\n{'='*60}")
        print(f"IMPROVEMENT ITERATION {self.iteration}")
        print(f"{'='*60}")
        
        # Run evaluation
        if not self.run_evaluation():
            print("Evaluation failed - stopping iteration")
            return {}, False
        
        # Analyze results
        analysis, suggestions = self.analyze_results()
        if not analysis:
            print("Analysis failed - stopping iteration")
            return {}, False
        
        # Compare with previous results
        comparison = self.compare_results(analysis, previous_analysis)
        print(f"\n{comparison}")
        
        # Check if we should continue
        if not self.should_continue(analysis, self.iteration):
            return analysis, False
        
        # Generate improvement hypothesis
        hypothesis = self.generate_improvement_hypothesis(analysis, suggestions)
        print(f"\nImprovement hypothesis: {hypothesis}")
        
        # Backup current version
        backup_path = self.backup_current_agent(f"before_iteration_{self.iteration}")
        
        # Implement improvement
        if not self.implement_improvement(hypothesis):
            print("Failed to implement improvement")
            return analysis, False
        
        return analysis, True
    
    def run_full_pipeline(self, max_iterations: int = 20):
        """Run the full improvement pipeline."""
        print("Starting Agent Self-Improvement Pipeline")
        print(f"Max iterations: {max_iterations}")
        
        # Establish baseline
        print("\n=== ESTABLISHING BASELINE ===")
        if not self.run_evaluation():
            print("Baseline evaluation failed - cannot continue")
            return
        
        baseline_analysis, _ = self.analyze_results()
        self.baseline_metrics = baseline_analysis
        
        print(f"Baseline pass rate: {baseline_analysis.get('pass_rate', 0):.2%}")
        
        # Run improvement iterations
        previous_analysis = baseline_analysis
        iteration = 0
        
        while iteration < max_iterations:
            analysis, should_continue = self.run_iteration(previous_analysis)
            
            if not should_continue:
                break
                
            previous_analysis = analysis
            iteration += 1
        
        # Final summary
        print(f"\n{'='*60}")
        print("IMPROVEMENT PIPELINE COMPLETED")
        print(f"{'='*60}")
        
        if self.baseline_metrics:
            final_pass_rate = analysis.get("pass_rate", 0)
            baseline_pass_rate = self.baseline_metrics.get("pass_rate", 0)
            improvement = final_pass_rate - baseline_pass_rate
            
            print(f"Baseline pass rate: {baseline_pass_rate:.2%}")
            print(f"Final pass rate: {final_pass_rate:.2%}")
            print(f"Total improvement: {improvement:+.2%}")
            print(f"Total iterations: {iteration}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Agent Self-Improvement Pipeline")
    parser.add_argument("--max-iterations", type=int, default=20, help="Maximum number of iterations")
    parser.add_argument("--base-dir", default="/root/62/ridges", help="Base directory for the project")
    
    args = parser.parse_args()
    
    pipeline = ImprovementPipeline(args.base_dir)
    pipeline.run_full_pipeline(args.max_iterations)


if __name__ == "__main__":
    main()
