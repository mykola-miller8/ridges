# Agent Self-Improvement Pipeline

This system automatically improves the `agent.py` file through iterative testing and refinement based on evaluation results.

## Quick Start

### Run Limited Test (3 iterations)
```bash
cd /root/62/ridges
python start_improvement.py
```

### Run Full Pipeline (up to 20 iterations)
```bash
cd /root/62/ridges
python improvement_pipeline.py --max-iterations 20
```

### Analyze Current Results
```bash
cd /root/62/ridges
python analyze_results.py
```

## How It Works

1. **Baseline Establishment**: Runs initial evaluation to establish baseline metrics
2. **Iterative Improvement**: For each iteration:
   - Runs evaluation on screener-1 problem set
   - Analyzes results to identify failure patterns
   - Generates improvement hypothesis
   - Implements targeted fixes to agent.py
   - Compares results with previous iteration
   - Decides whether to keep or revert changes

3. **Automatic Stopping**: Stops when:
   - Pass rate reaches 90%
   - No improvement for 3 consecutive iterations
   - Maximum iterations (20) reached

## Directory Structure

```
ridges/
├── agent.py                          # Current agent (gets modified)
├── agent_versions/                   # All agent versions
│   ├── v1_baseline.py               # Original backup
│   ├── v2_context_fix_20251024.py   # Version with context fixes
│   └── ...
├── improvement_logs/                 # Analysis reports
│   ├── iteration_1_20251024.json    # Detailed analysis
│   └── analysis_20251024.json       # Summary analysis
├── test_agent_results/              # Evaluation results
└── improvement_pipeline.py         # Main orchestration script
```

## Key Files

- **`improvement_pipeline.py`**: Main orchestration script
- **`analyze_results.py`**: Analyzes test results and generates suggestions
- **`start_improvement.py`**: Quick test script (3 iterations)
- **`agent_versions/`**: All agent versions for rollback
- **`improvement_logs/`**: Detailed analysis and metrics

## Metrics Tracked

- **Pass Rate**: Percentage of problems that pass all tests
- **Failure Categories**: Timeout, test failure, error, etc.
- **Performance**: Average time per problem, max tokens used
- **Error Patterns**: Common error messages and their frequency
- **Context Issues**: Token limit violations and context management problems

## Safety Features

- **Version Control**: Every change creates a new version in `agent_versions/`
- **Automatic Rollback**: Reverts changes if they cause regression
- **Detailed Logging**: All changes and results are logged
- **Non-destructive**: Never modifies test harness or problem sets

## Example Output

```
=== EVALUATION ANALYSIS ===
Total problems: 10
Passed: 4
Failed: 4
Timeout: 0
Error: 2
Pass rate: 50.00%
Avg time: 1608.4s
Max tokens: 0
Context errors: 10

=== IMPROVEMENT SUGGESTIONS ===
1. Fix context length management - too many context errors detected
2. Improve code generation quality - many test failures
3. Address common error: Traceback (most recent call last):
4. Pass rate needs improvement - focus on most common failure modes
```

## Improvement Types

The system focuses on these improvement areas:

1. **Context Management**: Fix token counting and truncation
2. **Retry Logic**: Optimize timeout and retry strategies  
3. **Code Generation**: Improve prompts and error handling
4. **Token Optimization**: Reduce redundancy and improve efficiency
5. **Error Recovery**: Better handling of test failures

## Manual Intervention

If you need to manually intervene:

1. **Stop the pipeline**: Ctrl+C during execution
2. **Revert to specific version**: 
   ```bash
   cp agent_versions/vN_description.py agent.py
   ```
3. **Resume from specific iteration**: Modify the pipeline code
4. **Analyze specific results**: Use `analyze_results.py` on any evaluation

## Monitoring Progress

Check progress by examining:
- `improvement_logs/` for detailed analysis
- `agent_versions/` for all agent versions
- Console output for real-time progress
- Pass rate improvements over iterations

## Expected Results

After 10-20 iterations, you should see:
- Systematic improvement in pass rates
- Clear documentation of what changes helped
- Ability to backtest any version
- Understanding of which behaviors correlate with success

The system is designed to be autonomous but provides full visibility into the improvement process.
