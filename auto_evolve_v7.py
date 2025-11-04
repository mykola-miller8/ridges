import os
import sys
import re
import json
import time
import glob
import subprocess
import random
from typing import List, Dict, Any, Tuple

from cursor_api_client import CursorAPIClient


ROOT = "/root/62/ridges"
AGENT_PATH = f"{ROOT}/my-agents/v7.py"
RESULTS_DIR = f"{ROOT}/test_agent_results"
TEST_AGENT_CLI = f"{ROOT}/test_agent.py"
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://172.17.0.1:1234")
PROBLEM_SET = os.getenv("PROBLEM_SET", "all-polyglot")
SOURCE_BRANCH = "cursor-work"
TARGET_BRANCH = f"cursor-work-{random.randint(10000, 99999)}"

# v7 solving uses the inference gateway (set in test_agent CLI). For rewriting v7 itself,
# we use the Cursor API only (no public LLM).
CURSOR_API_URL = os.getenv("CURSOR_API_URL", "https://api.cursor.com")
CURSOR_API_KEY = os.getenv("CURSOR_API_KEY", "key_d52e0c4d44b712e3c13ab15ee2d0713d4e683a8f4900699f210be3673e181fa3")


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _load_problem_set(name: str) -> List[str]:
    # Prefer using the problem sets file that test_agent already uses
    sets_path = f"{ROOT}/test_agent_problem_sets.json"
    try:
        data = json.loads(_read(sets_path))
        problems = data.get(name)
        if isinstance(problems, list) and problems:
            return problems
    except Exception:
        pass
    # Fallback: treat given name as a single problem
    return [name]


def _run_problem_set(inference_url: str, problem_set_name: str) -> str:
    """Run test_agent for a problem set; return the results directory path used for this run."""
    cmd = [
        sys.executable,
        TEST_AGENT_CLI,
        "--inference-url",
        inference_url,
        "--agent-path",
        AGENT_PATH,
        "test-problem-set",
        problem_set_name,
    ]
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=False)
    # Wait briefly for filesystem flush
    time.sleep(1.5)
    # Identify the latest evaluation dir for this agent
    dirs = sorted(
        glob.glob(f"{RESULTS_DIR}/*__{os.path.basename(AGENT_PATH)}__*"),
        key=os.path.getmtime,
    )
    return dirs[-1] if dirs else ""


def _find_problem_run_dir(eval_dir: str, problem_name: str) -> str:
    if not eval_dir:
        return ""
    subdirs = sorted(glob.glob(f"{eval_dir}/{problem_name}__*"), key=os.path.getmtime)
    return subdirs[-1] if subdirs else ""


def _find_all_problem_run_dirs(eval_dir: str) -> List[Tuple[str, str]]:
    """Find all problem run directories in the eval_dir.
    
    Returns list of (problem_name, run_dir_path) tuples.
    """
    if not eval_dir or not os.path.exists(eval_dir):
        return []
    
    problem_dirs = []
    for item in os.listdir(eval_dir):
        item_path = os.path.join(eval_dir, item)
        if not os.path.isdir(item_path):
            continue
        
        # Extract problem name from directory name (format: problem_name__timestamp)
        if "__" in item:
            problem_name = item.split("__")[0]
            problem_dirs.append((problem_name, item_path))
    
    return problem_dirs


def _aggregate_results(run_dir: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    metrics = {"pass": 0, "fail": 0, "skip": 0}
    failures: List[Dict[str, Any]] = []
    if not run_dir:
        return metrics, failures
    ev_path = os.path.join(run_dir, "evaluation_run.json")
    try:
        data = json.loads(_read(ev_path))
    except Exception:
        return metrics, failures
    tests = data.get("test_results") or []
    for t in tests:
        st = t.get("status")
        if st == "pass":
            metrics["pass"] += 1
        elif st == "fail":
            metrics["fail"] += 1
            failures.append({
                "name": t.get("name"),
                "message": (t.get("message") or "")[:800],
            })
        else:
            metrics["skip"] += 1
    return metrics, failures


def _aggregate_all_results(problem_run_dirs: List[Tuple[str, str]]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Aggregate results from all problem run directories.
    
    Returns:
        (total_metrics, problem_results) where:
        - total_metrics: aggregated metrics across all problems
        - problem_results: dict mapping problem_name to {"metrics": ..., "failures": ..., "run_dir": ...}
    """
    total_metrics = {"pass": 0, "fail": 0, "skip": 0}
    problem_results: Dict[str, Dict[str, Any]] = {}
    
    for problem_name, run_dir in problem_run_dirs:
        metrics, failures = _aggregate_results(run_dir)
        total_metrics["pass"] += metrics["pass"]
        total_metrics["fail"] += metrics["fail"]
        total_metrics["skip"] += metrics["skip"]
        
        problem_results[problem_name] = {
            "metrics": metrics,
            "failures": failures,
            "run_dir": run_dir,
        }
    
    return total_metrics, problem_results


def _read_agent_logs_tail(run_dir: str, max_chars: int = 8000) -> str:
    if not run_dir:
        return ""
    p = os.path.join(run_dir, "agent_logs.txt")
    txt = _read(p)
    if not txt:
        return ""
    return txt[-max_chars:]


def _read_eval_logs_tail(run_dir: str, max_chars: int = 8000) -> str:
    """Read evaluation logs from run directory."""
    if not run_dir:
        return ""
    p = os.path.join(run_dir, "eval_logs.txt")
    txt = _read(p)
    if not txt:
        return ""
    return txt[-max_chars:]


def _get_problem_context(problem_name: str) -> Dict[str, str]:
    """Get problem statement and tests.py for a given problem.
    
    Returns dict with 'problem_statement' and 'tests_py' keys.
    """
    # Try to find problem directory in polyglot dataset
    polyglot_path = f"{ROOT}/evaluator/datasets/polyglot"
    problem_dirs = [
        os.path.join(polyglot_path, problem_name),
        os.path.join(polyglot_path, problem_name, "repo"),
    ]
    
    context = {"problem_statement": "", "tests_py": ""}
    
    for prob_dir in problem_dirs:
        if not os.path.exists(prob_dir):
            continue
        
        # Look for instructions.md (polyglot format) or similar
        for inst_file in ["instructions.md", "instruction.md", "README.md", "problem.md"]:
            inst_path = os.path.join(prob_dir, inst_file)
            if os.path.exists(inst_path):
                context["problem_statement"] = _read(inst_path)[:10000]  # Limit size
                break
        
        # Look for tests.py
        tests_path = os.path.join(prob_dir, "tests.py")
        if os.path.exists(tests_path):
            context["tests_py"] = _read(tests_path)[:15000]  # Limit size
            break
        
        # Also check in repo subdirectory
        repo_tests = os.path.join(prob_dir, "repo", "tests.py")
        if os.path.exists(repo_tests):
            context["tests_py"] = _read(repo_tests)[:15000]
            break
    
    return context


def _categorize_failures(failures: List[Dict[str, Any]]) -> Dict[str, int]:
    cats: Dict[str, int] = {}
    for f in failures:
        msg = (f.get("message") or "").lower()
        key = "other"
        if "syntaxerror" in msg or "syntax error" in msg:
            key = "syntax"
        elif "patch" in msg or "apply" in msg or "diff" in msg:
            key = "patch_format"
        elif "importerror" in msg or "module" in msg:
            key = "import"
        elif "assert" in msg or "expected" in msg:
            key = "assertion"
        cats[key] = cats.get(key, 0) + 1
    return cats


def _genericity_checks(code: str) -> Tuple[bool, str]:
    """Check if code is generic. Returns (is_valid, reason)."""
    if not code:
        return False, "Code is empty"
    forbidden = [
        "swebench",
        "polyglot",
        "problem_name",
        "tests.py",
        "evaluation_run.json",
        "/evaluator/datasets/",
    ]
    code_lower = code.lower()
    for word in forbidden:
        if word in code_lower:
            return False, f"Contains forbidden word/phrase: '{word}'"
    
    test_match = re.search(r"\btest_[A-Za-z0-9_]+\b", code)
    if test_match:
        return False, f"Contains test function name pattern: '{test_match.group()}'"
    
    return True, ""


def _launch_initial_cursor_agent(client: CursorAPIClient) -> str:
    """Launch the initial Cursor agent for v7 evolution.
    
    Returns the agent_id.
    """
    agent_rel_path = os.path.relpath(AGENT_PATH, ROOT)
    initial_prompt = (
        "You are tasked with continuously improving my-agents/v7.py, a generic code-solving agent. "
        "Your goal is to make it more robust and effective at solving diverse programming problems.\n\n"
        "CRITICAL CONSTRAINTS (MUST FOLLOW):\n"
        "- Keep the exact entrypoint signature: agent_main(input_dict, repo_dir='repo', test_mode=False) -> str\n"
        "- Use only the existing inference gateway (INFERENCE_URL/SANDBOX_PROXY_URL env vars) for LLM calls\n"
        "- NEVER embed any problem-specific strings, dataset names (polyglot, swebench), expected outputs, or test names\n"
        "- The agent must remain completely generic - usable for ANY problem domain\n"
        "- IMPORTANT: The agent code deals with Python ONLY - no other programming languages\n"
        "- Focus on generic mechanisms: prompt format, code parsing, retries/backoff, patch generation, syntax validation\n\n"
        "IMPORTANT RUNTIME CONTEXT:\n"
        "- At runtime, the agent ONLY has access to:\n"
        "  * problem_statement (instruction.md - text description of what to implement)\n"
        "  * main.py (skeleton with function/class signatures only)\n"
        "- tests.py is NOT available at runtime (only during evaluation)\n"
        "- The agent cannot see expected outputs or test code when solving\n\n"
        "You will receive follow-up instructions with:\n"
        "- Agent execution logs showing what happened during a failed solve\n"
        "- Evaluation logs which is inside test_agent_results directory showing which tests failed\n"
        "- The problem statement and tests.py (for context only - tests aren't available at runtime)\n\n"
        "When improving v7.py, you can decide between:\n"
        "- Small tweaks: prompt wording, parameter adjustments, retry logic, parsing improvements\n"
        "- Major changes: architecture redesign, new strategies, flow restructuring\n"
        "Choose the approach that best addresses the root cause while maintaining genericity.\n\n"
        f"The agent file to improve is located at: {agent_rel_path}\n\n"
        "Start by reviewing the current implementation. You'll receive follow-ups with specific failure cases to address."
    )
    
    print("[BUILDER] Launching initial Cursor agent for v7 evolution...")
    launch_data = client.launch_agent(
        prompt_text=initial_prompt,
        skip_reviewer_request=True,
        auto_create_pr=True,
        source_branch=SOURCE_BRANCH,
        target_branch=TARGET_BRANCH
    )
    
    agent_id = launch_data.get("id") or launch_data.get("agent_id") or launch_data.get("agentId")
    if not agent_id:
        raise ValueError("No agent ID returned from launch")
    
    print(f"[BUILDER] Agent launched with ID: {agent_id}")
    return agent_id


def _add_followup_with_failure_context(
    client: CursorAPIClient,
    agent_id: str,
    problem_name: str,
    problem_dir: str,
    run_dir: str,
    metrics: Dict[str, Any],
    failures: List[Dict[str, Any]],
) -> str:
    """Add a follow-up to the Cursor agent with failure context.
    
    Returns the rewritten v7.py code, or empty string if failed.
    """
    cats = _categorize_failures(failures)
    
    # Get relative paths
    problem_dir_rel = os.path.relpath(problem_dir, ROOT) if problem_dir else ""
    run_dir_rel = os.path.relpath(run_dir, ROOT) if run_dir else ""
    agent_rel_path = os.path.relpath(AGENT_PATH, ROOT)
    
    # Read full content of files
    main_py_content = ""
    agent_logs_content = ""
    eval_logs_content = ""
    
    # Read main.py from problem directory
    if problem_dir:
        main_py_paths = [
            os.path.join(problem_dir, "main.py"),
            os.path.join(problem_dir, "repo", "main.py"),
        ]
        for main_path in main_py_paths:
            if os.path.exists(main_path):
                main_py_content = _read(main_path)
                break
    
    # Read agent_logs.txt from run directory
    if run_dir:
        agent_logs_path = os.path.join(run_dir, "agent_logs.txt")
        if os.path.exists(agent_logs_path):
            agent_logs_content = _read(agent_logs_path)
    
    # Read eval_logs.txt from run directory
    if run_dir:
        eval_logs_path = os.path.join(run_dir, "eval_logs.txt")
        if os.path.exists(eval_logs_path):
            eval_logs_content = _read(eval_logs_path)
    
    followup_text = (
        f"The v7 agent failed on problem '{problem_name}'. Here's the context:\n\n"
        f"METRICS:\n"
        f"- Tests passed: {metrics['pass']}\n"
        f"- Tests failed: {metrics['fail']}\n"
        f"- Tests skipped: {metrics['skip']}\n"
        f"- Failure categories: {json.dumps(cats, indent=2)}\n\n"
        f"=== MAIN.PY (the code the agent generated) ===\n"
        f"{main_py_content if main_py_content else '(main.py not found or empty)'}\n\n"
        f"=== AGENT LOGS (what the agent did during execution) ===\n"
        f"{agent_logs_content if agent_logs_content else '(agent_logs.txt not found or empty)'}\n\n"
        f"=== EVAL LOGS (which tests failed and why) ===\n"
        f"{eval_logs_content if eval_logs_content else '(eval_logs.txt not found or empty)'}\n\n"
        f"=== ADDITIONAL CONTEXT ===\n"
        f"- Problem directory: {problem_dir_rel}\n"
        f"- Evaluation run directory: {run_dir_rel}\n"
        f"- Agent file to improve: {agent_rel_path}\n\n"
        f"TASK:\n"
        f"Improve my-agents/v7.py to handle this failure case. Remember:\n"
        f"- CRITICAL: The agent must remain GENERIC - no problem-specific logic, always double check the code is generic before committing, remove any problem-specific logic\n"
        f"- IMPORTANT: The agent code deals with Python ONLY - no other programming languages\n"
        f"- At runtime, only problem_statement (instruction.md) and main.py skeleton are available\n"
        f"- tests.py is NOT available at runtime, so don't rely on test specifics\n"
        f"- The agent logs show exactly what happened during execution - use them to understand the failure\n"
        f"- The eval logs show which tests failed and why - this helps identify what the agent got wrong\n"
        f"- Decide whether small tweaks (prompt/params) or major changes (architecture/flow) are needed\n"
        f"- Focus on the root cause: why did the agent fail on this problem?\n"
        f"- Apply the minimal change that fixes this while maintaining genericity\n"
        f"- ALWAYS clean up the agent code: remove unused imports, functions, and variables\n"
        f"- ALWAYS check the code is generic before committing\n"
        f"- ALWAYS try to add verbose logging to the code to help debug the issues\n"
        f"- Make the code look professional: follow Python best practices, add proper docstrings, ensure consistent formatting, and improve readability\n\n"
        f"Return the complete updated my-agents/v7.py file."
    )
    
    print(f"[BUILDER] Adding follow-up to agent {agent_id}...")
    client.add_followup(agent_id, followup_text)
    
    # Poll until completion
    def status_callback(status_data: Dict[str, Any]) -> None:
        status = status_data.get("status") or status_data.get("state") or "unknown"
        # print(f"[BUILDER] Agent status: {status}")
    
    try:
        final_data = client.poll_until_complete(
            agent_id=agent_id,
            max_polls=6000000,
            poll_interval=10,
            status_callback=status_callback,
        )
        
        # Extract the rewritten code
        content = CursorAPIClient.extract_file_content(final_data, "v7.py")
        return content if content else ""
        
    except (TimeoutError, RuntimeError) as e:
        print(f"[BUILDER] Error waiting for agent: {e}")
        return ""


def _add_followup_with_all_failures(
    client: CursorAPIClient,
    agent_id: str,
    eval_dir: str,
    problem_results: Dict[str, Dict[str, Any]],
    total_metrics: Dict[str, Any],
) -> str:
    """Add a follow-up to the Cursor agent analyzing all failed problems together.
    
    Args:
        client: Cursor API client
        agent_id: ID of the cursor agent
        eval_dir: Evaluation directory containing all test results
        problem_results: Dict mapping problem_name to {"metrics": ..., "failures": ..., "run_dir": ...}
        total_metrics: Aggregated metrics across all problems
    
    Returns the rewritten v7.py code, or empty string if failed.
    """
    agent_rel_path = os.path.relpath(AGENT_PATH, ROOT)
    eval_dir_rel = os.path.relpath(eval_dir, ROOT) if eval_dir else ""
    
    # Collect all failures across all problems
    all_failures = []
    all_failure_cats = {}
    failed_problem_data = []
    
    for problem_name, problem_data in problem_results.items():
        if problem_data["metrics"]["fail"] > 0:
            failures = problem_data["failures"]
            all_failures.extend(failures)
            
            # Categorize failures for this problem
            cats = _categorize_failures(failures)
            for cat, count in cats.items():
                all_failure_cats[cat] = all_failure_cats.get(cat, 0) + count
            
            failed_problem_data.append({
                "name": problem_name,
                "metrics": problem_data["metrics"],
                "failures": failures,
                "run_dir": problem_data["run_dir"],
            })
    
    # Aggregate failure categories
    total_failure_cats = _categorize_failures(all_failures)
    
    # Build comprehensive context from all failed problems
    # Read samples of agent logs and eval logs from multiple problems (limit to avoid token limits)
    agent_logs_samples = []
    eval_logs_samples = []
    main_py_samples = []
    max_samples = 5  # Limit number of problems to include full logs from
    
    for i, problem_info in enumerate(failed_problem_data[:max_samples]):
        problem_name = problem_info["name"]
        run_dir = problem_info["run_dir"]
        
        # Get problem directory
        problem_dir_path = ""
        polyglot_path = f"{ROOT}/evaluator/datasets/polyglot"
        potential_dirs = [
            os.path.join(polyglot_path, problem_name),
            os.path.join(polyglot_path, problem_name, "repo"),
        ]
        for prob_dir in potential_dirs:
            if os.path.exists(prob_dir):
                problem_dir_path = prob_dir
                break
        
        # Read main.py
        main_py_content = ""
        if problem_dir_path:
            main_py_paths = [
                os.path.join(problem_dir_path, "main.py"),
                os.path.join(problem_dir_path, "repo", "main.py"),
            ]
            for main_path in main_py_paths:
                if os.path.exists(main_path):
                    main_py_content = _read(main_path)
                    if main_py_content:
                        # Limit size to avoid token limits
                        main_py_content = main_py_content[:3000]
                    break
        
        # Read agent logs
        agent_logs_content = ""
        if run_dir:
            agent_logs_path = os.path.join(run_dir, "agent_logs.txt")
            if os.path.exists(agent_logs_path):
                agent_logs_content = _read(agent_logs_path)
                if agent_logs_content:
                    # Limit size - take tail of logs (most recent/relevant)
                    agent_logs_content = agent_logs_content[-5000:]
        
        # Read eval logs
        eval_logs_content = ""
        if run_dir:
            eval_logs_path = os.path.join(run_dir, "eval_logs.txt")
            if os.path.exists(eval_logs_path):
                eval_logs_content = _read(eval_logs_path)
                if eval_logs_content:
                    # Limit size - take tail of logs
                    eval_logs_content = eval_logs_content[-5000:]
        
        main_py_samples.append((problem_name, main_py_content))
        agent_logs_samples.append((problem_name, agent_logs_content))
        eval_logs_samples.append((problem_name, eval_logs_content))
    
    # Build the follow-up text
    followup_text = (
        f"The v7 agent failed on {len(failed_problem_data)} problem(s). Here's the comprehensive context:\n\n"
        f"=== OVERALL METRICS ===\n"
        f"- Total tests passed: {total_metrics['pass']}\n"
        f"- Total tests failed: {total_metrics['fail']}\n"
        f"- Total tests skipped: {total_metrics['skip']}\n"
        f"- Number of failed problems: {len(failed_problem_data)}\n"
        f"- Aggregated failure categories: {json.dumps(total_failure_cats, indent=2)}\n\n"
        f"=== FAILED PROBLEMS SUMMARY ===\n"
    )
    
    # Add summary for each failed problem
    for problem_info in failed_problem_data:
        problem_name = problem_info["name"]
        metrics = problem_info["metrics"]
        failures = problem_info["failures"]
        problem_cats = _categorize_failures(failures)
        
        followup_text += (
            f"\nProblem: {problem_name}\n"
            f"  - Tests passed: {metrics['pass']}, failed: {metrics['fail']}, skipped: {metrics['skip']}\n"
            f"  - Failure categories: {json.dumps(problem_cats, indent=4)}\n"
            f"  - Sample failures:\n"
        )
        # Include first 3 failures from this problem
        for failure in failures[:3]:
            failure_name = failure.get("name", "unknown")
            failure_msg = (failure.get("message") or "")[:200]  # Limit message length
            followup_text += f"    * {failure_name}: {failure_msg}\n"
    
    # Add detailed logs from sample problems
    followup_text += f"\n=== DETAILED LOGS FROM SAMPLE PROBLEMS (showing {min(len(agent_logs_samples), max_samples)} problems) ===\n"
    
    # for problem_name, main_py_content in main_py_samples:
    #     followup_text += (
    #         f"\n--- Problem: {problem_name} - MAIN.PY (generated code) ---\n"
    #         f"{main_py_content if main_py_content else '(main.py not found or empty)'}\n"
    #     )
    
    for problem_name, agent_logs_content in agent_logs_samples:
        followup_text += (
            f"\n--- Problem: {problem_name} - AGENT LOGS (what the agent did) ---\n"
            f"{agent_logs_content if agent_logs_content else '(agent_logs.txt not found or empty)'}\n"
        )
    
    for problem_name, eval_logs_content in eval_logs_samples:
        followup_text += (
            f"\n--- Problem: {problem_name} - EVAL LOGS (which tests failed and why) ---\n"
            f"{eval_logs_content if eval_logs_content else '(eval_logs.txt not found or empty)'}\n"
        )
    
    followup_text += (
        f"\n=== ADDITIONAL CONTEXT ===\n"
        f"- Evaluation directory: {eval_dir_rel}\n"
        f"- Agent file to improve: {agent_rel_path}\n"
        f"- Test results are in: test_agent_results/\n\n"
        f"TASK:\n"
        f"Analyze ALL the failed problems above and improve my-agents/v7.py to handle these failure cases comprehensively. Look for:\n"
        f"- Common patterns across multiple problems (these indicate systemic issues in the agent)\n"
        f"- Recurring failure categories (syntax errors, assertion failures, import errors, etc.)\n"
        f"- Similar mistakes the agent made across different problems\n"
        f"- Architecture or prompt issues that cause failures across multiple problems\n\n"
        f"Remember:\n"
        f"- CRITICAL: The agent must remain GENERIC - no problem-specific logic, always double check the code is generic before committing, remove any problem-specific logic\n"
        f"- IMPORTANT: The agent code deals with Python ONLY - no other programming languages\n"
        f"- At runtime, only problem_statement (instruction.md) and main.py skeleton are available\n"
        f"- tests.py is NOT available at runtime, so don't rely on test specifics\n"
        f"- The agent logs show exactly what happened during execution - use them to understand the failures\n"
        f"- The eval logs show which tests failed and why - this helps identify what the agent got wrong\n"
        f"- Focus on ROOT CAUSES that affect multiple problems, not individual problem quirks\n"
        f"- Prioritize fixes that will help the most problems (systemic improvements over edge case handling)\n"
        f"- Decide whether small tweaks (prompt/params) or major changes (architecture/flow) are needed\n"
        f"- Apply changes that address the common patterns while maintaining genericity\n"
        f"- ALWAYS clean up the agent code: remove unused imports, functions, and variables\n"
        f"- ALWAYS check the code is generic before committing\n"
        f"- ALWAYS try to add verbose logging to the code to help debug the issues\n"
        f"- Make the code look professional: follow Python best practices, add proper docstrings, ensure consistent formatting, and improve readability\n\n"
        f"Return the complete updated my-agents/v7.py file."
    )
    
    print(f"[BUILDER] Adding comprehensive follow-up to agent {agent_id} analyzing {len(failed_problem_data)} failed problems...")
    client.add_followup(agent_id, followup_text)
    
    # Poll until completion
    def status_callback(status_data: Dict[str, Any]) -> None:
        status = status_data.get("status") or status_data.get("state") or "unknown"
        # print(f"[BUILDER] Agent status: {status}")
    
    try:
        final_data = client.poll_until_complete(
            agent_id=agent_id,
            max_polls=6000000,
            poll_interval=10,
            status_callback=status_callback,
        )
        
        # Extract the rewritten code
        content = CursorAPIClient.extract_file_content(final_data, "v7.py")
        return content if content else ""
        
    except (TimeoutError, RuntimeError) as e:
        print(f"[BUILDER] Error waiting for agent: {e}")
        return ""


def evolve_over_problems(max_attempts_per_problem: int = 50) -> None:
    """Evolve v7 by running all problems at once and using Cursor agent follow-ups for improvements."""
    if not CURSOR_API_URL or not CURSOR_API_KEY:
        print("[BUILDER] Cursor API not configured (CURSOR_API_URL/KEY missing)")
        sys.exit(1)
    
    # Initialize Cursor client
    client = CursorAPIClient(
        api_url=CURSOR_API_URL,
        api_key=CURSOR_API_KEY,
        default_repo_url=None,
    )
    
    # Git add, commit, and push at the beginning
    print("[GIT] Staging, committing, and pushing initial state...")
    try:
        subprocess.run(["git", "add", "."], cwd=ROOT, check=False)
        subprocess.run(
            ["git", "commit", "-m", "auto-evolve: initial state before evolution"],
            cwd=ROOT,
            check=False,
        )
        subprocess.run(["git", "push"], cwd=ROOT, check=False)
        print("[GIT] Initial git operations completed")
    except Exception as e:
        print(f"[WARN] Git operations failed: {e}")
    
    # Launch initial agent
    agent_id = _launch_initial_cursor_agent(client)
    
    attempts = 0
    while attempts < max_attempts_per_problem:
        attempts += 1
        print(f"\n=== Attempt {attempts}/{max_attempts_per_problem} (all problems) ===")

        # delete test_agent_results directory
        # subprocess.run(["rm", "-rf", f"{ROOT}/test_agent_results"], cwd=ROOT, check=False)
        
        # Run all problems at once
        print(f"[RUN] Running test-problem-set all-polyglot...")
        eval_dir = _run_problem_set(INFERENCE_URL, "all-polyglot")
        
        # Commit test results so agent can have access
        try:
            subprocess.run(["git", "add", "."], cwd=ROOT, check=False)
            subprocess.run(
                ["git", "commit", "-m", f"auto-evolve: test results for all-polyglot attempt {attempts}"],
                cwd=ROOT,
                check=False,
            )
            subprocess.run(["git", "push"], cwd=ROOT, check=False)
            print("[GIT] Test results committed and pushed")
        except Exception as e:
            print(f"[WARN] Git operations failed: {e}")
        
        # NEW STEP: Checkout to remote TARGET_BRANCH and pull from remote SOURCE_BRANCH
        # This allows cursor agent to be able to read the logs
        print(f"[GIT] Checking out to {TARGET_BRANCH} and pulling from {SOURCE_BRANCH}...")
        try:
            # Fetch latest from remote
            fetch_result = subprocess.run(
                ["git", "fetch", "origin"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False
            )
            if fetch_result.returncode != 0:
                print(f"[WARN] Failed to fetch from origin: {fetch_result.stderr}")
            
            # Checkout to TARGET_BRANCH (create local branch if it doesn't exist, tracking remote)
            checkout_result = subprocess.run(
                ["git", "checkout", "-B", TARGET_BRANCH, f"origin/{TARGET_BRANCH}"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False
            )
            if checkout_result.returncode != 0:
                print(f"[WARN] Failed to checkout to {TARGET_BRANCH}: {checkout_result.stderr}")
            else:
                # Pull from SOURCE_BRANCH into current branch (TARGET_BRANCH)
                pull_result = subprocess.run(
                    ["git", "pull", "origin", SOURCE_BRANCH],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False
                )
                if pull_result.returncode != 0:
                    print(f"[WARN] Failed to pull from origin/{SOURCE_BRANCH}: {pull_result.stderr}")
                else:
                    print(f"[GIT] Successfully checked out to {TARGET_BRANCH} and pulled from {SOURCE_BRANCH}")
                    
                    # Checkout back to SOURCE_BRANCH
                    checkout_back_result = subprocess.run(
                        ["git", "checkout", SOURCE_BRANCH],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    if checkout_back_result.returncode != 0:
                        print(f"[WARN] Failed to checkout back to {SOURCE_BRANCH}: {checkout_back_result.stderr}")
                    else:
                        print(f"[GIT] Successfully checked out back to {SOURCE_BRANCH}")
        except Exception as e:
            print(f"[WARN] Git checkout/pull operations failed: {e}")
        
        # Get all problem run directories
        problem_run_dirs = _find_all_problem_run_dirs(eval_dir)
        total_metrics, problem_results = _aggregate_all_results(problem_run_dirs)
        
        print(f"[METRICS] Total - pass={total_metrics['pass']} fail={total_metrics['fail']} skip={total_metrics['skip']}")
        print(f"[METRICS] Problems processed: {len(problem_run_dirs)}")
        
        # Success - all problems passed
        if total_metrics["fail"] == 0 and total_metrics["pass"] > 0:
            print("[OK] All tests passed for all problems; evolution complete")
            break
        
        # Find failed problems
        failed_problems = []
        for problem_name, problem_data in problem_results.items():
            if problem_data["metrics"]["fail"] > 0:
                failed_problems.append((problem_name, problem_data))
        
        print(f"[EVOLVE] {len(failed_problems)} problems failed; analyzing all failures together...")
        
        # Add comprehensive follow-up analyzing all failed problems together
        _add_followup_with_all_failures(
            client=client,
            agent_id=agent_id,
            eval_dir=eval_dir,
            problem_results=problem_results,
            total_metrics=total_metrics,
        )
        
        # After all follow-ups complete, checkout agent file from target_branch and apply it
        print(f"[CHECKOUT] All follow-ups completed; fetching and checking out {AGENT_PATH} from branch '{TARGET_BRANCH}'...")
        try:
            # First fetch to get latest remote branches
            fetch_result = subprocess.run(
                ["git", "fetch", "origin"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False
            )
            if fetch_result.returncode != 0:
                print(f"[WARN] Failed to fetch from origin: {fetch_result.stderr}")
            
            # Get the file from remote branch
            agent_rel_path = os.path.relpath(AGENT_PATH, ROOT)
            result = subprocess.run(
                ["git", "show", f"origin/{TARGET_BRANCH}:{agent_rel_path}"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                print(f"[WARN] Failed to checkout from branch 'origin/{TARGET_BRANCH}'")
                if result.stderr:
                    print(f"[WARN] Error details: {result.stderr}")
                break
            
            proposal = result.stdout
        except Exception as e:
            print(f"[WARN] Error during checkout: {e}; stopping evolution")
            break
        
        if not proposal:
            print("[WARN] No proposal available; stopping evolution")
            break
        
        # is_valid, reason = _genericity_checks(proposal)
        # if not is_valid:
        #     print(f"[REJECT] Proposed v7 contains non-generic/problem-specific content: {reason}")
        #     break
        
        # Apply the proposal
        _write(AGENT_PATH, proposal)
        
        try:
            subprocess.run(["git", "add", AGENT_PATH], cwd=ROOT, check=False)
            subprocess.run(
                ["git", "commit", "-m", f"auto-evolve v7 for all-polyglot attempt {attempts}"],
                cwd=ROOT,
                check=False,
            )
            subprocess.run(["git", "push"], cwd=ROOT, check=False)
        except Exception:
            pass

    else:
        print("[STOP] Reached attempt cap")


def main() -> None:
    evolve_over_problems()


if __name__ == "__main__":
    main()


