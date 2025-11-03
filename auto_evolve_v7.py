import os
import sys
import re
import json
import time
import glob
import subprocess
from typing import List, Dict, Any, Tuple

from cursor_api_client import CursorAPIClient


ROOT = "/root/62/ridges"
AGENT_PATH = f"{ROOT}/my-agents/v7.py"
RESULTS_DIR = f"{ROOT}/test_agent_results"
TEST_AGENT_CLI = f"{ROOT}/test_agent.py"
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://172.17.0.1:1234")
PROBLEM_SET = os.getenv("PROBLEM_SET", "all-polyglot")
SOURCE_BRANCH = "cursor-work"
TARGET_BRANCH = "cursor-work-11"

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


def _run_single_problem(inference_url: str, problem_name: str) -> str:
    """Run test_agent for a single problem; return the results directory path used for this run."""
    cmd = [
        sys.executable,
        TEST_AGENT_CLI,
        "--inference-url",
        inference_url,
        "--agent-path",
        AGENT_PATH,
        "test-problem",
        problem_name,
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
        "- Evaluation logs showing which tests failed\n"
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
    
    followup_text = (
        f"The v7 agent failed on problem '{problem_name}'. Here's the context:\n\n"
        f"METRICS:\n"
        f"- Tests passed: {metrics['pass']}\n"
        f"- Tests failed: {metrics['fail']}\n"
        f"- Tests skipped: {metrics['skip']}\n"
        f"- Failure categories: {json.dumps(cats, indent=2)}\n\n"
        f"FILE PATHS:\n"
        f"- Problem directory: {problem_dir_rel}\n"
        f"  * Contains: instructions.md (problem statement - IS available at runtime), main.py (skeleton), tests.py (NOT available at runtime, only for evaluation context)\n"
        f"- Evaluation run directory: {run_dir_rel}\n"
        f"  * Contains: agent_logs.txt (what v7 did during execution), eval_logs.txt (which tests failed and why), evaluation_run.json\n"
        f"- Agent file to improve: {agent_rel_path}\n\n"
        f"TASK:\n"
        f"Improve my-agents/v7.py to handle this failure case. Remember:\n"
        f"- CRITICAL: The agent must remain GENERIC - no problem-specific logic, always  double check the code is generic before committing, remove any problem-specific logic\n"
        f"- IMPORTANT: The agent code deals with Python ONLY - no other programming languages\n"
        f"- At runtime, only problem_statement (instruction.md) and main.py skeleton are available\n"
        f"- tests.py is NOT available at runtime, so don't rely on test specifics\n"
        f"- Decide whether small tweaks (prompt/params) or major changes (architecture/flow) are needed\n"
        f"- Focus on the root cause: why did the agent fail on this problem?\n"
        f"- Apply the minimal change that fixes this while maintaining genericity\n"
        f"- ALWAYS clean up the agent code: remove unused imports, functions, and variables\n"
        f"- ALWAYS check the code is generic before committing\n"
        f"- ALWAYS try to add verbose logging to the code to help debug the issues\n"
        f"- Make the code look professional: follow Python best practices, add proper docstrings, ensure consistent formatting, and improve readability\n\n"
        f"Review the files in the paths above to understand the failure context. Return the complete updated my-agents/v7.py file."
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


def evolve_over_problems(max_attempts_per_problem: int = 50) -> None:
    """Evolve v7 by running problems and using Cursor agent follow-ups for improvements."""
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
    
    problem_names = _load_problem_set(PROBLEM_SET)
    if not problem_names:
        print("No problems found.")
        sys.exit(1)
    
    for idx, name in enumerate(problem_names):
        print(f"\n=== Problem {idx+1}/{len(problem_names)}: {name} ===")
        
        attempts = 0
        while attempts < max_attempts_per_problem:
            attempts += 1
            print(f"[LOOP] Attempt {attempts}/{max_attempts_per_problem}")

            # delete test_agent_results directory
            subprocess.run(["rm", "-rf", f"{ROOT}/test_agent_results"], cwd=ROOT, check=False)
            
            # Run the problem
            eval_dir = _run_single_problem(INFERENCE_URL, name)
            
            # Commit test results so agent can have access
            try:
                subprocess.run(["git", "add", "."], cwd=ROOT, check=False)
                subprocess.run(
                    ["git", "commit", "-m", f"auto-evolve: test results for {name} attempt {attempts}"],
                    cwd=ROOT,
                    check=False,
                )
                subprocess.run(["git", "push"], cwd=ROOT, check=False)
            except Exception as e:
                print(f"[WARN] Git operations failed: {e}")
            
            run_dir = _find_problem_run_dir(eval_dir, name)
            metrics, failures = _aggregate_results(run_dir)
            print(f"[METRICS] pass={metrics['pass']} fail={metrics['fail']} skip={metrics['skip']}")
            
            # Success - move to next problem
            if metrics["fail"] == 0 and metrics["pass"] > 0:
                print("[OK] All tests passed; moving to next problem")
                break
            
            # Failure - add follow-up to Cursor agent with full context
            print("[EVOLVE] Tests failed; adding follow-up to Cursor agent...")
            
            # Get problem directory path
            problem_dir_path = ""
            polyglot_path = f"{ROOT}/evaluator/datasets/polyglot"
            potential_dirs = [
                os.path.join(polyglot_path, name),
                os.path.join(polyglot_path, name, "repo"),
            ]
            for prob_dir in potential_dirs:
                if os.path.exists(prob_dir):
                    problem_dir_path = prob_dir
                    break
            
            # Add follow-up and wait for completion
            _add_followup_with_failure_context(
                client=client,
                agent_id=agent_id,
                problem_name=name,
                problem_dir=problem_dir_path,
                run_dir=run_dir,
                metrics=metrics,
                failures=failures,
            )
            
            # After follow-up completes, checkout agent file from target_branch and apply it
            print(f"[CHECKOUT] Follow-up completed; fetching and checking out {AGENT_PATH} from branch '{TARGET_BRANCH}'...")
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
                print(f"[WARN] Error during checkout: {e}; stopping evolution for this problem")
                break
            
            if not proposal:
                print("[WARN] No proposal available; stopping evolution for this problem")
                break
            
            # is_valid, reason = _genericity_checks(proposal)
            # if not is_valid:
            #     print(f"[REJECT] Proposed v7 contains non-generic/problem-specific content: {reason}")
            #     break
            
            # Apply the proposal
            _write(AGENT_PATH, proposal)
            current_v7 = proposal  # Update for next iteration
            
            try:
                subprocess.run(["git", "add", AGENT_PATH], cwd=ROOT, check=False)
                subprocess.run(
                    ["git", "commit", "-m", f"auto-evolve v7 for {name} attempt {attempts}"],
                    cwd=ROOT,
                    check=False,
                )
                subprocess.run(["git", "push"], cwd=ROOT, check=False)
            except Exception:
                pass

        else:
            print("[STOP] Reached attempt cap; moving to next problem")


def main() -> None:
    evolve_over_problems()


if __name__ == "__main__":
    main()


