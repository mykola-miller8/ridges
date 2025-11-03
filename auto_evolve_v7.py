import os
import sys
import re
import json
import time
import glob
import pathlib
import subprocess
from typing import List, Dict, Any, Tuple

from cursor_api_client import CursorAPIClient


ROOT = "/root/62/ridges"
AGENT_PATH = f"{ROOT}/my-agents/v7.py"
RESULTS_DIR = f"{ROOT}/test_agent_results"
TEST_AGENT_CLI = f"{ROOT}/test_agent.py"
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://172.17.0.1:1234")
PROBLEM_SET = os.getenv("PROBLEM_SET", "all-polyglot")

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


def _anonymize(text: str) -> str:
    if not text:
        return ""
    s = text
    # Drop fenced code blocks
    s = re.sub(r"```[\s\S]*?```", "<CODE_BLOCK>", s)
    # Redact file paths and .py names
    s = re.sub(r"/[^\s\n]+", "<PATH>", s)
    s = re.sub(r"\\[^\s\n]+", "<PATH>", s)
    s = re.sub(r"[A-Za-z0-9_\-]+\.py\b", "<PYFILE>", s)
    # Redact test names
    s = re.sub(r"\btest_[A-Za-z0-9_]+\b", "<TEST>", s)
    # Collapse long hex/ids
    s = re.sub(r"\b[0-9a-f]{8,}\b", "<ID>", s)
    # Trim
    return s[:6000]


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


def _genericity_checks(code: str) -> bool:
    if not code:
        return False
    forbidden = [
        "swebench",
        "polyglot",
        "problem_name",
        "tests.py",
        "evaluation_run.json",
        "/evaluator/datasets/",
    ]
    if any(w in code.lower() for w in forbidden):
        return False
    if re.search(r"\btest_[A-Za-z0-9_]+\b", code):
        return False
    return True


def _rewrite_v7_with_cursor(current_v7: str, meta_feedback: Dict[str, Any]) -> str:
    """Use Cursor Cloud Agent API to rewrite v7.py.
    
    Uses the CursorAPIClient utility class to launch an agent, poll for completion,
    and extract the rewritten code.
    """
    if not CURSOR_API_URL or not CURSOR_API_KEY:
        print("[BUILDER] Cursor API not configured (CURSOR_API_URL/KEY missing)")
        return ""
    
    try:
        # Initialize client
        client = CursorAPIClient(
            api_url=CURSOR_API_URL,
            api_key=CURSOR_API_KEY,
            default_repo_url=None,  # Will auto-detect from git
        )
        
        # Construct the prompt for the Cursor agent
        prompt_text = (
            "Rewrite my-agents/v7.py to improve robustness and generic solving ability. "
            "CRITICAL constraints:\n"
            "- Keep the exact entrypoint signature: agent_main(input_dict, repo_dir='repo', test_mode=False) -> str\n"
            "- Use only the existing inference gateway (INFERENCE_URL/SANDBOX_PROXY_URL env vars) for LLM calls\n"
            "- NEVER embed any problem-specific strings, dataset names (polyglot, swebench), expected outputs, or test names\n"
            "- Focus ONLY on generic mechanisms: prompt format, code parsing, retries/backoff, patch generation, syntax validation\n"
            "- Return the complete new file content (not a patch or diff).\n\n"
            "Current my-agents/v7.py:\n```python\n" + current_v7[:20000] + "\n```\n\n"
            "Meta-feedback (anonymized):\n" + json.dumps(meta_feedback, indent=2, ensure_ascii=False) + "\n\n"
            "Analyze the failures, identify root causes, and rewrite the entire file to address issues while maintaining genericity."
        )
        
        # Status callback for logging
        def status_callback(status_data: Dict[str, Any]) -> None:
            status = status_data.get("status") or status_data.get("state") or "unknown"
            print(f"[BUILDER] Agent status: {status}")
        
        # Launch and wait for completion
        print("[BUILDER] Launching Cursor agent to rewrite v7.py...")
        agent_id, final_data = client.launch_and_wait(
            prompt_text=prompt_text,
            skip_reviewer_request=True,
            auto_create_pr=False,
            max_polls=1000,
            poll_interval=10,
            status_callback=status_callback,
            branch_name='cursor'
        )
        
        print(f"[BUILDER] Agent completed with ID: {agent_id}")
        
        # Extract the rewritten code
        content = CursorAPIClient.extract_file_content(final_data, "v7.py")
        
        if content:
            print(f"[BUILDER] Extracted v7.py rewrite (len={len(content)})")
            return content
        else:
            print("[BUILDER] Could not extract code from agent response")
            print(f"[BUILDER] Response keys: {list(final_data.keys())}")
            return ""
        
    except ValueError as e:
        print(f"[BUILDER] Configuration error: {e}")
        return ""
    except TimeoutError as e:
        print(f"[BUILDER] Timeout: {e}")
        return ""
    except RuntimeError as e:
        print(f"[BUILDER] Agent failed: {e}")
        return ""
    except Exception as e:
        print(f"[BUILDER] Unexpected error: {type(e).__name__}: {e}")
        import traceback
        print(f"[BUILDER] Traceback: {traceback.format_exc()}")
        return ""


def evolve_over_problems(max_attempts_per_problem: int = 50) -> None:
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
            eval_dir = _run_single_problem(INFERENCE_URL, name)
            run_dir = _find_problem_run_dir(eval_dir, name)
            metrics, failures = _aggregate_results(run_dir)
            print(f"[METRICS] pass={metrics['pass']} fail={metrics['fail']} skip={metrics['skip']}")
            if metrics["fail"] == 0 and metrics["pass"] > 0:
                print("[OK] All tests passed; moving to next problem")
                break

            # Prepare anonymized meta-feedback
            cats = _categorize_failures(failures)
            logs_tail = _read_agent_logs_tail(run_dir)
            anon_logs = _anonymize(logs_tail)
            meta_feedback = {
                "metrics": metrics,
                "failure_categories": cats,
                "anonymized_agent_logs_tail": anon_logs,
            }

            current_v7 = _read(AGENT_PATH)
            proposal = _rewrite_v7_with_cursor(current_v7, meta_feedback)
            if not proposal:
                print("[WARN] No proposal from Cursor; stopping evolution for this problem")
                break
            if not _genericity_checks(proposal):
                print("[REJECT] Proposed v7 contains non-generic/problem-specific content; skipping")
                break
            _write(AGENT_PATH, proposal)
            try:
                subprocess.run(["git", "add", AGENT_PATH], cwd=ROOT)
                subprocess.run(["git", "commit", "-m", f"auto-evolve v7 for {name} attempt {attempts}"], cwd=ROOT)
            except Exception:
                pass

        else:
            print("[STOP] Reached attempt cap; moving to next problem")


def main() -> None:
    evolve_over_problems()


if __name__ == "__main__":
    main()


