import os
import re
import json
import uuid
import time
import subprocess
from typing import Any, Dict, List, Tuple

import requests


# v7 seed agent: generic code agent that proposes a full-file patch for main.py only.
# - Never embeds problem-specific constants or dataset names
# - Uses only the inference gateway exposed via INFERENCE_URL/SANDBOX_PROXY_URL
# - Returns a unified diff that replaces main.py entirely


DEFAULT_PROXY_URL = (
    os.getenv("INFERENCE_URL")
    or os.getenv("SANDBOX_PROXY_URL")
    or "http://172.17.0.1:1234"
)

AGENT_MODELS: List[str] = [
    # Order loosely favors code-focused models first
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "zai-org/GLM-4.5-FP8",
    "moonshotai/Kimi-K2-Instruct",
    "deepseek-ai/DeepSeek-V3-0324",
]


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _build_single_file_patch(filename: str, new_content: str) -> str:
    """Build a full-file unified diff for filename replacing its contents with new_content."""
    old = _read(filename)
    old_lines = old.splitlines()
    new_lines = new_content.splitlines()
    header = [
        f"diff --git a/{filename} b/{filename}",
        "index 0000000..1111111 100644",
        f"--- a/{filename}",
        f"+++ b/{filename}",
        f"@@ -1,{max(1, len(old_lines))} +1,{max(1, len(new_lines))} @@",
    ]
    body: List[str] = []
    if old_lines:
        body.extend(["-" + ln for ln in old_lines])
    else:
        body.append("-")
    if new_lines:
        body.extend(["+" + ln for ln in new_lines])
    else:
        body.append("+")
    return "\n".join(header + body) + "\n"


def _extract_main_py(response: str) -> str:
    """Extract the most likely complete main.py source from an LLM response.

    Heuristics:
    - Prefer a python fenced block beginning with '# main.py'
    - Otherwise consider any python fenced block
    - Otherwise consider any fenced block without language tag
    - Choose the block with best score based on header, presence of defs, and length
    """
    if not response:
        return ""

    candidates: List[str] = []
    # Strict: python block with explicit header
    for pat in [
        r"```python\s*\n#\s*main\.py\n([\s\S]*?)\n```",
        r"```python\s*\n([\s\S]*?)\n```",
        r"```\s*\n#\s*main\.py\n([\s\S]*?)\n```",
        r"```\s*\n([\s\S]*?)\n```",
    ]:
        blocks = re.findall(pat, response, re.DOTALL)
        for b in blocks:
            c = (b or "").strip()
            if c:
                candidates.append(c)

    if not candidates:
        # As a last resort, if response looks like raw code, take it as-is
        txt = (response or "").strip()
        if "def " in txt or "class " in txt:
            return txt
        return ""

    def score(block: str) -> float:
        s = 0.0
        first_line = block.splitlines()[0].strip() if block.splitlines() else ""
        if first_line.startswith("# main.py"):
            s += 3.0
        if "def " in block or "class " in block:
            s += 1.0
        # Prefer reasonably sized blocks
        s += min(len(block) / 2000.0, 5.0)
        return s

    best = max(candidates, key=score)
    return best.strip()


def _validate_python(source: str) -> Tuple[bool, str]:
    """Return (ok, error_message) after attempting to parse Python source with ast."""
    try:
        import ast
        ast.parse(source or "")
        return True, ""
    except Exception as e:  # SyntaxError or others
        return False, f"{type(e).__name__}: {e}"


def _build_refinement_messages(problem_statement: str, repo_summary: str, prev_code: str, error_msg: str, mode: str) -> List[Dict[str, str]]:
    system_msg = (
        "You are a senior Python engineer.\n"
        + ("Do not modify tests.py; only change main.py.\n" if mode == "tests_available" else "")
        + "Return ONLY one code block containing the complete main.py with a '# main.py' header.\n"
        "Format exactly as:\n```python\n# main.py\n[complete code]\n```\n"
        "No prose. Deterministic code.\n"
        "Preserve all existing public function/class names and signatures from main.py. Implement bodies only.\n"
        "Do not add external dependencies; use only Python standard library.\n"
        "No top-level execution or side effects; define functions/classes only.\n"
    )
    user_msg = (
        "Your previous output could not be parsed due to a syntax error.\n"
        f"Error: {error_msg}\n\n"
        "Problem Statement (trimmed if long):\n" + (problem_statement or "")[:12000] + "\n\n"
        "Repository Summary:\n" + repo_summary + "\n\n"
        "Previous attempt:\n```python\n# main.py\n" + (prev_code or "") + "\n```\n\n"
        "Produce a corrected, fully parseable complete main.py following the required format."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _call_llm(messages: List[Dict[str, str]], run_id: str, attempt: int, timeout_s: int = 240) -> str:
    url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
    headers = {"Content-Type": "application/json"}
    model = AGENT_MODELS[attempt % len(AGENT_MODELS)]
    body = {
        "run_id": run_id,
        "messages": messages,
        "temperature": 0.0,
        "agent_id": "agent-v7",
        "model": model,
    }
    last_err: Exception | None = None
    for r in range(3):
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("choices"):
                return (data["choices"][0].get("message", {}) or {}).get("content") or ""
            if isinstance(data, str):
                return data
            return json.dumps(data)
        except Exception as e:
            last_err = e
            time.sleep(1 + r)
    raise last_err if last_err else RuntimeError("LLM call failed")


def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False) -> str:
    """Entry point required by the evaluation harness.

    Returns a unified diff patch that fully replaces main.py.
    """
    run_id = (input_dict or {}).get("run_id", os.getenv("RUN_ID", str(uuid.uuid4())))

    if repo_dir and os.path.exists(repo_dir):
        try:
            os.chdir(repo_dir)
        except Exception:
            pass

    problem_statement = (input_dict or {}).get("problem_statement", "") or ""
    mode = (input_dict or {}).get("problem_category", None)
    if mode not in ("spec_only", "tests_available"):
        mode = "tests_available" if os.path.exists("tests.py") else "spec_only"

    # Compact repository summary for context (generic, no problem-specific assumptions)
    parts: List[str] = []
    for name in ("main.py", "tests.py"):
        content = _read(name)
        if content:
            parts.append(f"### {name}\n```python\n{content[:8000]}\n```")
    summary = "\n\n".join(parts)

    system_msg = (
        "You are a senior Python engineer.\n"
        + ("Do not modify tests.py; only change main.py.\n" if mode == "tests_available" else "")
        + "Return ONLY one code block containing the complete main.py with a '# main.py' header.\n"
        "Format exactly as:\n```python\n# main.py\n[complete code]\n```\n"
        "No prose. Deterministic code.\n"
        "Preserve all existing public function/class names and signatures from main.py; implement bodies only unless the spec explicitly requires a different API.\n"
        "Use only the Python standard library; avoid external dependencies and I/O.\n"
        "No top-level execution or side effects; define functions/classes only.\n"
    )
    user_msg = (
        f"Problem Statement (trimmed if long):\n{problem_statement[:12000]}\n\n"
        f"Repository Summary:\n{summary}\n\n"
        "Implement strictly according to the problem statement so that all tests (if present) pass."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Try a few models and accept the first syntactically valid main.py block
    for attempt in range(len(AGENT_MODELS)):
        try:
            resp = _call_llm(messages, run_id, attempt, 300)
            main_src = _extract_main_py(resp)
            if not main_src:
                continue
            ok, err = _validate_python(main_src)
            if ok:
                return _build_single_file_patch("main.py", main_src)
            # Single refinement attempt with syntax error feedback, then one more if needed
            refine_msgs = _build_refinement_messages(problem_statement, summary, main_src, err, mode)
            try:
                resp2 = _call_llm(refine_msgs, run_id, attempt, 300)
                main_src2 = _extract_main_py(resp2)
                if main_src2:
                    ok2, err2 = _validate_python(main_src2)
                    if ok2:
                        return _build_single_file_patch("main.py", main_src2)
                    # Final refinement
                    refine_msgs2 = _build_refinement_messages(problem_statement, summary, main_src2, err2, mode)
                    resp3 = _call_llm(refine_msgs2, run_id, attempt, 300)
                    main_src3 = _extract_main_py(resp3)
                    if main_src3 and _validate_python(main_src3)[0]:
                        return _build_single_file_patch("main.py", main_src3)
            except Exception:
                pass
        except Exception:
            continue

    return ""

