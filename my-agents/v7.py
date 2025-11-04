import os
import re
import json
import uuid
import time
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
    """Extract a complete main.py code block from an LLM response.

    Priority:
    1) ```python\n# main.py\n...\n```
    2) First ```python``` block
    3) Any ```\n...\n``` block (language-agnostic)
    """
    if not response:
        return ""
    # Strict headered python block
    m = re.findall(r"```python\s*\n#\s*main\.py\n([\s\S]*?)\n```", response, re.DOTALL)
    if m and m[0].strip():
        return m[0].strip()
    # Any python block
    m2 = re.findall(r"```python\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    if m2 and m2[0].strip():
        return m2[0].strip()
    # Any fenced block
    m3 = re.findall(r"```\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    if m3 and m3[0].strip():
        return m3[0].strip()
    return ""


def _parse_api_signatures(src: str) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Return (functions[name,argcount], classes[name]) from module source without underscored names."""
    funcs: List[Tuple[str, int]] = []
    classes: List[str] = []
    try:
        import ast
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                arg_count = len([a for a in node.args.args])
                funcs.append((node.name, arg_count))
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                classes.append(node.name)
    except Exception:
        return [], []
    return funcs, classes


def _syntax_valid(src: str) -> Tuple[bool, str]:
    """Check Python syntax by parsing AST. Return (ok, error_message)."""
    if not isinstance(src, str) or not src.strip():
        return False, "empty source"
    try:
        import ast
        ast.parse(src)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {getattr(e, 'lineno', '?')} col {getattr(e, 'offset', '?')}"
    except Exception as e:
        return False, f"ParseError: {e}"


def _extract_exception_contracts(text: str) -> List[Tuple[str, str]]:
    """Find ExceptionType("exact message") pairs in the plain-text spec.

    Returns a de-duplicated list preserving order.
    """
    pat = r"raise\s+([A-Za-z_][A-Za-z0-9_]*Error)\(\s*['\"]([^'\"]+)['\"]\s*\)"
    found = [(m.group(1), m.group(2)) for m in re.finditer(pat, text or "")]
    seen = set()
    out: List[Tuple[str, str]] = []
    for it in found:
        if it not in seen:
            out.append(it)
            seen.add(it)
    return out


def _scan_policy_violations(src: str) -> List[str]:
    """Detect generic anti-patterns that often break evaluations.

    Used to request a corrected candidate from the model.
    """
    if not isinstance(src, str):
        return ["non-string candidate"]
    issues: List[str] = []
    # Discourage I/O and environment interactions
    banned_patterns = [
        r"\bprint\s*\(",
        r"\binput\s*\(",
        r"\bopen\s*\(",
        r"\bos\.system\s*\(",
        r"\bsubprocess\.",
        r"\brequests\.",
        r"\btime\.sleep\s*\(",
    ]
    for pat in banned_patterns:
        if re.search(pat, src):
            issues.append(f"disallowed usage: {pat}")
    # Randomness without control can cause flaky behavior
    if re.search(r"\brandom\.", src):
        issues.append("avoid random-based nondeterminism; implement deterministic logic instead")
    # Over-broad imports
    if re.search(r"^\s*from\s+\S+\s+import\s+\*", src, flags=re.MULTILINE):
        issues.append("avoid star-imports; import explicit names")
    return issues


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
            # Trim to keep token usage reasonable
            parts.append(f"### {name}\n```python\n{content[:6000]}\n```")
    summary = "\n\n".join(parts)

    # Capture skeleton API from existing main.py to enforce preservation in candidates
    skeleton_main = _read("main.py")
    sk_funcs, sk_classes = _parse_api_signatures(skeleton_main)

    contracts = _extract_exception_contracts(problem_statement)

    # Compose strict rules section (generic, not problem-specific)
    strict_rules = [
        "Preserve all public function and class names from the skeleton and fulfill their contracts.",
        "Implement exact exception types and messages if the spec states them (case and punctuation must match).",
        "Validate inputs rigorously (types, arity/shape, bounds) and raise precise exceptions per spec.",
        "Ensure deterministic behavior: avoid I/O, sleeps, global mutable state, or randomness.",
        "Prefer pure functions and canonicalized outputs; avoid relying on dict/set iteration order.",
        "Do not include any tests or prose in the answer.",
    ]
    rules_text = "\n- " + "\n- ".join(strict_rules)

    contracts_text = "\n".join([f"- {exc}: \"{msg}\"" for exc, msg in contracts]) if contracts else ""

    system_msg = (
        "You are a senior Python engineer.\n"
        + ("Do not modify tests.py; only change main.py.\n" if mode == "tests_available" else "")
        + "Follow the problem statement and the existing main.py skeleton strictly.\n"
        + "Strict rules:" + rules_text + "\n"
        + ("Exception contracts to enforce exactly:\n" + contracts_text + "\n" if contracts_text else "")
        + "Return ONLY one code block containing the complete main.py with a '# main.py' header.\n"
        "Format exactly as:\n```python\n# main.py\n[complete code]\n```\n"
    )
    user_msg = (
        f"Problem Statement (trimmed if long):\n{problem_statement[:12000]}\n\n"
        f"Repository Summary:\n{summary}\n\n"
        "Implement strictly so all tests (if present) pass."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Multi-model, multi-round with parsing and API-preservation checks
    max_rounds = 4
    for round_idx in range(max_rounds):
        for attempt in range(len(AGENT_MODELS)):
            try:
                resp = _call_llm(messages, run_id, attempt + round_idx, 300)
                candidate = _extract_main_py(resp)
                if not candidate:
                    continue
                ok, err = _syntax_valid(candidate)
                if not ok:
                    # Add repair hint and continue next attempt
                    repair_note = (
                        "Your last answer had Python syntax errors and was rejected.\n"
                        f"Parser error: {err}.\n"
                        "Return only a corrected complete main.py code block as specified."
                    )
                    messages.append({"role": "assistant", "content": f"```python\n# main.py\n{candidate}\n```"})
                    messages.append({"role": "user", "content": repair_note})
                    continue
                # Enforce preservation of skeleton public API names (best-effort, generic)
                if sk_funcs or sk_classes:
                    cand_funcs, cand_classes = _parse_api_signatures(candidate)
                    cand_func_names = {n for (n, _a) in cand_funcs}
                    missing_funcs = [n for (n, _a) in sk_funcs if n not in cand_func_names]
                    missing_classes = [c for c in sk_classes if c not in set(cand_classes)]
                    if missing_funcs or missing_classes:
                        miss_text = "".join(
                            [f"- function: {n}\n" for n in missing_funcs]
                            + [f"- class: {n}\n" for n in missing_classes]
                        )
                        repair_note = (
                            "Preserve all public API from the skeleton main.py. The following names are missing in your answer:\n"
                            f"{miss_text}"
                            "Return a corrected complete main.py with these names implemented."
                        )
                        messages.append({"role": "assistant", "content": f"```python\n# main.py\n{candidate}\n```"})
                        messages.append({"role": "user", "content": repair_note})
                        continue
                # Policy violations (I/O, randomness, etc.)
                violations = _scan_policy_violations(candidate)
                if violations:
                    messages.append({"role": "assistant", "content": f"```python\n# main.py\n{candidate}\n```"})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your last answer violated one or more strict rules and was rejected.\n"
                            + "\n".join(f"- {v}" for v in violations)
                            + "\nPlease return a corrected, deterministic implementation that adheres to all strict rules and preserves the skeleton API."
                        ),
                    })
                    continue
                # Candidate passes syntax and basic API checks
                return _build_single_file_patch("main.py", candidate)
            except Exception:
                # Try next attempt/model
                continue

    return ""

