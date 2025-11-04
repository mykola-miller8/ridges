import os
import re
import json
import uuid
import time
import subprocess
from typing import Any, Dict, List

import requests


# v7 seed agent: generic code agent that proposes a full-file patch for main.py only.
# - Never embeds problem-specific constants or dataset names
# - Uses only the inference gateway exposed via INFERENCE_URL/SANDBOX_PROXY_URL
# - Returns a unified diff that replaces main.py entirely


def _verbose_log(component: str, message: str, data: Any = None) -> None:
    """Verbose logging function that prints full details for debugging.
    
    Args:
        component: Component name (e.g., 'READ', 'LLM', 'PARSE', 'VALIDATE')
        message: Log message
        data: Optional data to include in log (will be JSON-serialized if dict/list)
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [V7-{component}] {message}"
    if data is not None:
        if isinstance(data, (dict, list)):
            try:
                data_str = json.dumps(data, indent=2, ensure_ascii=False)
                log_line += f"\n{data_str}"
            except Exception:
                log_line += f" | data={repr(data)}"
        else:
            log_line += f" | data={repr(data)}"
    print(log_line, flush=True)


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
    """Read file content with verbose logging."""
    _verbose_log("READ", f"Attempting to read file: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            _verbose_log("READ", f"Successfully read {path}", {
                "path": path,
                "size_bytes": len(content),
                "size_lines": len(content.splitlines())
            })
            return content
    except Exception as e:
        _verbose_log("READ", f"Failed to read {path}", {"error": str(e), "error_type": type(e).__name__})
        return ""


def _build_single_file_patch(filename: str, new_content: str) -> str:
    """Build a full-file unified diff for filename replacing its contents with new_content."""
    _verbose_log("PATCH", f"Building patch for {filename}")
    old = _read(filename)
    old_lines = old.splitlines()
    new_lines = new_content.splitlines()
    _verbose_log("PATCH", f"Patch stats for {filename}", {
        "old_lines": len(old_lines),
        "new_lines": len(new_lines),
        "old_size_bytes": len(old),
        "new_size_bytes": len(new_content)
    })
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
    patch = "\n".join(header + body) + "\n"
    _verbose_log("PATCH", f"Patch built for {filename}", {"patch_size_bytes": len(patch)})
    return patch


def _extract_main_py(response: str) -> str:
    """Extract a complete main.py code block from an LLM response.

    Priority:
    1) ```python\n# main.py\n...\n```
    2) First ```python``` block
    3) Any ```\n...\n``` block (language-agnostic)
    """
    _verbose_log("EXTRACT", "Extracting main.py from LLM response", {
        "response_length": len(response) if response else 0
    })
    if not response:
        _verbose_log("EXTRACT", "Empty response, returning empty string")
        return ""
    # Prefer explicitly headed block
    m = re.findall(r"```python\s*\n#\s*main\.py\n([\s\S]*?)\n```", response, re.DOTALL)
    if m and m[0].strip():
        extracted = m[0].strip()
        _verbose_log("EXTRACT", "Extracted using strict headered python block", {
            "extracted_length": len(extracted),
            "method": "strict_headered_python"
        })
        return extracted
    # Any python block
    m2 = re.findall(r"```python\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    if m2 and m2[0].strip():
        extracted = m2[0].strip()
        _verbose_log("EXTRACT", "Extracted using python block", {
            "extracted_length": len(extracted),
            "method": "python_block"
        })
        return extracted
    # Any fenced block
    m3 = re.findall(r"```\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    if m3 and m3[0].strip():
        extracted = m3[0].strip()
        _verbose_log("EXTRACT", "Extracted using generic fenced block", {
            "extracted_length": len(extracted),
            "method": "generic_fenced"
        })
        return extracted
    _verbose_log("EXTRACT", "No code block found in response", {
        "response_preview": response[:500] if response else ""
    })
    return ""


def _parse_api_signatures(src: str) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Return (functions[name,argcount], classes[name]) from module source without underscored names."""
    _verbose_log("PARSE", "Parsing API signatures from source", {"source_length": len(src) if src else 0})
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
        _verbose_log("PARSE", "Successfully parsed API signatures", {
            "functions": funcs,
            "classes": classes,
            "total_functions": len(funcs),
            "total_classes": len(classes)
        })
    except Exception as e:
        _verbose_log("PARSE", "Failed to parse API signatures", {
            "error": str(e),
            "error_type": type(e).__name__
        })
        return [], []
    return funcs, classes


def _syntax_valid(src: str) -> Tuple[bool, str]:
    """Check Python syntax by parsing AST. Return (ok, error_message)."""
    _verbose_log("VALIDATE", "Validating syntax", {"source_length": len(src) if src else 0})
    if not isinstance(src, str) or not src.strip():
        _verbose_log("VALIDATE", "Syntax validation failed: empty source")
        return False, "empty source"
    try:
        import ast
        ast.parse(src)
        _verbose_log("VALIDATE", "Syntax validation passed")
        return True, ""
    except SyntaxError as e:
        error_msg = f"SyntaxError: {e.msg} at line {getattr(e, 'lineno', '?')} col {getattr(e, 'offset', '?')}"
        _verbose_log("VALIDATE", "Syntax validation failed: SyntaxError", {
            "error": error_msg,
            "line": getattr(e, 'lineno', '?'),
            "offset": getattr(e, 'offset', '?')
        })
        return False, error_msg
    except Exception as e:
        error_msg = f"ParseError: {e}"
        _verbose_log("VALIDATE", "Syntax validation failed: ParseError", {
            "error": error_msg,
            "error_type": type(e).__name__
        })
        return False, error_msg


def _extract_exception_contracts(text: str) -> List[Tuple[str, str]]:
    """Find ExceptionType("exact message") pairs in the plain-text spec.

    Returns a de-duplicated list preserving order.
    """
    _verbose_log("CONTRACT", "Extracting exception contracts", {"text_length": len(text) if text else 0})
    pat = r"raise\s+([A-Za-z_][A-Za-z0-9_]*Error)\(\s*['\"]([^'\"]+)['\"]\s*\)"
    found = [(m.group(1), m.group(2)) for m in re.finditer(pat, text or "")]
    seen = set()
    out: List[Tuple[str, str]] = []
    for it in found:
        if it not in seen:
            out.append(it)
            seen.add(it)
    _verbose_log("CONTRACT", "Extracted exception contracts", {
        "contracts": out,
        "total_contracts": len(out),
        "raw_matches": len(found)
    })
    return out


def _scan_policy_violations(src: str) -> List[str]:
    """Detect generic anti-patterns that often break evaluations.

    Used to request a corrected candidate from the model.
    """
    _verbose_log("POLICY", "Scanning for policy violations", {"source_length": len(src) if src else 0})
    if not isinstance(src, str):
        _verbose_log("POLICY", "Policy violation: non-string candidate")
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
    if issues:
        _verbose_log("POLICY", "Policy violations found", {"violations": issues, "count": len(issues)})
    else:
        _verbose_log("POLICY", "No policy violations found")
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
    _verbose_log("LLM", f"Calling LLM (attempt {attempt})", {
        "url": url,
        "model": model,
        "run_id": run_id,
        "timeout_s": timeout_s,
        "num_messages": len(messages),
        "message_roles": [msg.get("role", "unknown") for msg in messages]
    })
    last_err: Exception | None = None
    for r in range(3):
        try:
            _verbose_log("LLM", f"LLM request attempt {r + 1}/3", {"retry": r})
            resp = requests.post(url, json=body, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            _verbose_log("LLM", f"LLM response received (attempt {r + 1})", {
                "status_code": resp.status_code,
                "response_type": type(data).__name__,
                "has_choices": isinstance(data, dict) and "choices" in data
            })
            if isinstance(data, dict) and data.get("choices"):
                content = (data["choices"][0].get("message", {}) or {}).get("content") or ""
                _verbose_log("LLM", "LLM call successful", {
                    "response_length": len(content),
                    "response_preview": content[:200] if content else ""
                })
                return content
            if isinstance(data, str):
                _verbose_log("LLM", "LLM call successful (string response)", {
                    "response_length": len(data),
                    "response_preview": data[:200] if data else ""
                })
                return data
            _verbose_log("LLM", "LLM call successful (JSON response)", {
                "response": json.dumps(data)[:500]
            })
            return json.dumps(data)
        except Exception as e:
            last_err = e
            _verbose_log("LLM", f"LLM request failed (attempt {r + 1}/3)", {
                "error": str(e),
                "error_type": type(e).__name__,
                "will_retry": r < 2,
                "sleep_seconds": 1 + r
            })
            time.sleep(1 + r)
    _verbose_log("LLM", "LLM call failed after all retries", {
        "final_error": str(last_err) if last_err else "unknown",
        "error_type": type(last_err).__name__ if last_err else "unknown"
    })
    raise last_err if last_err else RuntimeError("LLM call failed")


def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False) -> str:
    """Entry point required by the evaluation harness.

    Returns a unified diff patch that fully replaces main.py.
    """
    run_id = (input_dict or {}).get("run_id", os.getenv("RUN_ID", str(uuid.uuid4())))
    _verbose_log("AGENT", "agent_main called", {
        "run_id": run_id,
        "repo_dir": repo_dir,
        "test_mode": test_mode,
        "input_dict_keys": list(input_dict.keys()) if input_dict else []
    })

    if repo_dir and os.path.exists(repo_dir):
        try:
            os.chdir(repo_dir)
            _verbose_log("AGENT", f"Changed directory to {repo_dir}")
        except Exception as e:
            _verbose_log("AGENT", f"Failed to change directory to {repo_dir}", {"error": str(e)})

    problem_statement = (input_dict or {}).get("problem_statement", "") or ""
    mode = (input_dict or {}).get("problem_category", None)
    if mode not in ("spec_only", "tests_available"):
        mode = "tests_available" if os.path.exists("tests.py") else "spec_only"
    _verbose_log("AGENT", "Mode detected", {
        "mode": mode,
        "tests_py_exists": os.path.exists("tests.py"),
        "problem_statement_length": len(problem_statement)
    })

    # Compact repository summary for context (generic, no problem-specific assumptions)
    parts: List[str] = []
    for name in ("main.py", "tests.py"):
        content = _read(name)
        if content:
            original_length = len(content)
            trimmed_length = 6000
            if original_length > trimmed_length:
                _verbose_log("AGENT", f"Content trimmed for {name}", {
                    "original_length": original_length,
                    "trimmed_length": trimmed_length,
                    "trimmed_bytes": original_length - trimmed_length
                })
            # Trim to keep token usage reasonable
            parts.append(f"### {name}\n```python\n{content[:6000]}\n```")
    summary = "\n\n".join(parts)
    _verbose_log("AGENT", "Repository summary built", {
        "summary_length": len(summary),
        "num_parts": len(parts)
    })

    # Capture skeleton API from existing main.py to enforce preservation in candidates
    skeleton_main = _read("main.py")
    sk_funcs, sk_classes = _parse_api_signatures(skeleton_main)
    _verbose_log("AGENT", "Skeleton API parsed", {
        "skeleton_functions": sk_funcs,
        "skeleton_classes": sk_classes,
        "skeleton_main_length": len(skeleton_main)
    })

    contracts = _extract_exception_contracts(problem_statement)
    _verbose_log("AGENT", "Exception contracts extracted", {
        "num_contracts": len(contracts),
        "contracts": contracts
    })

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

    original_problem_length = len(problem_statement)
    trimmed_problem_length = 12000
    if original_problem_length > trimmed_problem_length:
        _verbose_log("AGENT", "Problem statement will be trimmed", {
            "original_length": original_problem_length,
            "trimmed_length": trimmed_problem_length,
            "trimmed_bytes": original_problem_length - trimmed_problem_length
        })

    system_msg = (
        "You are a senior Python engineer.\n"
        + ("Do not modify tests.py; only change main.py.\n" if mode == "tests_available" else "")
        + "Return ONLY one code block containing the complete main.py with a '# main.py' header.\n"
        "Format exactly as:\n```python\n# main.py\n[complete code]\n```\n"
        "No prose. Deterministic code. Be careful not to be stuck in an infinite loop."
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
    _verbose_log("AGENT", "Messages prepared for LLM", {
        "system_msg_length": len(system_msg),
        "user_msg_length": len(user_msg),
        "total_message_length": len(system_msg) + len(user_msg)
    })

    # Multi-model, multi-round with parsing and API-preservation checks
    max_rounds = 4
    _verbose_log("AGENT", "Starting multi-round generation", {
        "max_rounds": max_rounds,
        "models": AGENT_MODELS,
        "total_attempts": max_rounds * len(AGENT_MODELS)
    })
    for round_idx in range(max_rounds):
        _verbose_log("AGENT", f"Starting round {round_idx + 1}/{max_rounds}")
        for attempt in range(len(AGENT_MODELS)):
            _verbose_log("AGENT", f"Round {round_idx + 1}, attempt {attempt + 1}/{len(AGENT_MODELS)}", {
                "round": round_idx,
                "attempt": attempt,
                "model_index": (attempt + round_idx) % len(AGENT_MODELS),
                "model": AGENT_MODELS[(attempt + round_idx) % len(AGENT_MODELS)]
            })
            try:
                resp = _call_llm(messages, run_id, attempt + round_idx, 300)
                candidate = _extract_main_py(resp)
                if not candidate:
                    _verbose_log("AGENT", "No candidate extracted from response, continuing to next attempt")
                    continue
                _verbose_log("AGENT", "Candidate extracted, validating", {
                    "candidate_length": len(candidate),
                    "candidate_preview": candidate[:200]
                })
                ok, err = _syntax_valid(candidate)
                if not ok:
                    # Add repair hint and continue next attempt
                    _verbose_log("AGENT", "Syntax validation failed, adding repair hint", {
                        "error": err,
                        "current_messages_count": len(messages)
                    })
                    repair_note = (
                        "Your last answer had Python syntax errors and was rejected.\n"
                        f"Parser error: {err}.\n"
                        "Return only a corrected complete main.py code block as specified."
                    )
                    messages.append({"role": "assistant", "content": f"```python\n# main.py\n{candidate}\n```"})
                    messages.append({"role": "user", "content": repair_note})
                    continue
                _verbose_log("AGENT", "Syntax validation passed, checking API preservation")
                # Enforce preservation of skeleton public API names (best-effort, generic)
                if sk_funcs or sk_classes:
                    cand_funcs, cand_classes = _parse_api_signatures(candidate)
                    cand_func_names = {n for (n, _a) in cand_funcs}
                    missing_funcs = [n for (n, _a) in sk_funcs if n not in cand_func_names]
                    missing_classes = [c for c in sk_classes if c not in set(cand_classes)]
                    if missing_funcs or missing_classes:
                        _verbose_log("AGENT", "API preservation check failed", {
                            "missing_functions": missing_funcs,
                            "missing_classes": missing_classes,
                            "candidate_functions": cand_funcs,
                            "candidate_classes": cand_classes
                        })
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
                    else:
                        _verbose_log("AGENT", "API preservation check passed")
                else:
                    _verbose_log("AGENT", "No skeleton API to preserve (empty skeleton)")
                # Policy violations (I/O, randomness, etc.)
                violations = _scan_policy_violations(candidate)
                if violations:
                    _verbose_log("AGENT", "Policy violations found, adding repair hint", {
                        "violations": violations,
                        "current_messages_count": len(messages)
                    })
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
                _verbose_log("AGENT", "Candidate passed all checks, building patch", {
                    "round": round_idx + 1,
                    "attempt": attempt + 1,
                    "candidate_length": len(candidate)
                })
                patch = _build_single_file_patch("main.py", candidate)
                _verbose_log("AGENT", "Patch built successfully, returning", {
                    "patch_length": len(patch),
                    "total_rounds": round_idx + 1,
                    "total_attempts": (round_idx * len(AGENT_MODELS)) + attempt + 1
                })
                return patch
            except Exception as e:
                _verbose_log("AGENT", f"Exception during round {round_idx + 1}, attempt {attempt + 1}", {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "will_continue": True
                })
                # Try next attempt/model
                continue

    _verbose_log("AGENT", "All rounds exhausted, returning empty patch", {
        "total_rounds_attempted": max_rounds,
        "total_attempts": max_rounds * len(AGENT_MODELS)
    })
    return ""

