import os
import re
import json
import uuid
import time
import ast
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
    """Read a text file using UTF-8; return empty string on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _log(message: str) -> None:
    """Lightweight stdout logger for the v7 agent."""
    try:
        print(f"[V7] {message}")
    except Exception:
        # Best-effort logging only
        pass


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


def _extract_required_api_from_stub(stub_src: str) -> Dict[str, Any]:
    """Parse the existing main.py stub to discover required public API.

    Returns a dict with keys:
    - functions: List[Dict{name: str, params: List[str]}]
    - classes: List[Dict{name: str, init_params: List[str]}]
    """
    api: Dict[str, Any] = {"functions": [], "classes": []}
    if not stub_src:
        return api
    try:
        tree = ast.parse(stub_src)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                params = [a.arg for a in (node.args.args or [])]
                api["functions"].append({"name": node.name, "params": params})
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                init_params: List[str] = []
                for n in node.body:
                    if isinstance(n, ast.FunctionDef) and n.name == "__init__":
                        init_params = [a.arg for a in (n.args.args or [])]
                        break
                api["classes"].append({"name": node.name, "init_params": init_params})
    except Exception:
        # Best-effort; if parsing fails, we just return what we have
        return api
    return api


def _format_required_api_for_prompt(api: Dict[str, Any]) -> str:
    if not api:
        return ""
    lines: List[str] = []
    funcs = api.get("functions") or []
    clss = api.get("classes") or []
    if funcs:
        lines.append("Required top-level functions (preserve names and parameters):")
        for f in funcs:
            params = ", ".join(f.get("params") or [])
            lines.append(f"- def {f.get('name', 'func')}({params}): ...")
    if clss:
        lines.append("Required classes (preserve names; if __init__ exists, preserve its parameters):")
        for c in clss:
            ip = ", ".join(c.get("init_params") or [])
            if ip:
                lines.append(f"- class {c.get('name', 'Class')}:  def __init__({ip}): ...")
            else:
                lines.append(f"- class {c.get('name', 'Class')}: ...")
    return "\n".join(lines)


def _validate_required_api(code: str, api: Dict[str, Any]) -> Tuple[bool, str]:
    """Ensure generated code defines the required functions/classes and preserves parameters.

    We ignore leading '# main.py' header and parameter annotations/defaults; we check only names and order.
    For classes, only __init__ parameters (including 'self') are validated if present in the stub.
    """
    if not api:
        return True, ""
    try:
        src = re.sub(r"^#\s*main\.py\s*\n", "", code)
        tree = ast.parse(src)
    except Exception as e:
        return False, f"Cannot parse generated code for API validation: {e}"

    # Build lookup maps from generated code
    gen_funcs: Dict[str, List[str]] = {}
    gen_classes_inits: Dict[str, List[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            gen_funcs[node.name] = [a.arg for a in (node.args.args or [])]
        elif isinstance(node, ast.ClassDef):
            init_params: List[str] = []
            for n in node.body:
                if isinstance(n, ast.FunctionDef) and n.name == "__init__":
                    init_params = [a.arg for a in (n.args.args or [])]
                    break
            gen_classes_inits[node.name] = init_params

    # Validate required functions
    for f in (api.get("functions") or []):
        name = f.get("name")
        need_params = f.get("params") or []
        if name not in gen_funcs:
            return False, f"Missing required function: {name}"
        have_params = gen_funcs.get(name, [])
        if have_params != need_params:
            return False, f"Function '{name}' parameters mismatch. Expected ({', '.join(need_params)}), got ({', '.join(have_params)})."

    # Validate required classes
    for c in (api.get("classes") or []):
        cname = c.get("name")
        if cname not in gen_classes_inits and cname not in [n for n in gen_funcs.keys()]:
            # ensure class exists (ignore functions shadowing the name)
            return False, f"Missing required class: {cname}"
        need_ip = c.get("init_params") or []
        if need_ip:
            have_ip = gen_classes_inits.get(cname, [])
            if have_ip != need_ip:
                return False, f"Class '{cname}' __init__ parameters mismatch. Expected ({', '.join(need_ip)}), got ({', '.join(have_ip)})."

    return True, ""


def _repair_api(run_id: str, bad_code: str, error_text: str, required_api: Dict[str, Any], attempt_index: int) -> str:
    """Ask the LLM to adjust code to satisfy required API without changing behavior intent."""
    rules = (
        "You are a Python refactoring assistant.\n"
        "Task: Adjust ONLY the module's public API to match the required signatures below (names and parameters).\n"
        "Preserve semantics and logic as much as possible; do not add tests; do not add I/O.\n"
        "Return EXACTLY one code block formatted as:\n"
        "```python\n# main.py\n[complete file]\n```\n"
        "No prose. Deterministic output."
    )
    api_text = _format_required_api_for_prompt(required_api)
    user = (
        "The current file violates the required API. Fix only the signatures and definitions to match.\n\n"
        f"API validation error:\n{error_text}\n\n"
        f"Required API:\n{api_text}\n\n"
        "Current file:\n```python\n# main.py\n" + bad_code.strip() + "\n```\n"
    )
    messages = [{"role": "system", "content": rules}, {"role": "user", "content": user}]
    try:
        _log(f"Requesting API repair (attempt_index={attempt_index})")
        resp = _call_llm(messages, run_id, attempt_index, 180)
        fixed = _extract_main_py(resp)
        return fixed or ""
    except Exception:
        return ""


def _strip_code_fences(text: str) -> str:
    """Remove a single surrounding triple-backtick code fence from text, preserving content."""
    if not isinstance(text, str):
        return ""
    s = text.strip()
    # Remove a single surrounding ```...``` block if present
    if s.startswith("```") and s.endswith("```"):
        m = re.findall(r"```(?:python)?\s*\n([\s\S]*?)\n```", s, re.DOTALL)
        if m and m[0].strip():
            return m[0].strip()
    return s


def _ensure_header(code: str) -> str:
    """Ensure the file starts with the required '# main.py' header line."""
    code = code.lstrip("\n")
    if not code.startswith("# main.py"):
        return "# main.py\n" + code
    return code


def _extract_main_py(response: str) -> str:
    """Extract a Python code block representing complete main.py from an LLM response."""
    if not response:
        return ""
    # 1) Prefer explicitly headed python block
    m = re.findall(r"```python\s*\n#\s*main\.py\n([\s\S]*?)\n```", response, re.DOTALL)
    if m and m[0].strip():
        _log("Extracted main.py from headed python block")
        return _ensure_header(m[0].strip())
    # 2) Any python block
    m2 = re.findall(r"```python\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    if m2 and m2[0].strip():
        _log("Extracted main.py from unheaded python block")
        return _ensure_header(m2[0].strip())
    # 3) Any code fence without language
    m3 = re.findall(r"```\s*\n([\s\S]*?)\n```", response, re.DOTALL)
    if m3 and m3[0].strip():
        _log("Extracted main.py from generic code fence")
        return _ensure_header(m3[0].strip())
    # 4) Fallback to raw (strip if contains def/class)
    raw = response.strip()
    if ("def " in raw) or ("class " in raw):
        _log("Extracted main.py from raw content fallback")
        return _ensure_header(_strip_code_fences(raw))
    return ""


def _validate_python(code: str) -> Tuple[bool, str]:
    """Parse code with ast to check syntax. Returns (ok, error_text)."""
    try:
        # Strip header comment before parsing
        src = re.sub(r"^#\s*main\.py\s*\n", "", code)
        ast.parse(src)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}, col {e.offset}"
    except Exception as e:
        return False, f"Error: {e}"


def _repair_syntax(run_id: str, bad_code: str, error_text: str, attempt_index: int) -> str:
    """Ask the LLM to repair syntax only, returning a single code block for main.py."""
    system = (
        "You are a Python code fixer.\n"
        "Task: Repair ONLY syntax/parse errors in the provided Python file.\n"
        "Do not change behavioral intent or API. Do not add tests.\n"
        "Return EXACTLY one code block formatted as:\n"
        "```python\n# main.py\n[complete file]\n```\n"
        "No prose, no extra blocks. Deterministic output."
    )
    user = (
        "The following Python file fails to parse. Fix syntax only.\n\n"
        f"Parser error:\n{error_text}\n\n"
        "Current file:\n```python\n# main.py\n" + bad_code.strip() + "\n```\n"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        _log(f"Requesting syntax repair (attempt_index={attempt_index})")
        resp = _call_llm(messages, run_id, attempt_index, 180)
        fixed = _extract_main_py(resp)
        return fixed or ""
    except Exception:
        return ""


def _call_llm(messages: List[Dict[str, str]], run_id: str, attempt: int, timeout_s: int = 240) -> str:
    """Call the inference gateway with retries and return the content string."""
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
            _log(f"LLM POST {url} model={model} try={r+1}/3 run_id={run_id}")
            resp = requests.post(url, json=body, headers=headers, timeout=timeout_s)
            _log(f"LLM HTTP {resp.status_code} response_len={len(resp.text)}")
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("choices"):
                return (data["choices"][0].get("message", {}) or {}).get("content") or ""
            if isinstance(data, str):
                return data
            return json.dumps(data)
        except Exception as e:
            last_err = e
            # Linear backoff to be gentle on gateway
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
            _log(f"Changed working directory to: {os.getcwd()}")
        except Exception:
            pass

    problem_statement = (input_dict or {}).get("problem_statement", "") or ""
    mode = (input_dict or {}).get("problem_category", None)
    if mode not in ("spec_only", "tests_available"):
        mode = "tests_available" if os.path.exists("tests.py") else "spec_only"
    _log(f"run_id={run_id} mode={mode} problem_len={len(problem_statement)} tests_present={os.path.exists('tests.py')}")

    # Compact repository summary for context (generic, no problem-specific assumptions)
    parts: List[str] = []
    main_stub = _read("main.py")
    for name in ("main.py", "tests.py"):
        content = _read(name)
        if content:
            parts.append(f"### {name}\n```python\n{content[:8000]}\n```")
    summary = "\n\n".join(parts)
    _log(f"Repository summary included files: {', '.join([p.split()[1] for p in parts]) if parts else 'none'}")

    # Extract required API from the current stub
    required_api = _extract_required_api_from_stub(main_stub)
    api_prompt = _format_required_api_for_prompt(required_api)

    system_msg = (
        "You are a senior Python engineer.\n"
        + ("Do not modify tests.py; only change main.py.\n" if mode == "tests_available" else "")
        + "Strict output format: return EXACTLY one code block with the full file.\n"
        "Use this format only:\n```python\n# main.py\n[complete Python file]\n```\n"
        "Requirements:\n"
        "- Deterministic, no randomness, no I/O, no prints/logging.\n"
        "- Only Python standard library.\n"
        "- Precise input validation and clear exceptions consistent with the spec.\n"
        "- Type hints and concise docstrings for public functions/classes.\n"
        "- Keep code readable and minimal; avoid unnecessary abstractions.\n"
        + ("\nRequired public API (must be implemented exactly as listed):\n" + api_prompt + "\n" if api_prompt else "")
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

    # Rotate starting model index to diversify across runs
    start_idx = (hash(run_id) % len(AGENT_MODELS)) if AGENT_MODELS else 0
    indices = [(start_idx + i) % len(AGENT_MODELS) for i in range(len(AGENT_MODELS))]

    # Try models; validate syntax; attempt auto-repair up to 2 times per model
    for attempt_pos, model_idx in enumerate(indices):
        try:
            _log(f"Generating main.py with model_index={model_idx} (pos {attempt_pos+1}/{len(indices)})")
            resp = _call_llm(messages, run_id, model_idx, 300)
            main_src = _extract_main_py(resp)
            if not main_src:
                _log("No valid code block extracted; trying next model")
                continue
            ok, err = _validate_python(main_src)
            if ok:
                api_ok, api_err = _validate_required_api(main_src, required_api)
                if api_ok:
                    _log("Generated code parsed successfully and satisfies required API")
                    return _build_single_file_patch("main.py", main_src)
                # Attempt one API repair on syntactically valid code
                _log("Generated code violates required API; attempting API repair")
                fixed_api = _repair_api(f"{run_id}-apifix-{attempt_pos}", main_src, api_err, required_api, model_idx)
                if fixed_api:
                    okf, errf = _validate_python(fixed_api)
                    if okf:
                        api_ok2, _ = _validate_required_api(fixed_api, required_api)
                        if api_ok2:
                            _log("API repair succeeded; returning patch")
                            return _build_single_file_patch("main.py", fixed_api)
            # Try up to 2 syntax repairs
            repaired = main_src
            for fix_round in range(2):
                _log(f"Syntax repair round {fix_round+1}/2 for current model")
                fixed = _repair_syntax(f"{run_id}-fix{attempt_pos}-{fix_round+1}", repaired, err, model_idx)
                if not fixed:
                    _log("Repair attempt returned empty response; stopping repair for this model")
                    break
                ok2, err2 = _validate_python(fixed)
                if ok2:
                    api_ok2, api_err2 = _validate_required_api(fixed, required_api)
                    if api_ok2:
                        _log("Repair produced syntactically valid code that satisfies required API")
                        return _build_single_file_patch("main.py", fixed)
                    _log("Repaired code violates required API; attempting API repair")
                    fixed2 = _repair_api(f"{run_id}-apifix{attempt_pos}-{fix_round+1}", fixed, api_err2, required_api, model_idx)
                    if fixed2:
                        ok3, err3 = _validate_python(fixed2)
                        if ok3:
                            api_ok3, _ = _validate_required_api(fixed2, required_api)
                            if api_ok3:
                                _log("API repair after syntax repair succeeded; returning patch")
                                return _build_single_file_patch("main.py", fixed2)
                repaired, err = fixed, err2
        except Exception:
            _log("Model attempt raised exception; continuing with next model")
            continue

    return ""

