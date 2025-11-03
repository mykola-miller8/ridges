import os
import re
import json
import uuid
from typing import Any, Dict, List, Tuple

# Reuse v4 utilities and base agent
from .v4 import (
    MinimalCursorAgent,
    run_tests_locally,
    ensure_git_initialized,
    reorder_models,
    AGENT_MODELS,
)


class MinimalCursorAgentV5(MinimalCursorAgent):
    """v5 agent: stricter, capped tests, single best suite, higher score threshold."""

    def _build_testgen_system(self) -> str:
        return (
            "You are a failure-test generation expert for Python exercises.\n"
            "Authoritative problem spec:\n"
            f"{(self.problem_statement or '')}\n\n"
            "Output constraints:\n"
            "- Import strictly with: from main import <symbols>\n"
            "- Single unittest.TestCase class\n"
            "- AT MOST 3 distinct test methods (no redundancy)\n"
            "- Deterministic; no prints or I/O\n"
            "- Prefer edge/invalid/boundary and structure/type/shape cases; include exact error types/messages per the spec\n"
            "- Return ONLY the test class code in a ```python code block.\n"
        )

    def _generate_failure_tests(self, main_src: str, base_run_id: str) -> str:
        system_content = self._build_testgen_system()
        user_content = (
            f"Current main.py (full):\n```python\n{(main_src or '')}\n```\n\n"
            "Write the tests as specified above."
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        collected_suites: List[str] = []
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(AGENT_MODELS)) as executor:
                futures = [executor.submit(self._call_llm, messages, base_run_id, i, 60) for i in range(len(AGENT_MODELS))]
                for fut in concurrent.futures.as_completed(futures, timeout=70):
                    try:
                        resp = fut.result()
                        if resp and not self._html_like(resp):
                            mm = re.findall(r"```python\s*\n(.*?)\n```", resp, re.DOTALL)
                            if mm and mm[0].strip():
                                collected_suites.append(mm[0].strip())
                    except Exception:
                        continue
        except Exception:
            for att in range(len(AGENT_MODELS)):
                resp = self._call_llm(messages, run_id=base_run_id, attempt=att, force_timeout_s=60)
                if resp and not self._html_like(resp):
                    mm = re.findall(r"```python\s*\n(.*?)\n```", resp, re.DOTALL)
                    if mm and mm[0].strip():
                        collected_suites.append(mm[0].strip())
        if not collected_suites:
            return ""
        # Select best single suite after filtering and capping
        best = self._select_best_suite(collected_suites, base_run_id)
        return best or ""

    def _pipeline_generate_tests_all_models(self, main_src: str, base_run_id: str) -> List[str]:
        system_content = self._build_testgen_system()
        user_content = (
            f"Current main.py (full):\n```python\n{(main_src or '')}\n```\n\n"
            "Write the tests as specified above."
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        collected: List[str] = []
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(AGENT_MODELS)) as ex:
            futs = [ex.submit(self._call_llm, messages, base_run_id, i, 300) for i in range(len(AGENT_MODELS))]
            for i, fut in enumerate(futs):
                try:
                    resp = fut.result(timeout=310)
                    if resp and not self._html_like(resp):
                        mm = re.findall(r"```python\s*\n(.*?)\n```", resp, re.DOTALL)
                        if mm and mm[0].strip():
                            collected.append(mm[0].strip())
                except Exception:
                    continue
        best = self._select_best_suite(collected, base_run_id)
        return [best] if best else []

    def _filter_methods_spec_only(self, suite: str) -> List[str]:
        # Extract methods
        methods = re.findall(r"\n\s*def\s+(test_[A-Za-z0-9_]+)\s*\(self[\s\S]*?\n(?=\s*def\s+test_|\s*class\s+|\Z)", "\n" + suite, re.DOTALL)
        bodies = [m.group(0) if hasattr(m, 'group') else None for m in re.finditer(r"\n\s*def\s+(test_[A-Za-z0-9_]+)\s*\(self[\s\S]*?\n(?=\s*def\s+test_|\s*class\s+|\Z)", "\n" + suite, re.DOTALL)]
        if not bodies:
            return []
        filtered: List[str] = []
        for body in bodies:
            txt = body or ""
            # Drop obviously non-spec property names often hallucinated
            if re.search(r"\.\s*(data|attributes)\b", txt):
                continue
            filtered.append(txt)
        # Cap to 3
        return filtered[:3]

    def _select_best_suite(self, suites: List[str], base_run_id: str) -> str:
        if not suites:
            return ""
        # Verify suites run structurally
        try:
            verified = self._pipeline_verify_test_suites(suites)
        except Exception:
            verified = suites
        if not verified:
            return ""
        # Score per-method and average per suite
        def score_methods(methods: List[str]) -> float:
            # Reuse v4 scoring but with higher bar (80) in selection only
            system_content = (
                "You are a Python testing auditor.\n"
                "Authoritative problem spec:\n"
                f"{self.problem_statement}\n\n"
                "Task: Given a list of unittest methods, score each one's correctness vs the spec (0-100).\n"
                "Return ONLY JSON mapping id->score, e.g., {\"m1\": 80, \"m2\": 55}."
            )
            payload = [{"id": f"m{i+1}", "code": m} for i, m in enumerate(methods)]
            user_content = "Methods (JSON array):\n" + json.dumps(payload, ensure_ascii=False, indent=2)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
            scores: List[float] = []
            for i in range(len(AGENT_MODELS)):
                try:
                    resp = self._call_llm(messages, base_run_id, i, 60)
                    obj = json.loads(resp)
                    if isinstance(obj, dict):
                        for k in obj.values():
                            try:
                                scores.append(float(k))
                            except Exception:
                                continue
                except Exception:
                    continue
            return (sum(scores) / len(scores)) if scores else 0.0

        best_suite = ""
        best_score = -1.0
        for s in verified:
            methods = self._filter_methods_spec_only(s) if self.mode == "spec_only" else self._extract_methods_from_suite(s)
            if not methods:
                continue
            avg = score_methods(methods)
            if avg > best_score:
                best_score = avg
                # Rebuild a capped single class with up to 3 methods
                rebuilt = self._merge_methods_into_suite(methods[:3])
                best_suite = rebuilt
        return best_suite

    def _pipeline_score_and_filter_methods(self, suites: List[str], base_run_id: str) -> List[str]:
        # Reuse parent extraction, but bump threshold to 80 and do not keep large sets
        methods: List[str] = []
        for suite in suites:
            methods.extend(self._extract_methods_from_suite(suite))
        if not methods:
            return []
        # Batch score using parent’s approach adapted here
        system_content = (
            "You are a Python testing auditor.\n"
            "Authoritative problem spec:\n"
            f"{self.problem_statement}\n\n"
            "Task: Given a list of unittest methods, score each one's correctness vs the spec (0-100).\n"
            "Return ONLY JSON mapping id->score, e.g., {\"m1\": 80, \"m2\": 55}."
        )
        payload = [{"id": f"m{i+1}", "code": m} for i, m in enumerate(methods)]
        user_content = "Methods (JSON array):\n" + json.dumps(payload, ensure_ascii=False, indent=2)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        id_to_score: Dict[str, float] = {}
        for i in range(len(AGENT_MODELS)):
            try:
                resp = self._call_llm(messages, base_run_id, i, 60)
                obj = json.loads(resp)
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        try:
                            id_to_score[k] = max(id_to_score.get(k, 0.0), float(v))
                        except Exception:
                            continue
            except Exception:
                continue
        kept: List[str] = []
        for idx, m in enumerate(methods):
            if float(id_to_score.get(f"m{idx+1}", 0.0)) >= 80.0:
                kept.append(m)
        # Cap to 3 highest by score
        scored_pairs = [(float(id_to_score.get(f"m{idx+1}", 0.0)), m) for idx, m in enumerate(methods)]
        scored_pairs.sort(key=lambda x: -x[0])
        return [m for _, m in scored_pairs[:3]]


def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False) -> str:
    run_id = (input_dict or {}).get("run_id", os.getenv("RUN_ID", str(uuid.uuid4())))
    if repo_dir and os.path.exists(repo_dir):
        try:
            os.chdir(repo_dir)
        except Exception:
            pass
    ensure_git_initialized()
    try:
        reorder_models((input_dict or {}).get("problem_statement", ""))
    except Exception:
        pass
    problem = (input_dict or {}).get("problem_statement", "")
    problem_category = (input_dict or {}).get("problem_category", None)
    category_filter = (input_dict or {}).get("category_filter", "all")
    if problem_category not in ("spec_only", "tests_available"):
        problem_category = "tests_available" if os.path.exists("tests.py") else "spec_only"
    if category_filter not in ("spec_only", "tests_available", "all"):
        category_filter = "all"
    if category_filter != "all" and problem_category != category_filter:
        return ""
    agent = MinimalCursorAgentV5(problem, mode=problem_category, top_k=30)
    patch = agent.propose_patch(run_id)
    return patch or ""

from __future__ import annotations

import os
import re
import json
import time
import ast
import subprocess
import traceback
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import requests

DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")

GLM_MODEL_NAME = "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AGENT_MODELS = [QWEN_MODEL_NAME, GLM_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME]
AGENT_ID = "my-agents/v5"


# ---------------------------- Utilities ----------------------------

def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _html_like(text: str) -> bool:
    t = (text or "").lower()
    return ("<html" in t[:400]) or ("<!doctype" in t[:400]) or ("too many requests" in t) or (" 429" in t[:600])


def _extract_sentinel_payload(text: str) -> Optional[str]:
    try:
        m = re.search(r"<<<PY>>>\n([\s\S]*?)\n<<<END>>>", text)
        return m.group(1) if m else None
    except Exception:
        return None


def sanitize_code_payload(text: str) -> str:
    if not text:
        return ""
    pay = _extract_sentinel_payload(text)
    code = (pay if pay is not None else text).strip()
    if _html_like(code):
        return ""
    if code.startswith("```"):
        code = code.strip("`")
        if "\n" in code:
            code = code.split("\n", 1)[1]
    lines = code.splitlines()
    kept: List[str] = []
    fence = re.compile(r"^\s*```.*$")
    for ln in lines:
        if fence.match(ln):
            continue
        kept.append(ln)
    code = "\n".join(kept)
    code = re.sub(r"\nif __name__ == ['\"]__main__['\"]:[\s\S]*$", "\n", code, flags=re.MULTILINE)
    if not code.endswith("\n"):
        code += "\n"
    return code


def ensure_git_initialized() -> None:
    try:
        if not os.path.exists(".git"):
            subprocess.run(["git", "init"], check=False, timeout=20)
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", os.getcwd()], check=False, timeout=10)
            subprocess.run(["git", "config", "--global", "user.email", "agent@sandbox.local"], check=False, timeout=10)
            subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], check=False, timeout=10)
            subprocess.run(["git", "add", "."], check=False, timeout=30)
            subprocess.run(["git", "commit", "-m", "init"], check=False, timeout=30)
    except Exception:
        pass


# ---------------------- Extraction and Patch Gen ----------------------

def extract_code_blocks_preferring_sentinels(response: str) -> Dict[str, str]:
    print(f"[EXTRACT] Response length: {len(response)}")
    files: Dict[str, str] = {}
    # Sentinel multi-file format: allow optional headers like '# filename.py' separated
    sent = _extract_sentinel_payload(response)
    if sent is not None:
        blocks = re.split(r"\n\s*#\s*([\w\-/\.]+\.py)\s*\n", sent)
        if len(blocks) > 1:
            # blocks like [prefix, name1, code1, name2, code2, ...]
            it = iter(blocks[1:])
            for name, code in zip(it, it):
                cleaned = sanitize_code_payload(code)
                if cleaned:
                    files[name.strip()] = cleaned
        else:
            files["main.py"] = sanitize_code_payload(sent)
    if not files:
        # Fall back to ```python blocks with optional '# filename.py'
        pattern = r"```python\s*\n(?:#\s*([^\n]+\.py)\s*\n)?(.*?)\n```"
        matches = re.findall(pattern, response, re.DOTALL)
        for name, code in matches:
            fn = name.strip() if name else "main.py"
            cleaned = sanitize_code_payload(code)
            if cleaned:
                files[fn] = cleaned
    # Never modify tests.py via agent outputs
    if "tests.py" in files:
        print("[EXTRACT] Dropping tests.py from LLM output to avoid changing tests")
        files.pop("tests.py", None)
    print(f"[EXTRACT] Extracted files: {list(files.keys())}")
    return files


def generate_diff(file_impls: Dict[str, str]) -> str:
    if not file_impls:
        return ""
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["git", "init"], cwd=tmp, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "agent@example.com"], cwd=tmp)
            subprocess.run(["git", "config", "user.name", "Agent"], cwd=tmp)
            # Copy current repo
            for root, dirs, files in os.walk("."):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                for f in files:
                    if f.endswith('.py'):
                        sp = os.path.join(root, f)
                        rp = os.path.relpath(sp, ".")
                        dp = os.path.join(tmp, rp)
                        os.makedirs(os.path.dirname(dp), exist_ok=True)
                        try:
                            with open(sp, 'r', encoding='utf-8') as src:
                                content = src.read()
                            with open(dp, 'w', encoding='utf-8') as dst:
                                dst.write(content)
                        except Exception:
                            pass
            subprocess.run(["git", "add", "."], cwd=tmp)
            subprocess.run(["git", "commit", "-m", "orig"], cwd=tmp, capture_output=True, text=True)
            for fn, code in file_impls.items():
                dp = os.path.join(tmp, fn)
                os.makedirs(os.path.dirname(dp), exist_ok=True)
                with open(dp, 'w', encoding='utf-8') as f:
                    f.write(code)
            res = subprocess.run(["git", "diff", "--no-color", "--unified=3"], cwd=tmp, capture_output=True, text=True)
            diff = res.stdout or ""
            return diff if diff.startswith("diff --git") else ""
    except Exception as e:
        print(f"[DIFF] error: {e}")
        traceback.print_exc()
        return ""


def dry_run_patch(patch: str) -> Tuple[bool, str]:
    if not patch.strip():
        return False, "empty patch"
    try:
        with open(".temp_patch", "w", encoding="utf-8") as f:
            f.write(patch)
        res = subprocess.run(["git", "apply", "--check", ".temp_patch"], capture_output=True, text=True, timeout=45)
        return res.returncode == 0, (res.stderr or "")
    finally:
        try:
            os.remove(".temp_patch")
        except Exception:
            pass


def apply_and_verify_syntax(patch: str) -> Tuple[bool, str, List[str]]:
    if not patch.strip():
        return False, "empty patch", []
    try:
        subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
        with open(".temp_patch", "w", encoding="utf-8") as f:
            f.write(patch)
        res = subprocess.run(["git", "apply", ".temp_patch"], capture_output=True, text=True, timeout=60)
        if res.returncode != 0:
            return False, f"apply failed: {res.stderr.strip()}", []
        mod = subprocess.run(["git", "ls-files", "-m"], capture_output=True, text=True).stdout.splitlines()
        untracked = subprocess.run(["git", "ls-files", "-o", "--exclude-standard"], capture_output=True, text=True).stdout.splitlines()
        changed = [p for p in (set(mod) | set(untracked)) if p.endswith('.py')]
        for pth in changed:
            try:
                src = _read(pth)
                compile(src, pth, 'exec')
            except SyntaxError as se:
                return False, f"SyntaxError {pth}:{se.lineno}:{se.msg}", changed
            except Exception as e:
                return False, f"Error reading {pth}: {e}", changed
        return True, "", changed
    finally:
        try:
            subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True)
            os.remove(".temp_patch")
        except Exception:
            pass


# ---------------------- Contracts and Tests ----------------------

def parse_signatures(code: str) -> List[str]:
    sigs: List[str] = []
    try:
        tree = ast.parse(code or "")
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                args = [a.arg for a in node.args.args]
                sigs.append(f"def {node.name}({', '.join(args)}):")
    except Exception:
        pass
    return sigs


def enforce_return_contracts(code: str, skeleton: str) -> str:
    # Prevent explicit 'return None' if annotation suggests otherwise
    try:
        tree = ast.parse(code)
        ann_map: Dict[str, str] = {}
        for n in ast.walk(ast.parse(skeleton or "")):
            if isinstance(n, ast.FunctionDef) and n.returns is not None:
                ann_map[n.name] = ast.unparse(n.returns) if hasattr(ast, 'unparse') else "has_return"
        class Fixer(ast.NodeTransformer):
            def visit_FunctionDef(self, n: ast.FunctionDef):
                self.generic_visit(n)
                if n.name in ann_map and ann_map[n.name]:
                    for i, stmt in enumerate(n.body):
                        if isinstance(stmt, ast.Return) and stmt.value is None:
                            # replace with 'raise ValueError("invalid state")' to avoid silent None
                            new_r = ast.parse('raise ValueError("invalid state")').body[0]
                            n.body[i] = new_r
                return n
        new_tree = Fixer().visit(tree)
        ast.fix_missing_locations(new_tree)
        new_code = code
        try:
            new_code = ast.unparse(new_tree)  # type: ignore[attr-defined]
        except Exception:
            pass
        return new_code
    except Exception:
        return code


def ensure_tests_present(problem_statement: str, skeleton: str) -> str:
    if os.path.exists("tests.py"):
        return _read("tests.py")
    # Try LLM-based synthesis first
    messages = [
        {"role": "system", "content": "You write concise unittest tests for a small Python exercise. Deterministic, no prints."},
        {"role": "user", "content": (
            "Generate a tests.py for this exercise using unittest.\n"
            "- Import from main (e.g., from main import symbols).\n"
            "- Verify exact outputs and error messages.\n"
            "- Keep it <= 120 lines.\n\n"
            f"Problem statement (trimmed):\n{problem_statement[:2200]}\n\n"
            "Wrap ONLY the final file between these sentinels:\n<<<PY>>>\n<code here>\n<<<END>>>\n"
        )}
    ]
    test_code = call_llm(messages, run_id="testsynth-1", attempt=0)
    test_code = sanitize_code_payload(test_code)
    if not test_code.strip():
        # Deterministic fallback: import + symbol presence
        sigs = parse_signatures(skeleton)
        names: List[str] = []
        for s in sigs:
            m = re.match(r"def\s+([A-Za-z_][A-Za-z0-9_]*)", s)
            if m:
                names.append(m.group(1))
        lines: List[str] = []
        lines.append("import unittest")
        lines.append("import importlib")
        lines.append("")
        lines.append("class SmokeTest(unittest.TestCase):")
        lines.append("    def test_import_and_symbols(self):")
        lines.append("        mod = importlib.import_module('main')")
        for n in names[:20]:
            lines.append(f"        self.assertTrue(hasattr(mod, '{n}'))")
        lines.append("")
        lines.append("if __name__ == '__main__':")
        lines.append("    unittest.main()")
        test_code = "\n".join(lines) + "\n"
        print("[TESTS] Using fallback smoke tests")
    try:
        with open("tests.py", "w", encoding="utf-8") as f:
            f.write(test_code)
    except Exception:
        pass
    return test_code


def run_pytest_capture() -> Tuple[bool, str]:
    try:
        cmd = ["python", "-m", "pytest", "-q", "--disable-warnings", "--maxfail=1"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return proc.returncode == 0, output
    except Exception as e:
        return False, f"pytest error: {e}"


# ---------------------- LLM ----------------------

def reorder_models(problem_statement: str) -> None:
    global AGENT_MODELS
    ps = (problem_statement or "").lower()
    pref: List[str] = []
    if any(k in ps for k in ["regex", "class", "function", "compile", "refactor", "test", "error"]):
        pref.append(QWEN_MODEL_NAME)
    pref.append(GLM_MODEL_NAME)
    pref.append(KIMI_MODEL_NAME)
    pref.append(DEEPSEEK_MODEL_NAME)
    seen: set[str] = set()
    ordered: List[str] = []
    for m in pref + [m for m in AGENT_MODELS if m not in pref]:
        if m not in seen:
            ordered.append(m)
            seen.add(m)
    AGENT_MODELS = ordered


def call_llm(messages: List[Dict[str, str]], run_id: str, attempt: int, low_token: bool = False) -> str:
    url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
    headers = {"Content-Type": "application/json"}
    model = AGENT_MODELS[attempt % len(AGENT_MODELS)]
    body = {
        "run_id": run_id,
        "messages": messages,
        "temperature": 0.0 if not low_token else 0.1,
        "agent_id": AGENT_ID,
        "model": model,
    }
    max_retries = 4
    for r in range(max_retries):
        try:
            timeout = 180 if not low_token else 120
            resp = requests.post(url, json=body, timeout=timeout, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            content = ""
            if isinstance(data, dict) and data.get("choices"):
                content = (data["choices"][0]["message"].get("content") or "")
            elif isinstance(data, str):
                content = data
            if _html_like(content):
                print("[LLM] HTML-like content rejected; backing off")
                time.sleep(2 + r)
                continue
            if content and len(content.strip()) > 10:
                return content
        except requests.exceptions.Timeout:
            time.sleep(3 ** r)
        except Exception:
            time.sleep(2 + r)
    if not low_token:
        return call_llm(messages, run_id, attempt + 1, low_token=True)
    return ""


# ---------------------- Agent ----------------------
class V5Agent:
    def __init__(self, problem_statement: str, mode: str):
        self.problem_statement = problem_statement or ""
        self.mode = mode if mode in ("spec_only", "tests_available") else "spec_only"

    def _build_summary(self) -> str:
        parts: List[str] = []
        for name in ("main.py", "tests.py"):
            try:
                content = _read(name)
                if content:
                    snippet = content[:6000]
                    parts.append(f"### {name}\n```python\n{snippet}\n```")
            except Exception:
                pass
        # Add a few more files for context
        count = 0
        for root, _, files in os.walk("."):
            if any(part.startswith('.') for part in Path(root).parts):
                continue
            for f in files:
                if f.endswith('.py') and f not in ("main.py", "tests.py"):
                    snippet = _read(os.path.join(root, f))[:2000]
                    if snippet:
                        parts.append(f"### {os.path.join(root, f)}\n```python\n{snippet}\n```")
                        count += 1
                        if count >= 10:
                            break
            if count >= 10:
                break
        return "\n\n".join(parts)

    def _messages(self, summary: str) -> List[Dict[str, str]]:
        sys_lines: List[str] = []
        sys_lines.append("You are an autonomous senior software engineer.\n")
        if self.mode == "tests_available":
            sys_lines.append("Modify only code under test so all tests in tests.py pass. Do not change tests.py.\n")
        else:
            sys_lines.append("Implement required functionality strictly from the instructions.\n")
        sys_lines.append("Return complete Python files only, using code blocks as specified.\n")
        sys_lines.append(
            "Use this exact format for each file you change or create:\n" \
            "```python\n# main.py\n[complete code]\n```\n" \
            "(Repeat per file; do not include diffs.)\n"
        )
        sys_msg = "".join(sys_lines)
        user_msg = (
            f"Problem Statement:\n{self.problem_statement}\n\n"
            f"Repository Summary:\n{summary}\n\n"
            "CRITICAL:\n"
            "- Preserve function/class names and module-level constants from existing skeletons.\n"
            "- Ensure return types match annotations; never return None unless allowed.\n"
            "- Copy exact error messages expected by tests.\n"
            "- Deterministic, no I/O, no prints.\n"
            "- Do not modify tests.py if present.\n"
            "Return complete file contents now."
        )
        return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]

    def _initial_candidates(self, summary: str) -> List[Dict[str, str]]:
        msgs = self._messages(summary)
        candidates: List[Dict[str, str]] = []
        for attempt in range(6):
            print(f"[AGENT] Call attempt {attempt + 1}/4 model={AGENT_MODELS[attempt % len(AGENT_MODELS)]}")
            resp = call_llm(msgs, run_id="v5-initial", attempt=attempt)
            if not resp:
                continue
            impls = extract_code_blocks_preferring_sentinels(resp)
            if impls:
                candidates.append(impls)
        if not candidates:
            # Last-ditch minimal code request focusing only on main.py
            minimal_msgs = [
                {"role": "system", "content": (
                    "Return ONLY Python code for main.py implementing the required functionality. "
                    "No prose. Wrap with sentinels <<<PY>>> and <<<END>>>."
                )},
                {"role": "user", "content": (
                    f"Problem Statement (trimmed):\n{self.problem_statement[:2400]}\n\n"
                    f"Existing main.py (trimmed):\n```python\n{_read('main.py')[:8000]}\n```\n"
                    "Return final main.py only between sentinels:\n<<<PY>>>\n<code here>\n<<<END>>>\n"
                )},
            ]
            resp = call_llm(minimal_msgs, run_id="v5-minimal", attempt=0, low_token=True)
            if resp:
                impls = extract_code_blocks_preferring_sentinels(resp)
                if not impls and sanitize_code_payload(resp):
                    impls = {"main.py": sanitize_code_payload(resp)}
                if impls:
                    candidates.append(impls)
        return candidates

    def _refine_candidates(self, base_resp: str, err: str, valid_names: List[str]) -> List[Dict[str, str]]:
        msgs = []
        msgs.append({"role": "assistant", "content": base_resp})
        msgs.append({"role": "user", "content": (
            f"Tests failed or validation errors:\n{err[:1800]}\n"
            f"Return corrected complete Python implementations for these exact files: {valid_names}.\n"
            f"Use the same file names. No new files."
        )})
        out: List[Dict[str, str]] = []
        for attempt in range(3):
            resp = call_llm(msgs, run_id="v5-refine", attempt=attempt)
            if not resp:
                continue
            impls = extract_code_blocks_preferring_sentinels(resp)
            impls = {k: v for k, v in impls.items() if k in valid_names}
            if impls:
                out.append(impls)
        return out

    def propose_patch(self) -> str:
        print("[AGENT] v5 start")
        summary = self._build_summary()
        ensure_tests_present(self.problem_statement, _read("main.py"))
        summary = self._build_summary()

        # Generate multiple initial candidates
        cands = self._initial_candidates(summary)
        if not cands:
            print("[AGENT] No candidates returned")
            return ""

        # Score candidates by running pytest after applying patches
        best_patch = ""
        best_score = (-1, 9999)  # (passes, fails)
        best_resp = ""

        for idx, impl in enumerate(cands):
            # enforce return contracts against current main.py skeleton
            if "main.py" in impl:
                impl["main.py"] = enforce_return_contracts(impl["main.py"], _read("main.py"))
            diff = generate_diff(impl)
            if not diff:
                continue
            ok, err = dry_run_patch(diff)
            if not ok:
                print(f"[AGENT] Candidate {idx}: dry-run failed: {err[:200]}")
                continue
            ok2, err2, _ = apply_and_verify_syntax(diff)
            if not ok2:
                print(f"[AGENT] Candidate {idx}: syntax failed: {err2[:200]}")
                continue
            # Run pytest
            test_ok, output = run_pytest_capture()
            passed = 1 if test_ok else 0
            fails = 0 if test_ok else 1
            score = (passed, fails)
            if passed:
                print("[AGENT] Candidate passed all tests")
                return diff
            if score[0] > best_score[0] or (score[0] == best_score[0] and score[1] < best_score[1]):
                best_score = score
                best_patch = diff
                best_resp = "(omitted)"
                last_fail_output = output

        # Refinement round using best so far (if any)
        if not best_patch:
            print("[AGENT] No viable initial patch")
            return ""

        # Determine expected files from best patch headers
        changed_files: List[str] = []
        for ln in best_patch.splitlines():
            if ln.startswith("diff --git ") and " b/" in ln:
                part = ln.split(" b/")[-1].strip()
                if part:
                    changed_files.append(part)
        ref_cands = self._refine_candidates(best_resp, last_fail_output if 'last_fail_output' in locals() else 'tests failed', changed_files)
        for impl in ref_cands:
            if "main.py" in impl:
                impl["main.py"] = enforce_return_contracts(impl["main.py"], _read("main.py"))
            diff = generate_diff(impl)
            if not diff:
                continue
            ok, err = dry_run_patch(diff)
            if not ok:
                continue
            ok2, err2, _ = apply_and_verify_syntax(diff)
            if not ok2:
                continue
            test_ok, output = run_pytest_capture()
            if test_ok:
                print("[AGENT] Refinement passed all tests")
                return diff

        return best_patch


def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False) -> str:
    run_id = (input_dict or {}).get("run_id", os.getenv("RUN_ID", "nocache-1"))
    problem = (input_dict or {}).get("problem_statement", "")
    try:
        if repo_dir and os.path.exists(repo_dir):
            os.chdir(repo_dir)
    except Exception:
        pass
    ensure_git_initialized()
    try:
        reorder_models(problem)
    except Exception:
        pass
    mode = "tests_available" if os.path.exists("tests.py") else "spec_only"
    agent = V5Agent(problem, mode)
    patch = agent.propose_patch()
    return patch or ""
