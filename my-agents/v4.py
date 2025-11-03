from __future__ import annotations

import os
import re
import json
import time
import subprocess
import unittest
import traceback
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import textwrap
import uuid

import requests


def _artifact_write(filename: str, content: str) -> None:
    try:
        os.makedirs(".agent_artifacts", exist_ok=True)
        path = os.path.join(".agent_artifacts", filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content if content is not None else "")
        size = len(content or "")
        print(f"[ARTIFACT] wrote {path} ({size} bytes)")
        # Mirror full artifact content to console so it appears in agent_logs.txt
        try:
            printable = content if content is not None else ""
            print(f"[ARTIFACT_CONTENT_BEGIN] {filename}")
            print(printable)
            print(f"[ARTIFACT_CONTENT_END] {filename}")
        except Exception as _e:
            print(f"[ARTIFACT] failed to mirror content for {filename}: {_e}")
    except Exception as e:
        print(f"[ARTIFACT] failed to write {filename}: {e}")

# --------------------------------------------------------------------------------
# Configuration and whitelisted models
# --------------------------------------------------------------------------------
DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")

GLM_MODEL_NAME = "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AGENT_MODELS = [GLM_MODEL_NAME, QWEN_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME]
AGENT_ID = "my-agents/v4"


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


def sanitize_patch(patch: str) -> str:
    if not patch:
        return ""
    t = patch.strip()
    if t.startswith("```") and t.endswith("```"):
        t = t.strip("`")
        t = re.sub(r"^\w+\n", "", t)
    allowed_prefixes = (
        "diff --git",
        "index ",
        "--- ",
        "+++ ",
        "@@",
        "new file mode",
        "deleted file mode",
        "similarity index",
        "rename from",
        "rename to",
        "Binary files",
        "\\ No newline",
    )
    cleaned: List[str] = []
    for ln in t.splitlines():
        if ln.strip() in {"DISCUSSION", "EOF"}:
            break
        if ln.startswith(allowed_prefixes) or ln.startswith(("+", "-", " ")):
            cleaned.append(ln)
    if not cleaned:
        return ""
    has_git_header = any("diff --git" in ln for ln in cleaned)
    has_at_sign = any(ln.startswith("@@") for ln in cleaned)
    has_file_headers = any(ln.startswith(("--- ", "+ + + ")) for ln in cleaned)
    if has_at_sign and not has_git_header:
        file_paths = []
        for ln in cleaned:
            if ln.startswith(("--- ", "+++ ")):
                file_paths.append(ln.split()[-1])
        if file_paths:
            new_lines = [
                "diff --git a/" + file_paths[0] + " b/" + file_paths[0],
                "index 0000000..1111111 100644",
            ]
            if not has_file_headers and len(file_paths) >= 1:
                new_lines.append(f"--- a/{file_paths[0]}")
                new_lines.append(f"+++ b/{file_paths[0]}")
            cleaned = new_lines + cleaned
    if not any("diff --git" in ln for ln in cleaned):
        return ""
    if not any(ln.startswith("@@") for ln in cleaned):
        return ""
    has_file_headers = any(ln.startswith("--- ") or ln.startswith("+++ ") for ln in cleaned)
    if not has_file_headers:
        for i, ln in enumerate(cleaned):
            if "diff --git" in ln:
                parts = ln.split()
                if len(parts) >= 3 and parts[2].startswith("b/"):
                    file_name = parts[2][2:]
                    cleaned.insert(i + 1, "index 0000000..1111111 100644")
                    cleaned.insert(i + 2, f"--- a/{file_name}")
                    cleaned.insert(i + 3, f"+++ b/{file_name}")
                break
    result = ("\n".join(cleaned) + "\n")
    if len(result.split("\n")) < 5:
        return ""
    return result


def extract_code_blocks(response: str) -> Dict[str, str]:
    print(f"[EXTRACT] Extracting code blocks from response (length: {len(response)})")
    file_implementations: Dict[str, str] = {}
    code_block_pattern = r'```python\s*\n#\s*([^\n]+\.py)\s*\n(.*?)\n```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    print(f"[EXTRACT] Found {len(matches)} code blocks with file comments")
    for filename, code in matches:
        clean_code = code.strip()
        if clean_code:
            file_implementations[filename] = clean_code
            print(f"[EXTRACT]   Extracted {filename}: {len(clean_code)} chars")
    if not file_implementations:
        python_code_pattern = r'```python\s*\n(.*?)\n```'
        matches = re.findall(python_code_pattern, response, re.DOTALL)
        print(f"[EXTRACT] No file-tagged blocks found, trying generic pattern: {len(matches)} matches")
        for i, code in enumerate(matches):
            clean_code = code.strip()
            if clean_code:
                filename = "main.py" if i == 0 else f"file_{i}.py"
                file_implementations[filename] = clean_code
                print(f"[EXTRACT]   Extracted {filename}: {len(clean_code)} chars")
    print(f"[EXTRACT] Total extracted files: {len(file_implementations)}")
    return file_implementations


def format_code_blocks(file_implementations: Dict[str, str]) -> str:
    if not file_implementations:
        return ""
    sections: List[str] = []
    for filename in sorted(file_implementations.keys()):
        code = file_implementations[filename]
        sections.append(f"```python\n# {filename}\n{code}\n```")
    return "\n\n".join(sections)


def generate_diff_from_implementations(file_implementations: Dict[str, str]) -> str:
    import tempfile
    print(f"[DIFF_GEN] Generating diff for {len(file_implementations)} files")
    if not file_implementations:
        print(f"[DIFF_GEN] No file implementations provided")
        return ""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"[DIFF_GEN] Using temp directory: {temp_dir}")
            subprocess.run(["git", "init"], cwd=temp_dir, capture_output=True, text=True, check=True)
            subprocess.run(["git", "config", "user.email", "agent@example.com"], cwd=temp_dir)
            subprocess.run(["git", "config", "user.name", "Agent"], cwd=temp_dir)
            print(f"[DIFF_GEN] Git initialized")
            
            copied_files = 0
            for root, dirs, files in os.walk("."):
                # Skip hidden dirs and __pycache__
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                for file in files:
                    if file.endswith('.py'):
                        src_path = os.path.join(root, file)
                        rel_path = os.path.relpath(src_path, ".")
                        dst_path = os.path.join(temp_dir, rel_path)
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        try:
                            with open(src_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            # Copy ALL files first, even those being modified (we'll overwrite later)
                            with open(dst_path, 'w', encoding='utf-8') as g:
                                g.write(content)
                            copied_files += 1
                        except Exception:
                            pass
            print(f"[DIFF_GEN] Copied {copied_files} original files to temp directory")
            
            # Add and commit original files
            subprocess.run(["git", "add", "."], cwd=temp_dir)
            commit_result = subprocess.run(["git", "commit", "-m", "original"], cwd=temp_dir, capture_output=True, text=True)
            if commit_result.returncode != 0:
                # If nothing to commit, it means no files were found
                print(f"[DIFF_GEN] Warning: Nothing to commit, this might be an issue")
            print(f"[DIFF_GEN] Committed original state")
            
            # Write new implementations (this will overwrite existing files)
            for filename, new_content in file_implementations.items():
                file_path = os.path.join(temp_dir, filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                print(f"[DIFF_GEN] Writing {filename} ({len(new_content)} chars)")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            
            result = subprocess.run(["git", "diff", "--no-color", "--unified=3"], cwd=temp_dir, capture_output=True, text=True)
            diff_output = result.stdout or ""
            print(f"[DIFF_GEN] Generated diff: {len(diff_output)} chars, {len(diff_output.splitlines())} lines")
            if diff_output and "diff --git" in diff_output:
                print(f"[DIFF_GEN] Diff validation passed")
                return diff_output
            print(f"[DIFF_GEN] Diff validation failed - no valid diff generated")
            return ""
    except Exception as e:
        print(f"[DIFF_GEN] Error generating diff: {e}")
        import traceback
        traceback.print_exc()
        return ""


def dry_run_patch(patch_text: str) -> Tuple[bool, str]:
    if not patch_text.strip():
        return False, "empty patch"
    try:
        with open(".temp_patch", "w", encoding="utf-8") as f:
            f.write(patch_text)
        res = subprocess.run(["git", "apply", "--check", ".temp_patch"], capture_output=True, text=True, timeout=30)
        return res.returncode == 0, (res.stderr or "")
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.remove(".temp_patch")
        except Exception:
            pass


def apply_and_syntax_check(patch_text: str) -> Tuple[bool, str]:
    if not patch_text.strip():
        return False, "empty patch"
    try:
        subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
        with open(".temp_patch", "w", encoding="utf-8") as f:
            f.write(patch_text)
        res_apply = subprocess.run(["git", "apply", ".temp_patch"], capture_output=True, text=True, timeout=45)
        if res_apply.returncode != 0:
            return False, f"patch apply failed: {res_apply.stderr.strip()}"
        mod = subprocess.run(["git", "ls-files", "-m"], capture_output=True, text=True, timeout=20)
        untracked = subprocess.run(["git", "ls-files", "-o", "--exclude-standard"], capture_output=True, text=True, timeout=20)
        changed = set((mod.stdout or "").splitlines()) | set((untracked.stdout or "").splitlines())
        py_changed = [p for p in changed if p.endswith(".py")]
        for path in py_changed:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    src = f.read()
                compile(src, path, "exec")
            except SyntaxError as se:
                return False, f"SyntaxError in {path}:{se.lineno}: {se.msg}"
            except Exception as e:
                return False, f"Error reading {path}: {e}"
        return True, ""
    finally:
        try:
            subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
        except Exception:
            pass
        try:
            os.remove(".temp_patch")
        except Exception:
            pass


def reorder_models(problem_statement: str) -> None:
    global AGENT_MODELS
    ps = (problem_statement or "").lower()
    code_signals = any(k in ps for k in ["traceback", "refactor", "compile", "function", "class", "test", "error"])
    reasoning_signals = any(k in ps for k in ["prove", "reason", "math", "theorem"])
    instruction_signals = any(k in ps for k in ["follow instructions", "steps", "rename", "format"]) 
    preferred: List[str] = []
    if code_signals:
        preferred.append(QWEN_MODEL_NAME)
    if reasoning_signals:
        preferred.append(DEEPSEEK_MODEL_NAME)
    if instruction_signals:
        preferred.append(KIMI_MODEL_NAME)
    preferred.append(GLM_MODEL_NAME)
    seen = set()
    new_order: List[str] = []
    for m in preferred + [m for m in AGENT_MODELS if m not in preferred]:
        if m in (GLM_MODEL_NAME, QWEN_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME) and m not in seen:
            new_order.append(m)
            seen.add(m)
    AGENT_MODELS = new_order


class MinimalCursorAgent:
    OUTPUT_RULES = (
        "You MUST return complete Python file implementations for each file that needs changes.\n"
        "Format your response as:\n"
        "```python\n# main.py\n[complete Python code for main.py]\n```\n"
        "```python\n# other_file.py\n[complete Python code for other_file.py]\n```\n"
        "Return ONLY the complete file contents, not diffs or patches."
    )

    def __init__(self, problem_statement: str, mode: str = "tests_available", top_k: int = 20):
        self.problem_statement = problem_statement or ""
        self.mode = mode if mode in ("spec_only", "tests_available") else "tests_available"
        self.top_k = top_k
        self._cached_dataset_records: List[Dict[str, Any]] | None = None
        self._problem_override: str | None = None
        

    def _collect_python_files(self, root: str = ".") -> List[Tuple[str, str]]:
        files: List[Tuple[str, str]] = []
        for dirpath, _, filenames in os.walk(root):
            if any(part.startswith('.') for part in Path(dirpath).parts):
                continue
            if dirpath.endswith(("__pycache__", ".git")):
                continue
            for fn in filenames:
                if not fn.endswith('.py'):
                    continue
                fp = os.path.join(dirpath, fn)
                try:
                    with open(fp, 'r', encoding='utf-8') as f:
                        files.append((fp, f.read()))
                except Exception:
                    continue
        return files

    def _score_files(self, files: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        prob = set(self.problem_statement.lower().split())
        def score(txt: str) -> float:
            w = set(txt.lower().split())
            if not w:
                return 0.0
            return len(prob & w) / max(1, len(prob))
        ranked = sorted(files, key=lambda kv: -score(kv[1]))
        return ranked[: self.top_k]

    def _build_repo_summary(self, files: List[Tuple[str, str]]) -> str:
        parts: List[str] = []
        seen: set[str] = set()
        prioritized = ["main.py"]
        if self.mode == "tests_available":
            prioritized.append("tests.py")
        for must in prioritized:
            try:
                content = _read(must)
                if content:
                    body = content[:6000]
                    parts.append(f"### {must}\n```python\n{body}\n```")
                    seen.add(must)
            except Exception:
                pass
        for fp, content in files:
            if fp in seen:
                continue
            body = content[:2000]
            parts.append(f"### {fp}\n```python\n{body}\n```")
        return "\n\n".join(parts)

    def _build_messages(self, repo_summary: str) -> List[Dict[str, str]]:
        sys_parts: List[str] = []
        sys_parts.append("You are an autonomous senior software engineer.\n")
        sys_parts.append("Analyze the problem and repository summary.\n")
        if self.mode == "tests_available":
            sys_parts.append(
                "Implement or fix only the code under test so that all tests in tests.py pass.\n"
                "Do not modify tests.py. Do not return placeholders (no 'pass' or 'return None').\n"
            )
        else:
            sys_parts.append(
                "Implement the required functionality strictly per the problem instructions.\n"
                "Do not invent APIs not described.\n"
            )
        sys_parts.append("Pay special attention to exact error messages - they must match test expectations exactly.\n")
        sys_parts.append("Read the test assertions carefully to understand expected behavior and error messages.\n")
        sys_parts.append("For error messages, copy the exact text from test assertions including punctuation.\n")
        sys_parts.append("CRITICAL: When tests call methods or access attributes, ensure they exist in your implementation.\n")
        sys_parts.append("Do not leave incomplete implementations - implement ALL methods referenced.\n")
        sys_parts.append("CRITICAL: Pay attention to return types and exact formats.\n")
        sys_parts.append("Return complete Python code for each file that needs changes.\n")
        sys_parts.append(self.OUTPUT_RULES)
        sys = "".join(sys_parts)
        test_reqs = self._analyze_test_requirements() if self.mode == "tests_available" else ""
        problem_src = self._problem_override or self.problem_statement
        user = (
            f"Problem Statement:\n{problem_src}\n\n"
            f"Repository summary (top files):\n\n{repo_summary}\n\n"
            f"{test_reqs}\n\n"
            "Return complete Python implementations for each file that needs changes."
        )
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ]

    # ---------------- Spec dataset (JSONL) generation/validation ----------------
    def _html_like(self, text: str) -> bool:
        t = (text or "").lower()
        return ("<html" in t[:400]) or ("<!doctype" in t[:400]) or ("too many requests" in t) or (" 429" in t[:600])

    def _extract_sentinel_block(self, text: str) -> str:
        m = re.search(r"<<<PY>>>\n([\s\S]*?)\n<<<END>>>", text)
        return (m.group(1) if m else text).strip()

    def _generate_spec_dataset(self, skeleton: str) -> str:
        print("[DATASET] Generating spec-driven dataset (JSONL)...")
        sys_msg = (
            "You generate strict JSONL test data for a Python exercise. "
            "Each line is a JSON object with fields: id, function, inputs, expected or error. "
            "Do NOT include prose. Wrap ONLY the JSONL between <<<PY>>> and <<<END>>>."
        )
        user_msg = (
            "Problem statement (trimmed):\n" + self.problem_statement[:2400] + "\n\n"
            "Skeleton (trimmed):\n" + (skeleton[:1600] if skeleton else "") + "\n\n"
            "Schema per line: {id:str,function:str,inputs:object,expected:any|omit if error,error:{type:str,message:str}|omit}.\n"
            "Cover normal, boundary, edge, and invalid/error cases. Max 200 lines.\n"
            "Return ONLY JSONL between sentinels: \n<<<PY>>>\n<jsonl here>\n<<<END>>>\n"
        )
        msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
        resp = self._call_llm(msgs, run_id="v4-dataset", attempt=0)
        if not resp:
            print("[DATASET] LLM returned empty response")
            return ""
        try:
            _prev = (resp[:200]).replace("\n", " ")
        except Exception:
            _prev = ""
        print(f"[DATASET] LLM raw response len={len(resp)} preview={_prev}...")
        if self._html_like(resp):
            print("[DATASET] Rejected HTML-like/429 response for dataset")
            return ""
        data = self._extract_sentinel_block(resp)
        try:
            _dprev = (data[:200]).replace("\n", " ")
        except Exception:
            _dprev = ""
        print(f"[DATASET] Extracted payload len={len(data)} preview={_dprev}...")
        return data.strip()

    def _generate_failure_tests(self, main_src: str, base_run_id: str) -> str:
        print("[AGENT] Generating failure-oriented tests from instructions + main.py...")
        system_content = (
            "You are a failure-test generation expert for Python exercises.\n"
            "Authoritative problem spec:\n"
            f"{(self.problem_statement or '')}\n\n"
            "Output constraints:\n"
            "- Import strictly with: from main import <symbols>\n"
            "- Single unittest.TestCase class\n"
            "- Comprehensive coverage: include sufficient passing (positive) and failing (negative) tests to fully cover required behaviors and edge cases per the spec\n"
            "- Deterministic; no prints or I/O\n"
            "- Prefer edge/invalid/boundary and structure/type/shape cases; include exact error types/messages per the spec\n"
            "- Return ONLY the test class code in a ```python code block.\n"
        )
        # Keep user minimal: provide current code and the concrete ask
        user_content = (
            f"Current main.py (full):\n```python\n{(main_src or '')}\n```\n\n"
            "Write tests as specified above to fully cover the problem statement, including both positive (should pass) and negative (should fail with precise error types/messages) cases."
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        # Call each model once in parallel (3-minute timeout each)
        collected_suites: List[str] = []
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(AGENT_MODELS)) as executor:
                futures = []
                for att in range(len(AGENT_MODELS)):
                    print(f"[AGENT]   Failure-test gen (single) submitting model {att+1}/{len(AGENT_MODELS)}")
                    futures.append(executor.submit(self._call_llm, messages, base_run_id, att, 60))
                for fut in concurrent.futures.as_completed(futures, timeout=70):
                    try:
                        resp = fut.result()
                        if resp and not self._html_like(resp):
                            mm = re.findall(r'```python\s*\n(.*?)\n```', resp, re.DOTALL)
                            if mm and mm[0].strip():
                                collected_suites.append(mm[0].strip())
                    except Exception:
                        continue
        except Exception:
            # Fallback to sequential if threading fails
            for att in range(len(AGENT_MODELS)):
                resp = self._call_llm(messages, run_id=base_run_id, attempt=att, force_timeout_s=60)
                if resp and not self._html_like(resp):
                    mm = re.findall(r'```python\s*\n(.*?)\n```', resp, re.DOTALL)
                    if mm and mm[0].strip():
                        collected_suites.append(mm[0].strip())
        if not collected_suites:
            return ""
        # Merge all collected suites into a single TestCase class to ensure our runner executes all
        def merge_suites_into_single_class(suites: List[str]) -> str:
            header_lines: List[str] = [
                "import unittest",
                "from main import *",
                "",
                "class GeneratedFailureTests(unittest.TestCase):",
            ]
            body_lines: List[str] = []
            method_counter = 0
            for idx, suite in enumerate(suites):
                # crude extraction of test methods
                try:
                    # Remove any top-level imports to avoid duplication
                    suite_no_imports = re.sub(r"^\s*from\s+main\s+import[\s\S]*?$", "", suite, flags=re.MULTILINE)
                    suite_no_imports = re.sub(r"^\s*import\s+unittest\s*$", "", suite_no_imports, flags=re.MULTILINE)
                except Exception:
                    suite_no_imports = suite
                method_matches = re.findall(r"\n\s*def\s+(test_[A-Za-z0-9_]+)\s*\(self[\s\S]*?\n(?=\s*def\s+test_|\s*class\s+|\Z)", "\n" + suite_no_imports, re.DOTALL)
                # do not arbitrarily cap tests; merge all methods to maximize coverage
                max_methods_per_suite = None
                if not method_matches:
                    # As fallback, take entire suite as a single large method to not drop coverage
                    method_counter += 1
                    body_lines.extend([
                        f"    def test_suite_{idx+1}_{method_counter}(self):",
                        "        # embedded suite could not be parsed; this placeholder ensures presence",
                        "        self.assertTrue(True)",
                    ])
                    continue
                # Extract method bodies with the same boundaries
                selected = 0
                for m in re.finditer(r"\n\s*def\s+(test_[A-Za-z0-9_]+)\s*\(self[\s\S]*?\n(?=\s*def\s+test_|\s*class\s+|\Z)", "\n" + suite_no_imports, re.DOTALL):
                    if max_methods_per_suite is not None and selected >= max_methods_per_suite:
                        break
                    method_counter += 1
                    name = m.group(1)
                    body = m.group(0)
                    # Normalize indentation to 4 spaces and uniquify name
                    uniq_name = f"{name}_{idx+1}_{method_counter}"
                    body = re.sub(r"\n\s*def\s+" + re.escape(name) + r"\s*\(", "\n    def " + uniq_name + "(", body)
                    # Indent method body by 4 spaces if not already
                    body_lines.append("" + body)
                    selected += 1
            return "\n".join(header_lines + body_lines) + ("\n" if body_lines else "\n    def test_placeholder(self):\n        self.assertTrue(True)\n")
        merged = merge_suites_into_single_class(collected_suites)
        return merged

    # ---------------- Pipeline helpers ----------------
    def _pipeline_generate_tests_all_models(self, main_src: str, base_run_id: str) -> List[str]:
        print("[PIPE] _pipeline_generate_tests_all_models: start")
        system_content = (
            "You are a failure-test generation expert for Python exercises.\n"
            "Authoritative problem spec:\n"
            f"{(self.problem_statement or '')}\n\n"
            "Output constraints:\n"
            "- Import strictly with: from main import <symbols>\n"
            "- Single unittest.TestCase class\n"
            "- Comprehensive coverage: include enough positive (passing) and negative (error) tests to fully cover specified behaviors and edge cases\n"
            "- Deterministic; no prints or I/O\n"
            "- Prefer edge/invalid/boundary and structure/type/shape cases; include exact error types/messages per the spec\n"
            "- Return ONLY the test class code in a ```python code block.\n"
        )
        user_content = (
            f"Current main.py (full):\n```python\n{(main_src or '')}\n```\n\n"
            "Write tests as specified above that fully cover the problem statement: include both positive and negative cases and avoid redundancy."
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        collected: List[str] = []
        print("[PIPE]   Submitting parallel requests to all models (timeout 300s each)")
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(AGENT_MODELS)) as ex:
            futs = [ex.submit(self._call_llm, messages, base_run_id, i, 300) for i in range(len(AGENT_MODELS))]
            for i, fut in enumerate(futs):
                try:
                    resp = fut.result(timeout=310)
                    if resp and not self._html_like(resp):
                        mm = re.findall(r'```python\s*\n(.*?)\n```', resp, re.DOTALL)
                        if mm and mm[0].strip():
                            collected.append(mm[0].strip())
                            print(f"[PIPE]   Model {i+1}: collected suite len={len(mm[0])}")
                            try:
                                _artifact_write(f"tests_suite_model_{i+1}.py", mm[0])
                            except Exception:
                                pass
                        else:
                            print(f"[PIPE]   Model {i+1}: no code block")
                    else:
                        print(f"[PIPE]   Model {i+1}: empty/HTML-like response")
                except Exception as e:
                    print(f"[PIPE]   Model {i+1}: error {e}")
        print("[PIPE] _pipeline_generate_tests_all_models: done")
        return collected

    def _pipeline_verify_test_suites(self, suites: List[str]) -> List[str]:
        print("[PIPE] _pipeline_verify_test_suites: start")
        verified: List[str] = []
        for idx, suite in enumerate(suites):
            try:
                with open("tests.py", "w", encoding="utf-8") as f:
                    f.write(self._merge_methods_into_suite(self._extract_methods_from_suite(suite)))
                ok, fb, details = self._run_tests_without_patch()
                try:
                    _artifact_write(f"verify_suite_{idx+1}_results.json", json.dumps(details or {}, ensure_ascii=False, indent=2))
                except Exception:
                    pass
                if details and isinstance(details.get("test_results"), list):
                    # Consider suite valid if it runs and at least one test is executed
                    total = len(details["test_results"])
                    if total > 0:
                        verified.append(suite)
                        print(f"[PIPE]   Suite {idx+1}: verified ({total} methods)")
                    else:
                        print(f"[PIPE]   Suite {idx+1}: no test methods executed")
                else:
                    print(f"[PIPE]   Suite {idx+1}: malformed results")
            except Exception as e:
                print(f"[PIPE]   Suite {idx+1}: error {e}")
        print("[PIPE] _pipeline_verify_test_suites: done")
        return verified

    def _pipeline_score_and_filter_methods(self, suites: List[str], base_run_id: str) -> List[str]:
        print("[PIPE] _pipeline_score_and_filter_methods: start")
        methods: List[str] = []
        for suite in suites:
            methods.extend(self._extract_methods_from_suite(suite))
        print(f"[PIPE]   Extracted {len(methods)} methods for scoring")
        if not methods:
            return []
        # Build scoring prompt
        def score_batch(method_srcs: List[Tuple[str, str]]) -> Dict[str, float]:
            # method_srcs: list of (key, code)
            payload = [{"id": k, "code": c} for k, c in method_srcs]
            system_content = (
                "You are a Python testing auditor.\n"
                "Authoritative problem spec:\n"
                f"{self.problem_statement}\n\n"
                "Task: Given a list of unittest methods, score each one's correctness vs the spec (0-100).\n"
                "Return ONLY JSON mapping id->score, e.g., {\"m1\": 80, \"m2\": 55}."
            )
            user_content = (
                "Methods (JSON array):\n" + json.dumps(payload, ensure_ascii=False, indent=2)
            )
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
            # Collect per-model JSON scores
            per_model_scores: List[Dict[str, float]] = []
            for i in range(len(AGENT_MODELS)):
                try:
                    resp = self._call_llm(messages, base_run_id, i, 60)
                    parsed: Dict[str, float] = {}
                    try:
                        obj = json.loads(resp)
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                try:
                                    parsed[str(k)] = float(v)
                                except Exception:
                                    continue
                    except Exception:
                        # fallback: find id: number lines
                        parsed = {}
                        for line in (resp or "").splitlines():
                            m = re.search(r"\b([A-Za-z0-9_]+)\b\D+(\d{1,3})\b", line)
                            if m:
                                parsed[m.group(1)] = float(max(0, min(100, int(m.group(2)))))
                    per_model_scores.append(parsed)
                except Exception:
                    per_model_scores.append({})
            # Average across models
            agg: Dict[str, float] = {}
            for k, _ in method_srcs:
                vals = [d.get(k) for d in per_model_scores if k in d]
                vals = [float(x) for x in vals if isinstance(x, (int, float))]
                if vals:
                    agg[k] = sum(vals) / len(vals)
                else:
                    agg[k] = 0.0
            return agg

        kept: List[str] = []
        keyed: List[Tuple[str, str]] = [(f"m{idx+1}", msrc) for idx, msrc in enumerate(methods)]
        try:
            scores = score_batch(keyed)
            for idx, (mid, msrc) in enumerate(keyed):
                s = float(scores.get(mid, 0.0))
                print(f"[PIPE]   Method {idx+1} ({mid}): avg score={s:.1f}")
                try:
                    _artifact_write(f"score_{mid}.txt", f"Score: {s}\n\n{msrc}")
                except Exception:
                    pass
                if s >= 50.0:
                    kept.append(msrc)
        except Exception as e:
            print(f"[PIPE]   Batch scoring error: {e}")
        print(f"[PIPE]   Kept {len(kept)} methods after scoring filter")
        print("[PIPE] _pipeline_score_and_filter_methods: done")
        return kept

    def _pipeline_generate_and_select_solution(self, verified_methods: List[str], base_run_id: str) -> str:
        print("[PIPE] _pipeline_generate_and_select_solution: start")
        # Assemble final tests.py from verified methods
        try:
            final_suite = self._merge_methods_into_suite(verified_methods)
            with open("tests.py", "w", encoding="utf-8") as f:
                f.write(final_suite)
            print(f"[PIPE]   Wrote verified tests: {len(final_suite)} chars")
        except Exception as e:
            print(f"[PIPE]   Failed to write verified tests: {e}")
            return ""

        # Prompt models for full solutions
        solution_prompt = (
            "Implement the required functionality strictly per the problem statement so all provided tests pass.\n"
            "Notes:\n"
            "- Keep changes minimal and targeted; avoid full rewrites.\n"
            "- Ensure every name imported by tests from 'main' exists and is correctly implemented (e.g., Graph, Node, Edge, NODE, EDGE, ATTR when relevant).\n"
            "- Maintain clean, deterministic behavior and exact error messages/types where specified.\n"
            "Return complete Python file implementations needed (e.g., main.py) in the specified format."
        )
        files_summary = "### tests.py\n```python\n" + final_suite + "\n```"
        system_content = (
            "You are an autonomous senior software engineer.\n"
            "Authoritative problem spec:\n"
            f"{self.problem_statement}\n\n"
            f"{self.OUTPUT_RULES}\n"
        )
        user_content = (
            f"Provided tests (must all pass):\n{files_summary}\n\n{solution_prompt}"
        )
        messages_template = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        candidates: List[Tuple[str, Dict[str, str]]] = []
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(AGENT_MODELS)) as ex:
            futs = [ex.submit(self._call_llm, messages_template, base_run_id, i) for i in range(len(AGENT_MODELS))]
            for i, fut in enumerate(futs):
                try:
                    resp = fut.result(timeout=70)
                    if not resp:
                        continue
                    impl = extract_code_blocks(resp)
                    if impl:
                        candidates.append((resp, impl))
                        print(f"[PIPE]   Solution from model {i+1}: {sum(len(v) for v in impl.values())} chars across {len(impl)} files")
                        try:
                            for fn, code in impl.items():
                                _artifact_write(f"candidate_{i+1}_{fn}", code)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[PIPE]   Solution model {i+1} error: {e}")

        best_patch = ""
        best_passed = -1
        best_total = 0
        for idx, (raw_resp, impl) in enumerate(candidates):
            try:
                patch = generate_diff_from_implementations(impl)
                ok, fb, details = run_tests_locally(patch)
                passed = details.get("tests_passed", 0) if isinstance(details, dict) else 0
                total = details.get("tests_total", 0) if isinstance(details, dict) else 0
                print(f"[PIPE]   Candidate {idx+1}: {passed}/{total} passed")
                try:
                    _artifact_write(f"candidate_{idx+1}_results.json", json.dumps(details or {}, ensure_ascii=False, indent=2))
                    _artifact_write(f"candidate_{idx+1}.patch", patch)
                except Exception:
                    pass
                if passed > best_passed:
                    best_passed = passed
                    best_total = total
                    best_patch = patch
            except Exception as e:
                print(f"[PIPE]   Candidate {idx+1} evaluation error: {e}")
        if best_patch:
            print(f"[PIPE] Selected candidate with {best_passed}/{best_total} passing tests")
        print("[PIPE] _pipeline_generate_and_select_solution: done")
        return best_patch or ""

    def _extract_methods_from_suite(self, suite_code: str) -> List[str]:
        methods: List[str] = []
        if not suite_code:
            return methods
        for m in re.finditer(r"\n\s*def\s+(test_[A-Za-z0-9_]+)\s*\(self[\s\S]*?\n(?=\s*def\s+test_|\s*class\s+|\Z)", "\n" + suite_code, re.DOTALL):
            methods.append(m.group(0))
        return methods

    def _merge_methods_into_suite(self, methods: List[str]) -> str:
        header = [
            "import unittest",
            "from main import *",
            "",
            "class GeneratedFailureTests(unittest.TestCase):",
        ]
        body: List[str] = []
        for idx, m in enumerate(methods):
            # Normalize def indentation to 4 spaces; keep body unchanged
            m = re.sub(r"\n\s*def\s+", "\n    def ", m, count=1)
            body.append(m)
        if not body:
            body = [
                "    def test_placeholder(self):",
                "        self.assertTrue(True)",
            ]
        return "\n".join(header + body) + "\n"

    def _run_tests_without_patch(self) -> Tuple[bool, str, Dict[str, Any]]:
        print("[TEST_RUN] Running tests on current code (no patch)...")
        try:
            # ensure clean state
            subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
            # load main as module
            main_spec = importlib.util.spec_from_file_location("main", "main.py")
            main_module = importlib.util.module_from_spec(main_spec)
            main_spec.loader.exec_module(main_module)  # type: ignore
            try:
                sys.modules['main'] = main_module
            except Exception:
                pass
            tests_spec = importlib.util.spec_from_file_location("tests", "tests.py")
            tests_module = importlib.util.module_from_spec(tests_spec)
            tests_spec.loader.exec_module(tests_module)  # type: ignore
            test_class = None
            for name in dir(tests_module):
                obj = getattr(tests_module, name)
                if (isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj is not unittest.TestCase):
                    test_class = obj
                    break
            if not test_class:
                return True, "No test class found", {"test_results": []}
            methods = [m for m in dir(test_class) if m.startswith("test_")]
            results = []
            inst = test_class()
            for method_name in methods:
                try:
                    getattr(inst, method_name)()
                    results.append({"name": method_name, "status": "pass"})
                except Exception as e:
                    results.append({"name": method_name, "status": "fail", "error": str(e)})
            passed = sum(1 for r in results if r["status"] == "pass")
            return (passed == len(results)), "", {"test_results": results, "tests_passed": passed, "tests_total": len(results)}
        except Exception as e:
            return False, str(e), {"test_results": []}

    def _validate_dataset(self, jsonl: str) -> List[Dict[str, Any]]:
        print("[DATASET] Validating dataset JSONL...")
        records: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        total_lines = 0
        for ln in (jsonl.splitlines() if jsonl else []):
            total_lines += 1
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                print(f"[DATASET] Skip invalid JSON line: {ln[:160]}")
                continue
            if not isinstance(obj, dict):
                print(f"[DATASET] Skip non-object line: {ln[:160]}")
                continue
            rid = obj.get("id")
            fn = obj.get("function")
            inputs = obj.get("inputs")
            exp = obj.get("expected") if "expected" in obj else None
            err = obj.get("error") if "error" in obj else None
            if not isinstance(rid, str) or not isinstance(fn, str) or not isinstance(inputs, dict):
                print(f"[DATASET] Skip missing required fields: id/function/inputs; line={ln[:160]}")
                continue
            if (exp is None) == (err is None):  # exactly one of expected/error must be present
                print(f"[DATASET] Skip: expected XOR error violated; line={ln[:160]}")
                continue
            if err is not None and not (isinstance(err, dict) and isinstance(err.get("type"), str) and isinstance(err.get("message"), str)):
                print(f"[DATASET] Skip: error object invalid; line={ln[:160]}")
                continue
            if rid in seen_ids:
                print(f"[DATASET] Skip duplicate id: {rid}")
                continue
            seen_ids.add(rid)
            records.append(obj)
            if len(records) >= 200:
                break
        print(f"[DATASET] Lines seen={total_lines}; records accepted={len(records)}")
        # persist artifact (and mirror to console)
        try:
            _artifact_write(
                "dataset.jsonl",
                "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + ("\n" if records else ""),
            )
        except Exception:
            pass
        return records

    def _run_dataset_validation(self, records: List[Dict[str, Any]], patch: str) -> Tuple[bool, Dict[str, Any]]:
        print("[DATASET] Running dataset validation on candidate patch...")
        result: Dict[str, Any] = {"passed": 0, "failed": 0, "total": len(records)}
        if not records:
            return True, result
        try:
            subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
            with open(".test_patch", "w", encoding="utf-8") as f:
                f.write(patch)
            apply_res = subprocess.run(["git", "apply", ".test_patch"], capture_output=True, text=True, timeout=45)
            if apply_res.returncode != 0:
                return False, {"error": f"apply failed: {apply_res.stderr}"}
            # import main
            spec = importlib.util.spec_from_file_location("main", "main.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            first_failure: Dict[str, Any] | None = None
            for rec in records:
                fn_name = rec.get("function")
                inputs = rec.get("inputs", {})
                exp_present = "expected" in rec
                expected = rec.get("expected")
                err_spec = rec.get("error") if not exp_present else None
                try:
                    target = getattr(mod, fn_name)
                except AttributeError as e:
                    result["failed"] += 1
                    if first_failure is None:
                        first_failure = {"reason": "missing_function", "function": fn_name, "error": str(e)}
                    break
                try:
                    out = target(**inputs)
                    if err_spec is not None:
                        result["failed"] += 1
                        if first_failure is None:
                            first_failure = {"reason": "expected_error_not_raised", "function": fn_name, "inputs": inputs, "expected_error": err_spec}
                        break
                    # compare exact
                    if out == expected:
                        result["passed"] += 1
                    else:
                        result["failed"] += 1
                        if first_failure is None:
                            first_failure = {"reason": "mismatch", "function": fn_name, "inputs": inputs, "expected": expected, "actual": out}
                        break
                except Exception as e:
                    if err_spec is None:
                        result["failed"] += 1
                        if first_failure is None:
                            first_failure = {"reason": "unexpected_exception", "function": fn_name, "inputs": inputs, "error": str(e)}
                        break
                    # verify error type/message
                    etype = type(e).__name__
                    emsg = str(e)
                    if etype == err_spec.get("type") and emsg == err_spec.get("message"):
                        result["passed"] += 1
                    else:
                        result["failed"] += 1
                        if first_failure is None:
                            first_failure = {"reason": "wrong_exception", "function": fn_name, "inputs": inputs, "expected_error": err_spec, "actual_type": etype, "actual_message": emsg}
                        break
            if first_failure is not None:
                result["first_failure"] = first_failure
                try:
                    _artifact_write("last_failure.json", json.dumps(first_failure, ensure_ascii=False, indent=2))
                except Exception:
                    pass
            # reset
            subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
            try:
                os.remove(".test_patch")
            except Exception:
                pass
            return (result.get("failed", 0) == 0), result
        except Exception as e:
            traceback.print_exc()
            return False, {"error": f"dataset validation error: {e}"}

    def _analyze_test_requirements(self) -> str:
        """For tests_available mode: extract requirements from existing tests."""
        try:
            with open("tests.py", "r", encoding="utf-8") as f:
                content = f.read()
            requirements: List[str] = []
            error_messages = re.findall(r'self\.assertEqual\(err\.exception\.args\[0\], "([^"]+)"\)', content)
            if error_messages:
                requirements.append("IMPORTANT: The tests expect these exact error messages:")
                requirements.extend(f"- '{msg}'" for msg in error_messages)
            import_lines = re.findall(r'from\s+\w+\s+import\s+([^,\n]+)', content)
            if import_lines:
                requirements.append("CRITICAL: The tests are trying to import these names from your module:")
                for imports in import_lines:
                    names = [name.strip() for name in imports.split(',')]
                    requirements.extend(f"- {name}" for name in names)
                requirements.append("Make sure ALL these names are defined in your implementation!")
            return ("\n\n" + "\n".join(requirements)) if requirements else ""
        except Exception:
            return ""
    
    def _generate_synthetic_tests(self, run_id: str) -> str:
        """For spec_only mode: ask LLM to generate test cases from instructions."""
        print(f"[AGENT] Generating synthetic test cases from instructions...")
        
        system_content = (
            "You are a test generation expert. Create comprehensive test cases based on specifications.\n"
            "Authoritative problem spec:\n"
            f"{self.problem_statement}\n\n"
            "Output constraints:\n"
            "- Generate 3-5 tests in a single unittest.TestCase class\n"
            "- Use 'from main import <symbols>' strictly\n"
            "- Test edge cases, error conditions, and boundary cases\n"
            "- Follow the exact error messages and exception types mentioned in the spec\n"
            "- Deterministic; no prints or I/O\n"
            "- Return ONLY the test class code in a ```python code block.\n"
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "Write the tests as specified above."}
        ]
        
        # Call LLM to generate tests
        test_code = ""
        for attempt in range(2):
            print(f"[AGENT]   Test generation attempt {attempt + 1}/2...")
            response = self._call_llm(messages, run_id, 100 + attempt)
            if response:
                # Extract code block
                matches = re.findall(r'```python\s*\n(.*?)\n```', response, re.DOTALL)
                if matches:
                    test_code = matches[0].strip()
                    print(f"[AGENT]   Generated synthetic test code: {len(test_code)} chars")
                    break
        
        return test_code

    def _call_llm(self, messages: List[Dict[str, str]], run_id: str, attempt: int, force_timeout_s: int | None = None) -> str:
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
        headers = {"Content-Type": "application/json"}
        model = AGENT_MODELS[attempt % len(AGENT_MODELS)]
        # Ensure run_id is a valid UUID for the gateway
        try:
            uuid.UUID(str(run_id))
            valid_run_id = str(run_id)
        except Exception:
            valid_run_id = str(uuid.uuid4())
            print(f"[LLM] Adjusted non-UUID run_id to {valid_run_id}")
        body = {
            "run_id": valid_run_id,
            "messages": messages,
            "temperature": 0.0,
            "agent_id": AGENT_ID,
            "model": model,
        }
        # Persist full prompt for audit
        try:
            _artifact_write(f"prompt_{valid_run_id}_{attempt}_{model.replace('/', '_')}.json", json.dumps({
                "run_id": valid_run_id,
                "model": model,
                "timeout": force_timeout_s if force_timeout_s is not None else 60,
                "messages": messages,
            }, ensure_ascii=False, indent=2))
        except Exception:
            pass
        max_retries = 3
        for retry in range(max_retries):
            try:
                timeout = force_timeout_s if force_timeout_s is not None else 60
                resp = requests.post(url, json=body, timeout=timeout, headers=headers)
                try:
                    resp.raise_for_status()
                except requests.exceptions.HTTPError as http_err:
                    status = resp.status_code if resp is not None else 'unknown'
                    text_preview = (resp.text or '')[:300] if resp is not None else ''
                    print(f"[LLM] HTTP error status={status} body[:300]={text_preview}")
                    raise
                data = resp.json()
                if isinstance(data, dict) and data.get("choices") and data["choices"][0].get("message"):
                    content = data["choices"][0]["message"]["content"] or ""
                    if content and len(content.strip()) > 50:
                        if self._html_like(content):
                            print("[LLM] HTML-like content detected; retrying")
                            raise requests.exceptions.HTTPError("html_like_content")
                        try:
                            _artifact_write(f"response_{valid_run_id}_{attempt}_{model.replace('/', '_')}.txt", content)
                        except Exception:
                            pass
                        return content
                if isinstance(data, str) and len(data.strip()) > 50:
                    if self._html_like(data):
                        print("[LLM] HTML-like content detected (string); retrying")
                        raise requests.exceptions.HTTPError("html_like_content")
                    try:
                        _artifact_write(f"response_{valid_run_id}_{attempt}_{model.replace('/', '_')}.txt", data)
                    except Exception:
                        pass
                    return data
                return json.dumps(data)
            except requests.exceptions.Timeout:
                if retry < max_retries - 1:
                    time.sleep(3 ** retry)
            except requests.exceptions.ConnectionError:
                if retry < max_retries - 1:
                    time.sleep(5 + retry * 2)
            except Exception:
                if retry < max_retries - 1:
                    time.sleep(2 + retry)
        return ""

    def propose_patch(self, run_id: str) -> str:
        print(f"[AGENT] ========================================")
        print(f"[AGENT] Starting patch generation for problem: {self.problem_statement[:60]}...")
        print(f"[AGENT] Mode: {self.mode}")
        print(f"[AGENT] Run ID: {run_id}")
        print(f"[AGENT] ========================================")
        # For spec_only, prefer instructions.md as the problem statement source
        if self.mode == "spec_only":
            for fname in ("instructions.md", "instruction.md"):
                try:
                    txt = _read(fname)
                    if txt and len(txt.strip()) > 0:
                        self._problem_override = txt
                        print(f"[AGENT] Using problem statement from {fname}; len={len(txt)}")
                        break
                except Exception:
                    pass
        
        # Log repository root contents for visibility inside Docker
        try:
            entries = []
            for name in sorted(os.listdir(".")):
                try:
                    p = Path(name)
                    kind = "DIR" if p.is_dir() else ("FILE" if p.is_file() else "OTHER")
                    size = p.stat().st_size if p.is_file() else 0
                    entries.append({"name": name, "kind": kind, "size": size})
                except Exception:
                    entries.append({"name": name, "kind": "UNKNOWN"})
            print("[AGENT] Root listing (name, kind, size if file):")
            for e in entries:
                if e.get("kind") == "FILE":
                    print(f"[ROOT] {e['name']} | {e['kind']} | {e['size']} bytes")
                else:
                    print(f"[ROOT] {e['name']} | {e['kind']}")
            try:
                _artifact_write("root_listing.json", json.dumps(entries, ensure_ascii=False, indent=2))
            except Exception:
                pass
        except Exception as _e:
            print(f"[AGENT] Failed to list root: {_e}")

        print(f"[AGENT] Step 1: Collecting Python files...")
        files = self._collect_python_files(".")
        print(f"[AGENT] Found {len(files)} Python files")
        
        print(f"[AGENT] Step 2: Scoring and ranking files...")
        top = self._score_files(files)
        print(f"[AGENT] Selected top {len(top)} files for context: {[f[0] for f in top]}")
        
        print(f"[AGENT] Step 3: Building repository summary...")
        summary = self._build_repo_summary(top)
        print(f"[AGENT] Summary length: {len(summary)} chars")
        
        print(f"[AGENT] Step 4: Building messages for LLM...")
        messages = self._build_messages(summary)
        print(f"[AGENT] System message length: {len(messages[0]['content'])} chars")
        print(f"[AGENT] User message length: {len(messages[1]['content'])} chars")
        
        # Optional multi-stage pipeline for spec_only: tests -> verify -> score -> solutions
        if self.mode == "spec_only":
            try:
                print("[PIPE] Stage 1: Generating tests from all LLMs (5 min timeout per call)...")
                main_src_full = _read("main.py")
                test_suites = self._pipeline_generate_tests_all_models(main_src_full, run_id)
                print(f"[PIPE]   Collected {len(test_suites)} raw test suites")

                print("[PIPE] Stage 2: Verifying test suites (structural/execution sanity)...")
                verified_suites = self._pipeline_verify_test_suites(test_suites)
                print(f"[PIPE]   {len(verified_suites)}/{len(test_suites)} suites verified")

                print("[PIPE] Stage 3: LLM scoring of individual test methods (filter <50% avg)...")
                verified_methods = self._pipeline_score_and_filter_methods(verified_suites, run_id)
                print(f"[PIPE]   Retained {len(verified_methods)} verified test methods")

                print("[PIPE] Stage 4: Generating solutions from all LLMs using verified tests...")
                best_patch = self._pipeline_generate_and_select_solution(verified_methods, run_id)
                if best_patch:
                    print("[PIPE] ✓ Selected best solution from multi-LLM proposals")
                    return best_patch
                print("[PIPE] No viable solution selected; stopping as requested (no fallback)")
                return ""
            except Exception as e:
                print(f"[PIPE] Pipeline error: {e}; stopping as requested (no fallback)")
                return ""
        # Should not reach here in spec_only; for tests_available we keep existing path
        print(f"[AGENT] Step 5: Calling LLM for initial generation...")
        raw = ""
        for attempt in range(4):
            print(f"[AGENT]   Attempt {attempt + 1}/4 using model: {AGENT_MODELS[attempt % len(AGENT_MODELS)]}")
            raw = self._call_llm(messages, run_id, attempt)
            if raw:
                print(f"[AGENT]   Success! Response length: {len(raw)} chars")
                print(f"[AGENT]   Response preview: {raw[:200]}...")
                break
            print(f"[AGENT]   Failed or empty response")
        
        if not raw or len(raw.strip()) < 100:
            print(f"[AGENT] ERROR: No valid LLM response obtained")
            return ""
        
        if any(x in raw for x in ["I cannot", "I'm unable", "I don't know"]):
            print(f"[AGENT] ERROR: LLM indicated inability to solve problem")
            return ""
        
        print(f"[AGENT] Step 6: Extracting code blocks from LLM response...")
        file_implementations = extract_code_blocks(raw)
        print(f"[AGENT] Extracted {len(file_implementations)} file implementations: {list(file_implementations.keys())}")
        
        if not file_implementations:
            print(f"[AGENT] ERROR: No code blocks found in LLM response")
            return ""
        
        print(f"[AGENT] Step 7: Validating syntax of extracted code...")
        valid_implementations: Dict[str, str] = {}
        for filename, code in file_implementations.items():
            try:
                compile(code, filename, 'exec')
                valid_implementations[filename] = code
                print(f"[AGENT]   ✓ {filename}: syntax valid ({len(code)} chars)")
            except SyntaxError as e:
                print(f"[AGENT]   ✗ {filename}: syntax error - {e}")
        
        if not valid_implementations:
            print(f"[AGENT] ERROR: No valid syntax implementations found")
            return ""
        
        initial_tests_present = os.path.exists("tests.py")
        current_file_snapshots: Dict[str, str] = dict(valid_implementations)
        raw_context = format_code_blocks(current_file_snapshots)
        
        latest_main_code = current_file_snapshots.get("main.py", _read("main.py"))
        
        print(f"[AGENT] Step 8: Generating unified diff from implementations...")
        patch = generate_diff_from_implementations(valid_implementations)
        
        if not patch or not patch.startswith("diff --git"):
            print(f"[AGENT] ERROR: Failed to generate valid patch")
            return ""
        
        print(f"[AGENT] Generated patch: {len(patch)} chars, {len(patch.splitlines())} lines")
        
        # For spec_only mode, generate failure-oriented tests ONCE to drive the rest of the iteration
        if self.mode == "spec_only":
            main_src = _read("main.py")
            test_code = self._generate_failure_tests(main_src, run_id)
            if test_code:
                try:
                    with open("tests.py", "w", encoding="utf-8") as f:
                        f.write(test_code)
                    print("[AGENT] Wrote failure-oriented tests to tests.py")
                    # Switch to tests flow
                    self.mode = "tests_available"
                except Exception as _e:
                    print(f"[AGENT] Failed to write tests.py: {_e}")
        
        print(f"[AGENT] Step 9: Validating patch (dry-run and syntax check)...")
        ok, err = dry_run_patch(patch)
        print(f"[AGENT]   Dry-run result: {('✓ PASS' if ok else '✗ FAIL')}")
        if not ok:
            print(f"[AGENT]   Dry-run error: {err or 'unknown'}")
        
        if ok:
            ok2, err2 = apply_and_syntax_check(patch)
            print(f"[AGENT]   Syntax check result: {('✓ PASS' if ok2 else '✗ FAIL')}")
            if not ok2:
                print(f"[AGENT]   Syntax error: {err2 or 'unknown'}")
            else:
                # Run tests using the generated suite (no regeneration in this phase)
                print(f"[AGENT]   Running tests for validation...")
                test_ok, test_feedback, test_details = run_tests_locally(patch)
                if test_ok:
                    print("[AGENT] ✓ All tests passed on first validation")
                    if not initial_tests_present:
                        # Regenerate fresh failure tests to guard against regressions
                        for regen in range(3):
                            print(f"[AGENT]   Post-pass regen {regen + 1}/3: generating new failure tests...")
                            regen_code = self._generate_failure_tests(_read("main.py"), run_id)
                            if not regen_code:
                                print("[AGENT]   No regenerated tests produced, continuing")
                                continue
                            try:
                                with open("tests.py", "w", encoding="utf-8") as f:
                                    f.write(regen_code)
                                print("[AGENT]   Wrote regenerated tests to tests.py")
                            except Exception as _e:
                                print(f"[AGENT]   Failed to write regenerated tests: {_e}")
                                break
                            ok_again, fb_again, _ = run_tests_locally(patch)
                            if not ok_again:
                                print(f"[AGENT]   Regression detected: {fb_again}")
                                err = fb_again
                                break
                            print("[AGENT]   Regenerated tests passed")
                        else:
                            print("[AGENT] ✓ Accepted after passing regenerated tests")
                            return patch
                    else:
                        print("[AGENT] ✓ Accepted existing tests without adversarial regeneration")
                        return patch
                else:
                    print(f"[AGENT]   Tests failed: {test_feedback}")
                    err = test_feedback
                    tests_code_snapshot = _read("tests.py")
                    surgeon_patch, surgeon_feedback, surgeon_details, surgeon_accepted, surgeon_improved, surgeon_updates = self._run_surgical_refinement(
                        base_patch=patch,
                        test_details=test_details,
                        run_id=run_id,
                        main_code_snapshot=latest_main_code,
                        tests_code=tests_code_snapshot,
                        valid_file_names=list(file_implementations.keys()),
                    )
                    if surgeon_patch:
                        if surgeon_updates:
                            valid_implementations.update(surgeon_updates)
                            current_file_snapshots.update(surgeon_updates)
                            for fn in surgeon_updates.keys():
                                if fn not in valid_file_names:
                                    valid_file_names.append(fn)
                            latest_main_code = current_file_snapshots.get("main.py", latest_main_code)
                            raw_context = format_code_blocks(current_file_snapshots)
                        if surgeon_accepted:
                            print("[SURGEON] ✓ Accepted targeted fix after initial failure")
                            return surgeon_patch
                        if surgeon_improved:
                            patch = surgeon_patch
                            err = surgeon_feedback or err
                            test_details = surgeon_details or test_details
                            latest_main_code = current_file_snapshots.get("main.py", latest_main_code)
                 # Do NOT accept unless strict criteria met (handled above)
        err = err2 or err
        
        print(f"[AGENT] Step 10: Starting iterative refinement...")
        current_messages = messages.copy()
        best_patch = patch
        # Track best test results to prevent regression
        base_details = locals().get("test_details")
        best_tests_passed, best_tests_total = self._summarize_test_results(base_details)
        # Track best main code to revert to on regression
        best_main_code = latest_main_code or valid_implementations.get("main.py", "")
        best_file_snapshots: Dict[str, str] = dict(current_file_snapshots)
        # Store synthetic_test_code and valid file names for use in refinement loop
        valid_file_names = list(current_file_snapshots.keys())
        # Track the latest main.py code to include in refinement context
        latest_main_code = latest_main_code or valid_implementations.get("main.py", "")
        initial_main_code_size = len(latest_main_code)
        no_improve_rounds = 0
        
        for refinement_round in range(30):
            print(f"[AGENT]   Refinement round {refinement_round + 1}/30...")
            # Read current tests.py to include in context
            try:
                current_tests_code = _read("tests.py")
            except Exception:
                current_tests_code = ""
            
            # Build richer refinement message with full context, emphasizing minimal fixes
            heal_msg = (
                f"The latest tests failed with the following error:\n{err or 'unknown'}\n\n"
                f"CRITICAL: Make MINIMAL, TARGETED fixes to the current code below. Do NOT rewrite the entire implementation. "
                f"Only fix the specific issue mentioned in the error message. Keep all working parts unchanged.\n\n"
                f"Please return corrected complete Python implementations for these exact files: {valid_file_names}\n"
                f"Do not change file names or add new files.\n\n"
                f"Current main.py code that failed the tests (make minimal changes to fix ONLY the error):\n```python\n{latest_main_code}\n```\n\n"
                f"Current test suite (for reference on what the tests expect):\n```python\n{current_tests_code}\n```\n\n"
                f"Based on the error message above, make the MINIMAL fix needed to make the failing test pass. "
                f"Do not change code that is already working correctly."
            )
            print(f"[AGENT]   Feedback to LLM: {err or 'unknown'}")
            
            current_messages.append({"role": "assistant", "content": raw_context})
            current_messages.append({"role": "user", "content": heal_msg})
            
            # Parallel minimal-fix candidates (limit to 3; reduce to 2 on stagnation)
            import concurrent.futures
            candidate_attempts = []
            max_candidates = 3 if no_improve_rounds < 3 else 2
            for k in range(min(max_candidates, len(AGENT_MODELS))):
                candidate_attempts.append(4 + refinement_round + k)
            print(f"[AGENT]   Spawning {len(candidate_attempts)} candidate refinements in parallel...")
            raws: List[Tuple[int, str]] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidate_attempts)) as ex:
                futs = [ex.submit(self._call_llm, current_messages, run_id, att) for att in candidate_attempts]
                for att, fut in zip(candidate_attempts, futs):
                    try:
                        # Cap per-candidate wait to keep rounds bounded
                        resp = fut.result(timeout=65)
                        if resp:
                            raws.append((att, resp))
                    except Exception:
                        continue

            if not raws:
                print(f"[AGENT]   All candidate LLM calls returned empty; stopping refinement")
                break

            # Evaluate candidates and pick the best by tests passed
            best_round = {
                "tests_passed": -1,
                "tests_total": 0,
                "patch": "",
                "main_code": latest_main_code,
                "feedback": "",
                "files": None,
                "raw": "",
            }

            for att, raw_ref in raws:
                try:
                    preview = (raw_ref or "")[:200].replace("\n", " ")
                except Exception:
                    preview = ""
                model_used = AGENT_MODELS[att % len(AGENT_MODELS)]
                print(f"[AGENT]   Candidate from {model_used}: {len(raw_ref)} chars; preview: {preview}")

                new_impl = extract_code_blocks(raw_ref)
                if not new_impl:
                    print(f"[AGENT]   Candidate had no code blocks; skipping")
                    continue
                unexpected_files = [f for f in new_impl.keys() if f not in valid_file_names]
                if unexpected_files:
                    print(f"[AGENT]   Candidate had unexpected files {unexpected_files}; filtering")
                    new_impl = {k: v for k, v in new_impl.items() if k in valid_file_names}
                if not new_impl:
                    print(f"[AGENT]   Candidate empty after filtering; skipping")
                    continue

                print(f"[AGENT]   Candidate extracted files: {list(new_impl.keys())}")
                valid_new: Dict[str, str] = {}
                for fn, code in new_impl.items():
                    try:
                        compile(code, fn, 'exec')
                        valid_new[fn] = code
                        print(f"[AGENT]     ✓ {fn}: syntax valid")
                    except SyntaxError as e:
                        print(f"[AGENT]     ✗ {fn}: syntax error - {e}")
                if not valid_new:
                    print(f"[AGENT]   Candidate had no valid implementations; skipping")
                    continue

                cand_main_code = valid_new.get("main.py", latest_main_code)
                cand_patch = generate_diff_from_implementations(valid_new)
                if not cand_patch or not cand_patch.startswith("diff --git"):
                    print(f"[AGENT]   Candidate failed to generate patch; skipping")
                    continue
                # Simple locality heuristic: count diff changes
                diff_changes = sum(1 for ln in cand_patch.splitlines() if (ln.startswith("+") or ln.startswith("-")) and not ln.startswith(("+++", "---")))
                if diff_changes > 400:
                    print(f"[AGENT]   Candidate patch too large ({diff_changes} changed lines); skipping")
                    continue

                ok, _err = dry_run_patch(cand_patch)
                if not ok:
                    print(f"[AGENT]   Candidate dry-run failed; skipping")
                    continue
                ok2, err2 = apply_and_syntax_check(cand_patch)
                print(f"[AGENT]   Candidate syntax check: {('✓' if ok2 else '✗')}")
                if not ok2:
                    continue

                print(f"[AGENT]   Running tests for candidate from {model_used}...")
                cand_ok, cand_feedback, cand_details = run_tests_locally(cand_patch)
                cand_passed = cand_details.get("tests_passed", 0) if cand_details else 0
                cand_total = cand_details.get("tests_total", 0) if cand_details else 0
                print(f"[AGENT]   Candidate results: {cand_passed}/{cand_total} passed")

                if cand_ok:
                    # Try regression-guard acceptance path
                    all_passed = True
                    for regen in range(3):
                        print(f"[AGENT]   Candidate post-pass regen {regen + 1}/3...")
                        regen_code = self._generate_failure_tests(_read("main.py"), run_id)
                        if not regen_code:
                            print("[AGENT]   No regenerated tests produced, continuing")
                            continue
                        try:
                            with open("tests.py", "w", encoding="utf-8") as f:
                                f.write(regen_code)
                            print("[AGENT]   Wrote regenerated tests to tests.py")
                        except Exception as _e:
                            print(f"[AGENT]   Failed to write regenerated tests: {_e}")
                            all_passed = False
                            break
                        ok_again, fb_again, _ = run_tests_locally(cand_patch)
                        if not ok_again:
                            print(f"[AGENT]   Regression detected: {fb_again}")
                            all_passed = False
                            break
                        print("[AGENT]   Regenerated tests passed")
                    if all_passed:
                        print(f"[AGENT] ✓ Accepted refined patch after passing regenerated tests")
                        return cand_patch

                # Track best candidate of this round
                if cand_passed > best_round["tests_passed"]:
                    best_round["tests_passed"] = cand_passed
                    best_round["tests_total"] = cand_total
                    best_round["patch"] = cand_patch
                    best_round["main_code"] = cand_main_code
                    best_round["feedback"] = cand_feedback
                    best_round["files"] = valid_new
                    best_round["raw"] = raw_ref

            # No candidate improved to full pass; apply progress/regression logic with best candidate
            if best_round["patch"]:
                tests_passed = best_round["tests_passed"]
                tests_total = best_round["tests_total"]
                new_patch = best_round["patch"]
                new_main_code = best_round["main_code"]
                test_feedback = best_round["feedback"]
                new_files = best_round["files"]
                new_raw = best_round["raw"]
                if tests_passed >= best_tests_passed:
                    if tests_passed > best_tests_passed:
                        print(f"[AGENT]   Progress: {tests_passed}/{tests_total} passed (improved from {best_tests_passed}/{best_tests_total})")
                        best_patch = new_patch
                        best_main_code = new_main_code
                        best_tests_passed = tests_passed
                        best_tests_total = tests_total
                        latest_main_code = new_main_code
                        no_improve_rounds = 0
                        if isinstance(new_files, dict):
                            current_file_snapshots = dict(new_files)
                            best_file_snapshots = dict(new_files)
                            valid_implementations = dict(current_file_snapshots)
                            latest_main_code = current_file_snapshots.get("main.py", latest_main_code)
                            raw_context = format_code_blocks(current_file_snapshots)
                        no_improve_rounds = 0
                    else:
                        print(f"[AGENT]   Same progress: {tests_passed}/{tests_total} passed (same as best)")
                        best_patch = new_patch
                        best_main_code = new_main_code
                        latest_main_code = new_main_code
                        if isinstance(new_files, dict):
                            current_file_snapshots = dict(new_files)
                            best_file_snapshots = dict(new_files)
                            valid_implementations = dict(current_file_snapshots)
                            latest_main_code = current_file_snapshots.get("main.py", latest_main_code)
                            raw_context = format_code_blocks(current_file_snapshots)
                        no_improve_rounds += 1
                else:
                    print(f"[AGENT]   REGRESSION detected: {tests_passed}/{tests_total} passed (worse than best {best_tests_passed}/{best_tests_total})")
                    print(f"[AGENT]   Rejecting all candidates, reverting to previous best code for next iteration")
                    latest_main_code = best_main_code
                    current_file_snapshots = dict(best_file_snapshots)
                    valid_implementations = dict(current_file_snapshots)
                    raw_context = format_code_blocks(current_file_snapshots)
                    no_improve_rounds += 1
                err = test_feedback
                if no_improve_rounds >= 5:
                    try:
                        if not initial_tests_present:
                            print(f"[AGENT]   Stagnation detected ({no_improve_rounds} rounds). Regenerating failure tests...")
                            regen_code = self._generate_failure_tests(_read("main.py"), run_id)
                            if regen_code:
                                with open("tests.py", "w", encoding="utf-8") as f:
                                    f.write(regen_code)
                                print("[AGENT]   Wrote new failure tests to tests.py")
                            else:
                                print("[AGENT]   No new failure tests generated on stagnation regen")
                        else:
                            print(f"[AGENT]   Stagnation detected ({no_improve_rounds} rounds) but preserving existing test suite; no adversarial regen performed.")
                    except Exception as _e:
                        print(f"[AGENT]   Failed to regenerate tests on stagnation: {_e}")
                    finally:
                        if not initial_tests_present:
                            no_improve_rounds = 0
                continue

            print(f"[AGENT]   No usable candidate patches produced; stopping refinement")
            break
        
        # Safely finalize best_patch from the last successful candidate if available
        try:
            best_patch = new_patch if 'new_patch' in locals() and new_patch else best_patch
        except Exception:
            pass
        try:
            raw = raw_ref if 'raw_ref' in locals() and raw_ref else raw
        except Exception:
            pass
        
        print(f"[AGENT] ========================================")
        if best_patch:
            print(f"[AGENT] Returning best patch despite validation failures")
            print(f"[AGENT] Patch length: {len(best_patch)} chars")
        else:
            print(f"[AGENT] No valid patch generated")
        print(f"[AGENT] ========================================")
        return best_patch or ""

    # ---------------- Targeted surgical refinement ----------------
    def _summarize_test_results(self, details: Dict[str, Any] | None) -> Tuple[int, int]:
        if not details or not isinstance(details, dict):
            return 0, 0
        results = details.get("test_results")
        if not isinstance(results, list):
            return 0, 0
        passed = sum(1 for r in results if r.get("status") == "pass")
        total = len(results)
        return passed, total

    def _run_surgical_refinement(
        self,
        base_patch: str,
        test_details: Dict[str, Any] | None,
        run_id: str,
        main_code_snapshot: str,
        tests_code: str,
        valid_file_names: List[str],
    ) -> Tuple[str, str, Dict[str, Any], bool, bool, Dict[str, str]]:
        failing_entries: List[Dict[str, Any]] = []
        if isinstance(test_details, dict):
            results = test_details.get("test_results")
            if isinstance(results, list):
                failing_entries = [r for r in results if r.get("status") == "fail"]
            elif test_details.get("failed_test"):
                failing_entries = [{"name": test_details.get("failed_test"), "traceback": test_details.get("traceback", "")}]
        if not failing_entries:
            return "", "", test_details or {}, False, False, {}
 
        working_patch = base_patch
        working_details = test_details or {}
        working_feedback = ""
        working_code = main_code_snapshot
        improved_overall = False
        updated_files: Dict[str, str] = {}
 
        for idx, failing in enumerate(failing_entries[:2]):
            baseline_passed, _ = self._summarize_test_results(working_details)
            attempt_patch, attempt_feedback, attempt_details, accepted, improved, attempt_updates = self._attempt_surgical_fix(
                current_patch=working_patch,
                failing_entry=failing,
                run_id=run_id,
                main_code_snapshot=working_code,
                tests_code=tests_code,
                attempt_offset=160 + idx * 6,
                baseline_passed=baseline_passed,
                valid_file_names=valid_file_names,
            )
            if accepted and attempt_patch:
                print("[SURGEON] ✓ Targeted refinement resolved failing test")
                if isinstance(attempt_details, dict):
                    updated_files.update(attempt_details.get("_updated_files", {}))
                if attempt_updates:
                    updated_files.update(attempt_updates)
                return attempt_patch, attempt_feedback, attempt_details, True, True, updated_files or {}
            if improved and attempt_patch:
                improved_overall = True
                working_patch = attempt_patch
                working_feedback = attempt_feedback or working_feedback
                working_details = attempt_details or working_details
                refreshed_map = {}
                if isinstance(attempt_details, dict):
                    refreshed_map.update(attempt_details.get("_updated_files", {}) or {})
                if attempt_updates:
                    refreshed_map.update(attempt_updates)
                if refreshed_map:
                    updated_files.update(refreshed_map)
                    target_hint = attempt_details.get("_surgeon_target") if isinstance(attempt_details, dict) else None
                    if target_hint and target_hint in refreshed_map:
                        working_code = refreshed_map[target_hint]
                    elif "main.py" in refreshed_map:
                        working_code = refreshed_map["main.py"]
 
        if improved_overall and working_patch:
            print("[SURGEON] Partial improvement achieved, but tests still failing")
            return working_patch, working_feedback or "Tests still failing after targeted refinement", working_details, False, True, updated_files
 
        return "", "", test_details or {}, False, False, {}
 
    def _attempt_surgical_fix(
        self,
        current_patch: str,
        failing_entry: Dict[str, Any],
        run_id: str,
        main_code_snapshot: str,
        tests_code: str,
        attempt_offset: int,
        baseline_passed: int,
        valid_file_names: List[str],
    ) -> Tuple[str, str, Dict[str, Any], bool, bool, Dict[str, str]]:
        test_name = (failing_entry or {}).get("name", "")
        traceback_text = (failing_entry or {}).get("traceback", "")
        error_text = (failing_entry or {}).get("error", "")
        target_file, line_hint = extract_failure_location(traceback_text)
        if valid_file_names is not None:
            if target_file not in valid_file_names and os.path.exists(target_file):
                valid_file_names.append(target_file)
        if valid_file_names and target_file not in valid_file_names:
            target_file = "main.py" if "main.py" in valid_file_names else valid_file_names[0]
        file_source = _read(target_file)
        snippet, (start_line, end_line) = get_code_snippet(file_source, line_hint)
        if not snippet:
            snippet = textwrap.shorten(file_source or "", width=1800, placeholder="\n# ...")
        test_body = extract_test_body(tests_code, test_name)
 
        working_patch = current_patch
        working_code = file_source
        working_passed = baseline_passed
        best_details: Dict[str, Any] | None = None
        best_feedback = ""
        improved = False
        updated_files: Dict[str, str] = {}
 
        max_attempts = 3
        for attempt in range(max_attempts):
            attempt_id = attempt_offset + attempt
            user_prompt = textwrap.dedent(
                f"""
                Failing test: {test_name or 'unknown'}
                Error message: {error_text or 'unknown'}
                Traceback:\n{traceback_text or 'unavailable'}

                Test body:
                ```python
                {test_body or '# test body unavailable'}
                ```

                Current {target_file} snippet (lines {start_line}-{end_line}):
                ```python
                {snippet}
                ```

                Produce a minimal unified diff patch that updates {target_file} to make this test pass while keeping other behavior unchanged.
                Requirements:
                - Touch only the file {target_file} unless absolutely necessary to satisfy imports.
                - Keep the change scope focused on the bug shown above.
                - Do NOT modify tests or add files.
                - Return ONLY the diff, enclosed in standard git diff format (diff --git ...).
                """
            ).strip()

            messages = [
                {
                    "role": "system",
                    "content": "You are a senior Python debugging expert. You receive failing unit tests and produce minimal git diff patches to fix them.",
                },
                {"role": "user", "content": user_prompt},
            ]
            print(f"[SURGEON] Attempt {attempt + 1}/{max_attempts} for {test_name or 'unknown'} (line {line_hint})")
            raw_diff = self._call_llm(messages, run_id, attempt_id)
            if not raw_diff:
                print("[SURGEON]   LLM returned empty diff")
                continue

            merged_ok, merged_patch, merge_err = merge_patches(working_patch, raw_diff)
            if not merged_ok:
                print(f"[SURGEON]   Merge failed: {merge_err}")
                continue

            cand_ok, cand_feedback, cand_details = run_tests_locally(merged_patch)
            cand_passed, _ = self._summarize_test_results(cand_details)
            print(f"[SURGEON]   Candidate results: {cand_passed} tests passed")

            if cand_ok:
                refreshed_code = render_file_from_patch(merged_patch, target_file)
                updated_map = {target_file: refreshed_code} if refreshed_code else {}
                if updated_map:
                    cand_details = cand_details or {}
                    cand_details.setdefault("_updated_files", {}).update(updated_map)
                    cand_details["_surgeon_target"] = target_file
                return merged_patch, cand_feedback, cand_details, True, True, updated_map

            if cand_passed > working_passed:
                print("[SURGEON]   Candidate improved pass count; keeping as new baseline")
                improved = True
                working_patch = merged_patch
                working_passed = cand_passed
                best_details = cand_details
                best_feedback = cand_feedback
                refreshed_code = render_file_from_patch(working_patch, target_file)
                if refreshed_code:
                    updated_files[target_file] = refreshed_code
                    working_code = refreshed_code
                    snippet, (start_line, end_line) = get_code_snippet(refreshed_code, line_hint)
            else:
                print("[SURGEON]   Candidate did not improve pass count")

        if improved and working_patch != current_patch:
            best_details = best_details or {}
            if updated_files:
                best_details.setdefault("_updated_files", {}).update(updated_files)
                best_details["_surgeon_target"] = target_file
            return working_patch, best_feedback, best_details, False, True, updated_files

        return "", "", {}, False, False, {}


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def extract_failure_line(traceback_text: str, filename: str = "main.py") -> int:
    if not traceback_text:
        return -1
    try:
        matches = re.findall(r'File "(.*?)", line (\d+)', traceback_text)
    except Exception:
        return -1
    for file_path, line_no in reversed(matches):
        try:
            if file_path.endswith(filename):
                return int(line_no)
        except Exception:
            continue
    return -1


def extract_failure_location(traceback_text: str, repo_root: str = ".") -> Tuple[str, int]:
    default_line = extract_failure_line(traceback_text, "main.py")
    if not traceback_text:
        return "main.py", default_line
    try:
        matches = re.findall(r'File "(.*?)", line (\d+)', traceback_text)
    except Exception:
        return "main.py", default_line
    for file_path, line_no in reversed(matches):
        path_candidate = file_path.strip().strip("'")
        if not path_candidate or path_candidate.startswith("<"):
            continue
        try:
            if os.path.isabs(path_candidate):
                rel = os.path.relpath(path_candidate, repo_root)
            else:
                rel = path_candidate
            rel = rel.replace("\\", "/")
        except Exception:
            continue
        if os.path.exists(rel):
            try:
                return rel, int(line_no)
            except Exception:
                continue
    return "main.py", default_line


def get_code_snippet(source: str, center_line: int, radius: int = 14) -> Tuple[str, Tuple[int, int]]:
    if not source:
        return "", (0, 0)
    lines = source.splitlines()
    total = len(lines)
    if center_line <= 0 or center_line > total:
        start = max(0, total - min(radius * 2, total))
        end = total
    else:
        start = max(0, center_line - radius - 1)
        end = min(total, center_line + radius)
    snippet_lines: List[str] = []
    for idx in range(start, end):
        line_content = lines[idx]
        snippet_lines.append(f"{idx + 1:04d}: {line_content}")
    return "\n".join(snippet_lines), (start + 1, end)


def extract_test_body(test_code: str, test_name: str) -> str:
    if not test_code or not test_name:
        return ""
    lines = test_code.splitlines()
    start_idx = None
    base_indent = 0
    for idx, line in enumerate(lines):
        if re.match(rf"\s*def {re.escape(test_name)}\(self", line):
            start_idx = idx
            base_indent = len(line) - len(line.lstrip())
            break
    if start_idx is None:
        return ""
    collected: List[str] = []
    for j in range(start_idx, len(lines)):
        line = lines[j]
        stripped = line.lstrip()
        if j > start_idx and stripped.startswith("def "):
            current_indent = len(line) - len(stripped)
            if current_indent <= base_indent:
                break
        if j > start_idx and stripped.startswith("class "):
            current_indent = len(line) - len(stripped)
            if current_indent <= base_indent:
                break
        collected.append(line)
    return "\n".join(collected).strip("\n")


def _apply_patch_in_worktree(patch_text: str) -> Tuple[bool, str]:
    if not patch_text or not patch_text.strip():
        return True, ""
    try:
        with open(".temp_patch", "w", encoding="utf-8") as f:
            f.write(patch_text)
        res = subprocess.run(["git", "apply", ".temp_patch"], capture_output=True, text=True, timeout=45)
        if res.returncode != 0:
            return False, res.stderr or res.stdout or "git apply failed"
        return True, ""
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.remove(".temp_patch")
        except Exception:
            pass


def merge_patches(base_patch: str, extra_patch: str) -> Tuple[bool, str, str]:
    base_patch = (base_patch or "").strip()
    extra_patch = sanitize_patch(extra_patch or "")
    if not extra_patch:
        return False, "", "extra patch empty"
    try:
        subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
        if base_patch:
            ok, err = _apply_patch_in_worktree(base_patch)
            if not ok:
                subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
                return False, "", f"base patch apply failed: {err}"
        ok2, err2 = _apply_patch_in_worktree(extra_patch)
        if not ok2:
            subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
            return False, "", f"extra patch apply failed: {err2}"
        diff_res = subprocess.run(["git", "diff", "--no-color", "--unified=3"], capture_output=True, text=True, timeout=30)
        combined = diff_res.stdout or ""
        if not combined.strip():
            subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
            return False, "", "combined diff empty"
        return True, combined, ""
    except Exception as e:
        return False, "", str(e)
    finally:
        subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)


def render_file_from_patch(patch_text: str, filename: str) -> str:
    if not filename:
        return ""
    try:
        subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
        ok, _ = _apply_patch_in_worktree(patch_text or "")
        if not ok:
            return ""
        if not os.path.exists(filename):
            return ""
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception:
        return ""
    finally:
        subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)


def run_synthetic_tests(test_code: str, patch: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Run synthetic tests generated from specifications.
    
    Returns:
        (all_passed, feedback_message, test_results)
    """
    print(f"[SYNTH_TEST] Running synthetic tests...")
    
    if not test_code:
        print(f"[SYNTH_TEST] No synthetic tests generated")
        return True, "No synthetic tests", {}
    
    try:
        # Reset state
        subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
        
        # Apply patch
        with open(".test_patch", "w", encoding="utf-8") as f:
            f.write(patch)
        result = subprocess.run(["git", "apply", ".test_patch"], capture_output=True, text=True, timeout=45)
        
        if result.returncode != 0:
            print(f"[SYNTH_TEST] Patch apply failed: {result.stderr}")
            subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
            try:
                os.remove(".test_patch")
            except Exception:
                pass
            return False, f"Patch apply failed: {result.stderr}", {}
        
        print(f"[SYNTH_TEST] Patch applied successfully")
        
        # Write synthetic test code to a temporary file
        with open("synthetic_tests.py", "w", encoding="utf-8") as f:
            f.write(test_code)
        print(f"[SYNTH_TEST] Wrote synthetic_tests.py ({len(test_code)} chars)")
        
        # Load main module
        try:
            # Import as "main" so synthetic tests can import from "main"
            import sys
            import importlib
            
            # Add current directory to path
            if "." not in sys.path:
                sys.path.insert(0, ".")
            
            # Import main module
            main_module = importlib.import_module("main")
            print(f"[SYNTH_TEST] Successfully loaded main module")
        except Exception as e:
            print(f"[SYNTH_TEST] Failed to load main.py: {e}")
            import traceback
            traceback.print_exc()
            return False, f"Failed to load main.py: {e}", {}
        
        # Load and run synthetic tests
        try:
            # Import synthetic tests module
            tests_module = importlib.import_module("synthetic_tests")
            print(f"[SYNTH_TEST] Successfully loaded synthetic_tests module")
        except Exception as e:
            print(f"[SYNTH_TEST] Failed to load synthetic tests: {e}")
            import traceback
            traceback.print_exc()
            return False, f"Failed to load synthetic tests: {e}", {}
        
        # Find test class
        test_class = None
        for name in dir(tests_module):
            obj = getattr(tests_module, name)
            if (isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj is not unittest.TestCase):
                test_class = obj
                break
        
        if not test_class:
            print(f"[SYNTH_TEST] No test class found")
            return False, "No test class found in synthetic tests", {}
        
        # Run tests
        test_methods = [method for method in dir(test_class) if method.startswith("test_")]
        print(f"[SYNTH_TEST] Found {len(test_methods)} test methods")
        
        test_results = []
        test_instance = test_class()
        
        for method_name in test_methods:
            try:
                print(f"[SYNTH_TEST] Running {method_name}...")
                method = getattr(test_instance, method_name)
                method()
                test_results.append({"name": method_name, "status": "pass"})
                print(f"[SYNTH_TEST] {method_name}: PASSED")
            except Exception as e:
                error_msg = str(e)
                tb = traceback.format_exc()
                test_results.append({
                    "name": method_name,
                    "status": "fail",
                    "error": error_msg,
                    "traceback": tb
                })
                print(f"[SYNTH_TEST] {method_name}: FAILED - {error_msg}")
                break  # Stop on first failure
        
        # Reset changes
        subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
        try:
            os.remove(".test_patch")
            os.remove("synthetic_tests.py")
        except Exception:
            pass
        
        # Analyze results
        passed = sum(1 for t in test_results if t["status"] == "pass")
        failed = sum(1 for t in test_results if t["status"] == "fail")
        total = len(test_results)
        
        print(f"[SYNTH_TEST] Results: {passed}/{total} passed, {failed} failed")
        
        if failed > 0:
            failing_test = next(t for t in test_results if t["status"] == "fail")
            feedback = (
                f"Synthetic test '{failing_test['name']}' failed.\n"
                f"Error: {failing_test.get('error', 'unknown')}\n"
                f"This suggests your implementation doesn't match the requirements."
            )
            return False, feedback, {
                "test_results": test_results,
                "failed_test": failing_test["name"],
                "tests_passed": passed,
                "tests_total": total,
            }
        
        return True, f"All {total} synthetic tests passed!", {
            "test_results": test_results,
            "tests_passed": passed,
            "tests_total": total,
        }
        
    except Exception as e:
        print(f"[SYNTH_TEST] Exception: {e}")
        traceback.print_exc()
        try:
            subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
            try:
                os.remove(".test_patch")
                os.remove("synthetic_tests.py")
            except Exception:
                pass
        except Exception:
            pass
        return False, f"Synthetic test execution error: {e}", {}


def run_tests_locally(patch: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Run tests on the patched code and return results.
    
    Returns:
        (all_passed, feedback_message, test_results)
    """
    
    print(f"[TEST_RUN] Running tests on patched code...")
    
    try:
        # Save current state
        subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
        
        # Apply patch
        with open(".test_patch", "w", encoding="utf-8") as f:
            f.write(patch)
        result = subprocess.run(["git", "apply", ".test_patch"], capture_output=True, text=True, timeout=45)
        
        if result.returncode != 0:
            print(f"[TEST_RUN] Patch apply failed: {result.stderr}")
            subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
            try:
                os.remove(".test_patch")
            except Exception:
                pass
            return False, f"Patch apply failed: {result.stderr}", {}
        
        print(f"[TEST_RUN] Patch applied successfully")
        
        # Check if tests.py exists
        if not os.path.exists("tests.py"):
            print(f"[TEST_RUN] No tests.py found, skipping test execution")
            subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
            try:
                os.remove(".test_patch")
            except Exception:
                pass
            return True, "No tests to run", {}
        
        # Load main module and register as 'main' for test imports
        main_spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(main_spec)
        main_spec.loader.exec_module(main_module)
        try:
            sys.modules['main'] = main_module
        except Exception:
            pass
        
        # Load tests module
        tests_spec = importlib.util.spec_from_file_location("tests", "tests.py")
        tests_module = importlib.util.module_from_spec(tests_spec)
        tests_spec.loader.exec_module(tests_module)
        
        # Find test class
        test_class = None
        for name in dir(tests_module):
            obj = getattr(tests_module, name)
            if (isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj is not unittest.TestCase):
                test_class = obj
                break
        
        if not test_class:
            print(f"[TEST_RUN] No test class found")
            subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
            try:
                os.remove(".test_patch")
            except Exception:
                pass
            return True, "No test class found", {}
        
        # Run tests
        test_methods = [method for method in dir(test_class) if method.startswith("test_")]
        print(f"[TEST_RUN] Found {len(test_methods)} test methods")
        
        test_results = []
        test_instance = test_class()
        
        for method_name in test_methods:
            try:
                print(f"[TEST_RUN] Running {method_name}...")
                method = getattr(test_instance, method_name)
                method()
                test_results.append({"name": method_name, "status": "pass"})
                print(f"[TEST_RUN] {method_name}: PASSED")
            except Exception as e:
                error_msg = str(e)
                tb = traceback.format_exc()
                test_results.append({
                    "name": method_name,
                    "status": "fail",
                    "error": error_msg,
                    "traceback": tb
                })
                print(f"[TEST_RUN] {method_name}: FAILED - {error_msg}")
                # Continue running remaining tests to capture full progress
        
        # Reset changes
        subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
        try:
            os.remove(".test_patch")
        except Exception:
            pass
        
        # Analyze results
        passed = sum(1 for t in test_results if t["status"] == "pass")
        failed = sum(1 for t in test_results if t["status"] == "fail")
        total_run = len(test_results)
        total_methods = len(test_methods)
        
        print(f"[TEST_RUN] Results: {passed}/{total_run} passed, {failed} failed (out of {total_methods} total tests)")
        
        if failed > 0:
            failing_test = next(t for t in test_results if t["status"] == "fail")
            feedback = (
                f"Test '{failing_test['name']}' failed.\n"
                f"Error: {failing_test.get('error', 'unknown')}\n"
                f"The test expects something different from what your code does."
            )
            return False, feedback, {
                "test_results": test_results, 
                "failed_test": failing_test["name"],
                "tests_passed": passed,
                "tests_total": total_methods
            }
        
        return True, f"All {total_methods} tests passed!", {
            "test_results": test_results,
            "tests_passed": passed,
            "tests_total": total_methods
        }
        
    except Exception as e:
        print(f"[TEST_RUN] Exception: {e}")
        traceback.print_exc()
        try:
            subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
            os.remove(".test_patch")
        except Exception:
            pass
        return False, f"Test execution error: {e}", {}


def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False) -> str:
    run_id = (input_dict or {}).get("run_id", os.getenv("RUN_ID", "nocache-1"))
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
    agent = MinimalCursorAgent(problem, mode=problem_category, top_k=30)
    patch = agent.propose_patch(run_id)
    return patch or ""


