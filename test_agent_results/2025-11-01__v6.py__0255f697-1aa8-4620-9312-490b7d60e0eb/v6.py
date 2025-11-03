import os
import re
import json
import uuid
import subprocess
import unittest
import traceback
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import time

# Self-contained configuration and infra (no dependency on other agent files)
GLM_MODEL_NAME = "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AGENT_MODELS = [GLM_MODEL_NAME, QWEN_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME]

def _artifact_write_console(filename: str, content: str) -> None:
    try:
        printable = content if content is not None else ""
        print(f"[ARTIFACT_CONTENT_BEGIN] {filename}")
        print(printable)
        print(f"[ARTIFACT_CONTENT_END] {filename}")
    except Exception as e:
        print(f"[ARTIFACT] failed to mirror content for {filename}: {e}")

DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://172.17.0.1:1234")

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

def run_tests_locally(patch: str) -> Tuple[bool, str, Dict[str, Any]]:
    print(f"[V6][TEST_RUN] Running tests on patched code...")
    try:
        subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
        with open(".test_patch", "w", encoding="utf-8") as f:
            f.write(patch)
        result = subprocess.run(["git", "apply", ".test_patch"], capture_output=True, text=True, timeout=45)
        if result.returncode != 0:
            print(f"[V6][TEST_RUN] Patch apply failed: {result.stderr}")
            subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
            try:
                os.remove(".test_patch")
            except Exception:
                pass
            return False, f"Patch apply failed: {result.stderr}", {}

        print(f"[V6][TEST_RUN] Patch applied successfully")

        if not os.path.exists("tests.py"):
            print(f"[V6][TEST_RUN] No tests.py found, skipping test execution")
            subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
            try:
                os.remove(".test_patch")
            except Exception:
                pass
            return True, "No tests to run", {}

        main_spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(main_spec)
        main_spec.loader.exec_module(main_module)
        try:
            sys.modules['main'] = main_module
        except Exception:
            pass

        tests_spec = importlib.util.spec_from_file_location("tests", "tests.py")
        tests_module = importlib.util.module_from_spec(tests_spec)
        tests_spec.loader.exec_module(tests_module)

        test_class = None
        for name in dir(tests_module):
            obj = getattr(tests_module, name)
            if (isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj is not unittest.TestCase):
                test_class = obj
                break

        if not test_class:
            print(f"[V6][TEST_RUN] No test class found")
            subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
            try:
                os.remove(".test_patch")
            except Exception:
                pass
            return True, "No test class found", {}

        test_methods = [method for method in dir(test_class) if method.startswith("test_")]
        print(f"[V6][TEST_RUN] Found {len(test_methods)} test methods")

        test_results = []
        test_instance = test_class()
        for method_name in test_methods:
            try:
                print(f"[V6][TEST_RUN] Running {method_name}...")
                method = getattr(test_instance, method_name)
                method()
                test_results.append({"name": method_name, "status": "pass"})
                print(f"[V6][TEST_RUN] {method_name}: PASSED")
            except Exception as e:
                error_msg = str(e)
                tb = traceback.format_exc()
                test_results.append({
                    "name": method_name,
                    "status": "fail",
                    "error": error_msg,
                    "traceback": tb
                })
                print(f"[V6][TEST_RUN] {method_name}: FAILED - {error_msg}")

        subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
        try:
            os.remove(".test_patch")
        except Exception:
            pass

        passed = sum(1 for t in test_results if t["status"] == "pass")
        failed = sum(1 for t in test_results if t["status"] == "fail")
        total_run = len(test_results)
        total_methods = len(test_methods)
        print(f"[V6][TEST_RUN] Results: {passed}/{total_run} passed, {failed} failed (out of {total_methods} total tests)")

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
        print(f"[V6][TEST_RUN] Exception: {e}")
        traceback.print_exc()
        try:
            subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
            os.remove(".test_patch")
        except Exception:
            pass
        return False, f"Test execution error: {e}", {}


class AgentV6:
    """From-scratch v6 agent: small, strict, Python-first, generic-friendly."""

    def __init__(self, problem_statement: str, mode: str = "tests_available", top_k: int = 20) -> None:
        self.problem_statement = problem_statement or ""
        self.mode = mode if mode in ("spec_only", "tests_available") else "tests_available"
        self.top_k = top_k
        print(f"[V6] init: mode={self.mode} top_k={self.top_k} problem_len={len(self.problem_statement)}")

    # ---------- Discovery ----------
    def _collect_python_files(self, root: str = ".") -> List[Tuple[str, str]]:
        print(f"[V6][DISCOVERY] Walking repo for .py files under {root}")
        files: List[Tuple[str, str]] = []
        for dirpath, _, filenames in os.walk(root):
            if any(part.startswith('.') for part in Path(dirpath).parts):
                continue
            if dirpath.endswith(("__pycache__", ".git")):
                continue
            for fn in filenames:
                if fn.endswith('.py'):
                    fp = os.path.join(dirpath, fn)
                    try:
                        with open(fp, 'r', encoding='utf-8') as f:
                            files.append((fp, f.read()))
                    except Exception as e:
                        print(f"[V6][DISCOVERY] Skipped unreadable file {fp}: {e}")
                        continue
        print(f"[V6][DISCOVERY] Collected {len(files)} python files")
        return files

    def _build_repo_summary(self, files: List[Tuple[str, str]]) -> str:
        # Only summarize main.py and tests.py to keep prompts tight
        parts: List[str] = []
        print("[V6][SUMMARY] Building repository summary for main.py and tests.py")
        for must in ("main.py", "tests.py"):
            try:
                with open(must, "r", encoding="utf-8") as f:
                    body = f.read()[:6000]
                parts.append(f"### {must}\n```python\n{body}\n```")
                print(f"[V6][SUMMARY] Included {must} (len={len(body)})")
            except Exception as e:
                print(f"[V6][SUMMARY] {must} not found or unreadable: {e}")
                continue
        out = "\n\n".join(parts)
        print(f"[V6][SUMMARY] Summary length={len(out)}")
        return out

    # ---------- Prompt helpers ----------
    def _testgen_system(self) -> str:
        # Note: not “failure-test” only; request both positive and negative coverage
        return (
            "You are a Python unit test generation expert.\n"
            "Authoritative problem spec:\n"
            f"{self.problem_statement}\n\n"
            "Output constraints:\n"
            "- Import strictly with: from main import <symbols>\n"
            "- Single unittest.TestCase class\n"
            "- Generate up to 20 concise, non-redundant test methods\n"
            "- Cover these categories:\n"
            "  1) successful construction/usage\n"
            "  2) invalid input type\n"
            "  3) shape/arity errors (missing/extra elements)\n"
            "  4) value/type validation errors (wrong types for fields)\n"
            "  5) unknown/unsupported case\n"
            "- Deterministic; no prints or I/O\n"
            "- Use exact exception types/messages from the spec when asserting failures\n"
            "- Return ONLY the test class code in a ```python code block.\n"
        )

    def _solution_system(self, output_rules: str) -> str:
        return (
            "You are an autonomous senior software engineer.\n"
            "Authoritative problem spec:\n"
            f"{self.problem_statement}\n\n"
            f"{output_rules}\n"
        )

    # ---------- LLM call ----------
    def _call_llm(self, messages: List[Dict[str, str]], run_id: str, attempt: int, timeout_s: int = 300) -> str:
        # Use sandbox proxy and mirror prompt/response
        import requests
        # Resolve proxy URL once, in a generic order: INFERENCE_URL -> SANDBOX_PROXY_URL -> default
        DEFAULT_PROXY_URL = (
            os.getenv("INFERENCE_URL")
            or os.getenv("SANDBOX_PROXY_URL")
            or "http://sandbox_proxy"
        )
        model = AGENT_MODELS[attempt % len(AGENT_MODELS)]
        print(f"[V6][LLM] POST {DEFAULT_PROXY_URL.rstrip('/')}/api/inference model={model} attempt={attempt+1}/{len(AGENT_MODELS)} run_id={run_id}")
        try:
            uuid.UUID(str(run_id))
            valid_run_id = str(run_id)
        except Exception:
            valid_run_id = str(uuid.uuid4())
        body = {
            "run_id": valid_run_id,
            "messages": messages,
            "temperature": 0.0,
            "agent_id": "agent-v6",
            "model": model,
        }
        try:
            _artifact_write_console(
                f"prompt_{valid_run_id}_{attempt}_{model.replace('/', '_')}.json",
                json.dumps({"run_id": valid_run_id, "model": model, "timeout": timeout_s, "messages": messages}, ensure_ascii=False, indent=2),
            )
        except Exception:
            pass

        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
        headers = {"Content-Type": "application/json"}
        max_retries = 3
        last_err = None
        for r in range(max_retries):
            try:
                resp = requests.post(url, json=body, timeout=timeout_s, headers=headers)
                print(f"[V6][LLM] HTTP {resp.status_code} len={len(resp.text)}")
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                last_err = e
                print(f"[V6][LLM] Request failed (retry {r+1}/{max_retries}): {e}")
                time.sleep(1)
        else:
            raise last_err
        if isinstance(data, dict) and data.get("choices"):
            content = (data["choices"][0].get("message", {}).get("content") or "")
            print(f"[V6][LLM] Received choices content len={len(content)}")
            try:
                _artifact_write_console(
                    f"llm_response_{valid_run_id}_{attempt}_{model.replace('/', '_')}.txt",
                    content,
                )
            except Exception:
                pass
            return content
        if isinstance(data, str):
            print(f"[V6][LLM] Received raw string len={len(data)}")
            try:
                _artifact_write_console(
                    f"llm_response_{valid_run_id}_{attempt}_{model.replace('/', '_')}.txt",
                    data,
                )
            except Exception:
                pass
            return data
        dumped = json.dumps(data)
        print(f"[V6][LLM] Received JSON len={len(dumped)}")
        try:
            _artifact_write_console(
                f"llm_response_{valid_run_id}_{attempt}_{model.replace('/', '_')}.json",
                dumped,
            )
        except Exception:
            pass
        return dumped

    # ---------- Test curation (spec_only only) ----------
    def _generate_and_select_tests(self, main_src: str, run_id: str) -> str:
        print("[V6][TESTS] Generate tests: start")
        system_content = self._testgen_system()
        user_content = (
            f"Current main.py (full):\n```python\n{main_src}\n```\n\n"
            "Write the tests as specified above."
        )
        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]
        suites: List[str] = []
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(AGENT_MODELS)) as ex:
            print(f"[V6][TESTS] Submitting {len(AGENT_MODELS)} parallel model requests")
            futs = [ex.submit(self._call_llm, messages, run_id, i, 300) for i in range(len(AGENT_MODELS))]
            per_req = 300
            margin = 60
            wait_timeout = per_req + margin
            try:
                for fut in concurrent.futures.as_completed(futs, timeout=wait_timeout):
                    try:
                        resp = fut.result()
                        if resp and "```python" in resp:
                            m = re.findall(r"```python\s*\n(.*?)\n```", resp, re.DOTALL)
                            if m and m[0].strip():
                                suites.append(m[0].strip())
                                print(f"[V6][TESTS] Collected suite len={len(m[0].strip())}")
                            else:
                                print("[V6][TESTS] No python block found in response")
                        else:
                            print("[V6][TESTS] Empty/invalid response from model")
                    except Exception as e:
                        print(f"[V6][TESTS] Model call error: {e}")
                        continue
            except TimeoutError:
                print(f"[V6][TESTS] Timeout waiting for futures; proceeding with {len(suites)} collected suites")
        if not suites:
            print("[V6][TESTS] No suites generated")
            return ""
        selected = self._select_best_suite(suites, run_id)
        # Audit and augment for generic coverage gaps
        if selected:
            selected = self._augment_suite_for_gaps(run_id, selected)
        # Ensure all exception contracts from the spec are asserted at least once
        if selected:
            selected = self._augment_suite_for_exception_contracts(run_id, selected)
        # General negative variants (type, arity/shape, unknown) based on contracts
        if selected:
            selected = self._augment_suite_for_exception_variants(run_id, selected)
        # If coverage is thin (<3 methods), attempt a single coverage-expansion prompt
        if selected:
            methods = self._extract_methods(selected)
            if len(methods) < 3:
                print("[V6][TESTS] Selected suite has <3 methods; requesting expanded coverage once")
                expand_messages = [
                    {"role": "system", "content": self._testgen_system()},
                    {"role": "user", "content": "Expand coverage to include at least three distinct tests, avoiding redundancy. Return only the unittest.TestCase class in a ```python block."},
                ]
                try:
                    more = self._call_llm(expand_messages, run_id, 0, 300)
                    if more and "```python" in more:
                        mm = re.findall(r"```python\s*\n(.*?)\n```", more, re.DOTALL)
                        if mm and mm[0].strip():
                            print("[V6][TESTS] Collected expanded suite")
                            suites.append(mm[0].strip())
                            # Reselect best including expansion
                            selected = self._select_best_suite(suites, run_id) or selected
                except Exception as e:
                    print(f"[V6][TESTS] Expansion request failed: {e}")
        # Cap final number of test methods to 20 to keep iterations efficient
        if selected:
            methods = self._extract_methods(selected)
            if len(methods) > 20:
                methods = methods[:20]
                selected = self._merge_suite(methods)
        print(f"[V6][TESTS] Selected suite present={bool(selected)} len={len(selected) if selected else 0}")
        return selected

    def _extract_methods(self, suite: str) -> List[str]:
        methods = [m.group(0) for m in re.finditer(r"\n\s*def\s+(test_[A-Za-z0-9_]+)\s*\(self[\s\S]*?\n(?=\s*def\s+test_|\s*class\s+|\Z)", "\n" + suite, re.DOTALL)]
        print(f"[V6][TESTS] Extracted {len(methods)} methods from suite")
        return methods

    def _filter_methods_strict(self, methods: List[str]) -> List[str]:
        filtered: List[str] = []
        for body in methods:
            # Drop methods that import from modules other than main
            if re.search(r"^\s*from\s+(?!main\b).*import", body, flags=re.MULTILINE):
                continue
            # Very conservative: drop references to common hallucinated fields
            # if re.search(r"\.(data|attributes)\b", body):
            #     continue
            filtered.append(body)
        # Cap to 3
        kept = filtered[:]
        print(f"[V6][TESTS] Kept {len(kept)} methods after strict filter")
        return kept

    def _score_methods(self, methods: List[str], run_id: str) -> float:
        if not methods:
            return 0.0
        system_content = (
            "You are a Python testing auditor.\n"
            "Authoritative problem spec:\n"
            f"{self.problem_statement}\n\n"
            "Task: Given a list of unittest methods, score each one's correctness vs the spec (0-100).\n"
            "Return ONLY JSON mapping id->score, e.g., {\"m1\": 80, \"m2\": 55}."
        )
        payload = [{"id": f"m{i+1}", "code": m} for i, m in enumerate(methods)]
        user_content = "Methods (JSON array):\n" + json.dumps(payload, ensure_ascii=False, indent=2)
        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]
        scores: List[float] = []
        for i in range(len(AGENT_MODELS)):
            try:
                resp = self._call_llm(messages, run_id, i, 300)
                obj = json.loads(resp)
                if isinstance(obj, dict):
                    for v in obj.values():
                        try:
                            scores.append(float(v))
                        except Exception:
                            continue
            except Exception as e:
                print(f"[V6][TESTS] Scoring error: {e}")
                continue
        avg = (sum(scores) / len(scores)) if scores else 0.0
        print(f"[V6][TESTS] Average score={avg:.1f} from {len(scores)} votes")
        return avg

    def _merge_suite(self, methods: List[str]) -> str:
        header = [
            "import unittest",
            "from main import *",
            "",
            "class GeneratedTests(unittest.TestCase):",
        ]
        body = []
        for m in methods:
            body.append(re.sub(r"\n\s*def\s+", "\n    def ", m, count=1))
        if not body:
            body = [
                "    def test_placeholder(self):",
                "        self.assertTrue(True)",
            ]
        merged = "\n".join(header + body) + "\n"
        print(f"[V6][TESTS] Merged suite len={len(merged)}")
        return merged

    def _select_best_suite(self, suites: List[str], run_id: str) -> str:
        print(f"[V6][TESTS] Selecting best suite from {len(suites)} candidates")
        best_suite = ""
        best_score = -1.0
        for s in suites:
            methods = self._extract_methods(s)
            methods = self._filter_methods_strict(methods)
            if not methods:
                continue
            avg = self._score_methods(methods, run_id)
            if avg > best_score:
                best_score = avg
                best_suite = self._merge_suite(methods)
        # Require strong confidence
        decided = best_suite if best_score >= 80.0 else ""
        print(f"[V6][TESTS] Best score={best_score:.1f} accepted={bool(decided)}")
        return decided

    def _audit_suite_gaps(self, suite: str) -> Dict[str, bool]:
        txt = suite or ""
        has_success = ("assert" in txt) or ("self.assert" in txt)
        has_type = "TypeError" in txt
        has_value = "ValueError" in txt
        has_unknown = ("Unknown" in txt) or ("unknown" in txt)
        # Try to distinguish missing vs extra arity
        has_shape_incomplete = ("incomplete" in txt) or ("missing" in txt)
        has_shape_extra = ("too many" in txt) or ("extra" in txt)
        return {
            "success": has_success,
            "invalid_type": has_type,
            "shape_incomplete": has_shape_incomplete,
            "shape_extra": has_shape_extra,
            "value": has_value,
            "unknown": has_unknown,
        }

    def _augment_suite_for_gaps(self, run_id: str, base_suite: str) -> str:
        # Determine desired categories from spec hints
        desired = set(["success", "invalid_type", "value"])  # always useful
        ps_lower = (self.problem_statement or "").lower()
        if "incomplete" in ps_lower or "missing" in ps_lower or "arity" in ps_lower:
            desired.add("shape_incomplete")
        if "extra" in ps_lower or "too many" in ps_lower:
            desired.add("shape_extra")
        if "unknown" in ps_lower:
            desired.add("unknown")

        gaps = self._audit_suite_gaps(base_suite)
        missing = [k for k in desired if not gaps.get(k, False)]
        if not missing:
            return base_suite
        print(f"[V6][TESTS] Audit found gaps: {missing}; requesting augmentation")
        messages = [
            {"role": "system", "content": self._testgen_system()},
            {"role": "user", "content": (
                "Augment the following unittest.TestCase to add ONLY the missing categories: "
                + ", ".join(missing)
                + ". Keep the class name and existing tests; add new non-redundant tests. "
                "Return only the complete class in a ```python block.\n\n"
                "Existing suite:\n```python\n" + base_suite + "\n```"
            )},
        ]
        try:
            resp = self._call_llm(messages, run_id, 0, 300)
            m = re.findall(r"```python\s*\n(.*?)\n```", resp, re.DOTALL)
            if m and m[0].strip():
                print("[V6][TESTS] Received augmented suite")
                return m[0].strip()
        except Exception as e:
            print(f"[V6][TESTS] Augmentation error: {e}")
        return base_suite

    def _extract_exception_contracts_from_spec(self) -> List[Tuple[str, str]]:
        text = self.problem_statement or ""
        # Capture ExceptionType("Exact message") patterns
        pat = r"raise\s+(?P<exc>[A-Za-z_][A-Za-z0-9_]*Error)\(\s*[\"\'](?P<msg>[^\"\']+)[\"\']\s*\)"
        found = [(m.group("exc"), m.group("msg")) for m in re.finditer(pat, text)]
        # Deduplicate preserving order
        seen = set()
        uniq: List[Tuple[str, str]] = []
        for it in found:
            if it not in seen:
                uniq.append(it)
                seen.add(it)
        print(f"[V6][TESTS] Extracted {len(uniq)} exception contracts from spec")
        return uniq

    def _audit_exception_coverage(self, suite: str, contracts: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        missing: List[Tuple[str, str]] = []
        code = suite or ""
        for exc, msg in contracts:
            msg_present = (msg in code)
            exc_present = (exc in code)
            if not (msg_present and exc_present):
                missing.append((exc, msg))
        if missing:
            print(f"[V6][TESTS] Missing exception assertions: {missing}")
        return missing

    def _augment_suite_for_exception_contracts(self, run_id: str, base_suite: str) -> str:
        contracts = self._extract_exception_contracts_from_spec()
        if not contracts:
            return base_suite
        missing = self._audit_exception_coverage(base_suite, contracts)
        if not missing:
            return base_suite
        # Compose a generic augmentation prompt listing contracts
        lines = [f"- {exc}: \"{msg}\"" for exc, msg in missing]
        list_text = "\n".join(lines)
        print(f"[V6][TESTS] Requesting augmentation for contracts:\n{list_text}")
        messages = [
            {"role": "system", "content": self._testgen_system()},
            {"role": "user", "content": (
                "Augment the following unittest.TestCase by adding the minimal non-redundant tests so that each of these (ExceptionType, exact message) pairs is asserted at least once with the correct exception type and exact message.\n"
                "Do not remove or rename existing tests; keep class name; avoid redundancy; stay deterministic.\n\n"
                "Contracts to cover:\n" + list_text + "\n\n"
                "Return only the complete class in a ```python block.\n\n"
                "Existing suite:\n```python\n" + base_suite + "\n```"
            )},
        ]
        try:
            resp = self._call_llm(messages, run_id, 1, 300)
            m = re.findall(r"```python\s*\n(.*?)\n```", resp, re.DOTALL)
            if m and m[0].strip():
                print("[V6][TESTS] Received contract-augmented suite")
                return m[0].strip()
        except Exception as e:
            print(f"[V6][TESTS] Contract augmentation error: {e}")
        return base_suite

    def _augment_suite_for_exception_variants(self, run_id: str, base_suite: str) -> str:
        # General, domain-agnostic augmentation to add missing negative variants
        contracts = self._extract_exception_contracts_from_spec()
        if not contracts or not base_suite:
            return base_suite

        guide = (
            "For each (ExceptionType, message) below, ensure there is at least one test that triggers it via a TYPE mismatch "
            "and, if the API uses positional elements or arguments, at least one test that triggers it via an ARITY/shape mismatch "
            "(missing or extra elements / wrong number of args). Also add one UNKNOWN/unsupported-value variant where applicable. "
            "Do not duplicate existing tests; add only minimal missing cases. Keep the existing class name and imports. "
            "Use only 'from main import ...'. Deterministic; no I/O or randomness. "
            "Return ONLY the complete unittest.TestCase class in a ```python block."
        )

        lines = [f"- {exc}: \"{msg}\"" for exc, msg in contracts]
        list_text = "\n".join(lines)

        messages = [
            {"role": "system", "content": self._testgen_system()},
            {"role": "user", "content": (
                guide + "\n\n"
                "Existing suite:\n```python\n" + base_suite + "\n```\n\n"
                "Exception contracts:\n" + list_text
            )},
        ]
        try:
            resp = self._call_llm(messages, run_id, 0, 300)
            m = re.findall(r"```python\s*\n(.*?)\n```", resp, re.DOTALL)
            if m and m[0].strip():
                print("[V6][TESTS] Received exception variant augmentation")
                return m[0].strip()
        except Exception as e:
            print(f"[V6][TESTS] Exception variant augmentation error: {e}")
        return base_suite

    def _format_failure_feedback(self, details: Dict[str, Any]) -> str:
        try:
            failed = details.get("failed_test") or ""
            tests_total = details.get("tests_total", 0)
            tests_passed = details.get("tests_passed", 0)
            tr = details.get("test_results", []) or []
            failing = next((t for t in tr if t.get("status") == "fail"), None)
            err = failing.get("error", "") if failing else ""
            tb = failing.get("traceback", "") if failing else ""
            return (
                f"Local tests: {tests_passed}/{tests_total} passed.\n"
                f"Failing test: {failed}\n"
                f"Error: {err}\n"
                f"Traceback:\n{tb}\n"
            )
        except Exception:
            return "Local tests failed, but failure details could not be parsed."

    def _extract_pass_fail_sets(self, details: Dict[str, Any]) -> Tuple[set, set]:
        tr = (details or {}).get("test_results", []) or []
        passed = {t.get("name") for t in tr if t.get("status") == "pass"}
        failed = {t.get("name") for t in tr if t.get("status") == "fail"}
        return passed, failed

    # ---------- Solve ----------
    def propose_patch(self, run_id: str) -> str:
        print("[V6][SOLVE] Propose patch: start")
        # Build summary and tests
        main_src = ""
        try:
            with open("main.py", "r", encoding="utf-8") as f:
                main_src = f.read()
            print(f"[V6][SOLVE] Loaded main.py len={len(main_src)}")
        except Exception as e:
            print(f"[V6][SOLVE] main.py not found: {e}")
            main_src = ""

        if self.mode == "spec_only":
            print("[V6][SOLVE] Mode=spec_only: generating tests")
            tests = self._generate_and_select_tests(main_src, run_id)
            if not tests:
                print("[V6][SOLVE] No tests generated; aborting")
                return ""
            try:
                with open("tests.py", "w", encoding="utf-8") as f:
                    f.write(tests)
                print(f"[V6][SOLVE] Wrote tests.py len={len(tests)}")
            except Exception as e:
                print(f"[V6][SOLVE] Failed writing tests.py: {e}")
                return ""

        # Request implementation from LLM (single pass, then enforce 100% with retries could be added)
        repo_summary = self._build_repo_summary(self._collect_python_files("."))
        OUTPUT_RULES = (
            "You MUST return complete Python file implementations for each file that needs changes.\n"
            "Format your response as:\n"
            "```python\n# main.py\n[complete Python code for main.py]\n```\n"
            "Return ONLY the complete file contents, not diffs or patches.\n"
            "IMPORTANT: Return EXACTLY ONE code block and NOTHING else.\n"
            "Start the block with:```python\\n# main.py\\n (include this header).\n"
            "If you include prose or omit the '# main.py' header, the tool may discard your answer.\n"
        )
        system_content = self._solution_system(OUTPUT_RULES)
        user_content = (
            "Implement strictly per the spec so all tests pass.\n\n"
            f"Repository summary:\n{repo_summary}\n"
        )
        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]

        # Collect one or more candidates and pick the one with 100% pass
        candidates: List[Tuple[str, str]] = []
        for i in range(len(AGENT_MODELS)):
            try:
                resp = self._call_llm(messages, run_id, i, 300)
                blocks = re.findall(r"```python\s*\n#\s*main\.py\n([\s\S]*?)\n```", resp, re.DOTALL)
                if not blocks:
                    # Fallback: accept first python block without header
                    blocks = re.findall(r"```python\s*\n([\s\S]*?)\n```", resp, re.DOTALL)
                if blocks and blocks[0].strip():
                    src = blocks[0].strip()
                    patch = self._build_single_file_patch("main.py", src)
                    candidates.append((src, patch))
                    print(f"[V6][SOLVE] Candidate {len(candidates)} collected (patch_len={len(patch)})")
            except Exception as e:
                print(f"[V6][SOLVE] Candidate generation error: {e}")
                continue
        # Evaluate initial candidates and keep the best-so-far (max tests_passed)
        best_patch = ""
        best_src = ""
        best_passed = -1
        best_details: Dict[str, Any] = {}
        for src, patch in candidates:
            print("[V6][SOLVE] Running local tests for a candidate patch")
            ok, _, details = run_tests_locally(patch)
            total = details.get("tests_total", 0) if isinstance(details, dict) else 0
            passed = details.get("tests_passed", 0) if isinstance(details, dict) else 0
            if ok and isinstance(details, dict) and total == passed:
                print("[V6][SOLVE] Found 100% passing patch")
                return patch
            if passed > best_passed:
                best_patch = patch
                best_src = src
                best_passed = passed
                best_details = details if isinstance(details, dict) else {}

        # Iterative repair with regression protection
        print("[V6][SOLVE] Entering repair loop (up to 10 rounds, no regressions)")
        tests_snapshot = ""
        try:
            with open("tests.py", "r", encoding="utf-8") as f:
                tests_snapshot = f.read()
        except Exception:
            tests_snapshot = ""

        rounds = 10
        for r in range(rounds):
            if not best_patch:
                break
            # Re-run to get latest baseline
            ok, _, details = run_tests_locally(best_patch)
            total = details.get("tests_total", 0) if isinstance(details, dict) else 0
            passed = details.get("tests_passed", 0) if isinstance(details, dict) else 0
            if ok and isinstance(details, dict) and total == passed:
                print("[V6][SOLVE] Repair loop: found 100% passing patch")
                return best_patch
            feedback = self._format_failure_feedback(details if isinstance(details, dict) else {})
            print(f"[V6][SOLVE] Repair round {r+1}/{rounds}: preparing fix request (baseline passed={passed}/{total})")

            repair_rules = (
                "You MUST return complete Python file implementations for each file that needs changes.\n"
                "Format your response as:\n"
                "```python\n# main.py\n[complete Python code for main.py]\n```\n"
                "Return ONLY the complete file contents, not diffs or patches.\n"
                "Do NOT regress: all previously passing behaviors must continue to pass.\n"
                "IMPORTANT: Return EXACTLY ONE code block and NOTHING else.\n"
                "Start the block with:```python\\n# main.py\\n (include this header).\n"
                "If you include prose or omit the '# main.py' header, the tool may discard your answer.\n"
            )
            repair_messages = [
                {"role": "system", "content": self._solution_system(repair_rules)},
                {"role": "user", "content": (
                    "Your previous implementation failed some unit tests.\n"
                    "Use the failure details and the current tests to fix the code so that ALL tests pass 100%.\n"
                    "Critically, do NOT regress: preserve all previously passing behavior.\n\n"
                    f"Failure details:\n{feedback}\n\n"
                    "Current tests.py:\n```python\n" + tests_snapshot + "\n```\n"
                    "Current main.py:\n```python\n" + best_src + "\n```\n"
                    "Return only the complete main.py file in the specified code block format."
                )},
            ]
            try:
                resp = self._call_llm(repair_messages, run_id, r, 300)
                blocks = re.findall(r"```python\s*\n#\s*main\.py\n([\s\S]*?)\n```", resp, re.DOTALL)
                if not blocks:
                    # Fallback: accept first python block without header
                    blocks = re.findall(r"```python\s*\n([\s\S]*?)\n```", resp, re.DOTALL)
                if blocks and blocks[0].strip():
                    repaired_src = blocks[0].strip()
                    repaired_patch = self._build_single_file_patch("main.py", repaired_src)
                    print("[V6][SOLVE] Repair attempt produced a new candidate; validating with local tests")
                    rok, _, rdetails = run_tests_locally(repaired_patch)
                    rtotal = rdetails.get("tests_total", 0) if isinstance(rdetails, dict) else 0
                    rpassed = rdetails.get("tests_passed", 0) if isinstance(rdetails, dict) else 0
                    # Accept only if not a regression (rpassed >= passed)
                    if rpassed >= passed:
                        best_patch = repaired_patch
                        best_src = repaired_src
                        best_passed = rpassed
                        best_details = rdetails if isinstance(rdetails, dict) else {}
                        if rok and rtotal == rpassed:
                            print("[V6][SOLVE] Repair loop: found 100% passing patch")
                            return best_patch
                    else:
                        # Attempt merge if there is positive progress (some newly fixed tests)
                        base_passed_set, _ = self._extract_pass_fail_sets(details if isinstance(details, dict) else {})
                        cand_passed_set, _ = self._extract_pass_fail_sets(rdetails if isinstance(rdetails, dict) else {})
                        newly_fixed = sorted(list(cand_passed_set - base_passed_set))
                        regressed = sorted(list(base_passed_set - cand_passed_set))
                        if newly_fixed:
                            print(f"[V6][SOLVE] Regression with positive progress: newly_fixed={newly_fixed}, regressed={regressed}")
                            merge_rules = (
                                "You MUST return complete Python file implementations for each file that needs changes.\n"
                                "Format your response as:\n"
                                "```python\n# main.py\n[complete Python code for main.py]\n```\n"
                                "Return ONLY the complete file contents.\n"
                                "Hard requirement: preserve all behaviors that kept baseline tests passing; "
                                "incorporate only the minimal changes from the candidate necessary to also pass the newly fixed tests; "
                                "do not reintroduce any regressions."
                            )
                            merge_messages = [
                                {"role": "system", "content": self._solution_system(merge_rules)},
                                {"role": "user", "content": (
                                    "Combine the strengths of these two versions of main.py:\n"
                                    f"- Newly fixed tests (by candidate): {newly_fixed}\n"
                                    f"- Regressed tests (vs baseline): {regressed}\n\n"
                                    "Baseline main.py (passes the regressed tests):\n"
                                    "```python\n# main.py\n" + best_src + "\n```\n\n"
                                    "Candidate main.py (fixes the newly fixed tests):\n"
                                    "```python\n# main.py\n" + repaired_src + "\n```\n\n"
                                    "Produce a merged main.py that preserves all baseline passing behavior and also passes the newly fixed tests. "
                                    "Avoid unnecessary refactors. Return only the complete main.py file in the specified code block."
                                )},
                            ]
                            try:
                                merge_resp = self._call_llm(merge_messages, run_id, r, 300)
                                mblocks = re.findall(r"```python\s*\n#\s*main\.py\n([\s\S]*?)\n```", merge_resp, re.DOTALL)
                                if mblocks and mblocks[0].strip():
                                    merged_src = mblocks[0].strip()
                                    merged_patch = self._build_single_file_patch("main.py", merged_src)
                                    mok, _, mdetails = run_tests_locally(merged_patch)
                                    mtotal = mdetails.get("tests_total", 0) if isinstance(mdetails, dict) else 0
                                    mpassed = mdetails.get("tests_passed", 0) if isinstance(mdetails, dict) else 0
                                    m_passed_set, _ = self._extract_pass_fail_sets(mdetails if isinstance(mdetails, dict) else {})
                                    if mpassed >= passed and base_passed_set.issubset(m_passed_set):
                                        print("[V6][SOLVE] Accepted merged candidate (preserved baseline and added fixes)")
                                        best_patch = merged_patch
                                        best_src = merged_src
                                        best_passed = mpassed
                                        best_details = mdetails if isinstance(mdetails, dict) else {}
                                        if mok and mtotal == mpassed:
                                            print("[V6][SOLVE] Merge produced 100% passing patch")
                                            return best_patch
                                    else:
                                        print("[V6][SOLVE] Merge did not preserve baseline or did not improve; ignoring")
                            except Exception as e:
                                print(f"[V6][SOLVE] Merge attempt error: {e}")
                        else:
                            print(f"[V6][SOLVE] Rejected repair due to regression (new {rpassed} < baseline {passed}) with no newly fixed tests")
                else:
                    print("[V6][SOLVE] Repair attempt returned no main.py block")
            except Exception as e:
                print(f"[V6][SOLVE] Repair attempt error: {e}")
                continue

        # Final validation of best-so-far
        if best_patch:
            ok, _, details = run_tests_locally(best_patch)
            total = details.get("tests_total", 0) if isinstance(details, dict) else 0
            passed = details.get("tests_passed", 0) if isinstance(details, dict) else 0
            if ok and total == passed:
                return best_patch
        print("[V6][SOLVE] No fully passing patch found after repairs")
        return ""

    # ---------- Patch helpers ----------
    def _build_single_file_patch(self, filename: str, new_content: str) -> str:
        # Build a git-compatible unified diff replacing entire file content
        try:
            with open(filename, "r", encoding="utf-8") as f:
                old = f.read()
        except Exception:
            old = ""
        old_lines = old.splitlines()
        new_lines = new_content.splitlines()
        # Construct full-file replace hunk
        header = [
            f"diff --git a/{filename} b/{filename}",
            "index 0000000..1111111 100644",
            f"--- a/{filename}",
            f"+++ b/{filename}",
            f"@@ -1,{max(1, len(old_lines))} +1,{max(1, len(new_lines))} @@",
        ]
        body = []
        if old_lines:
            body.extend(["-" + ln for ln in old_lines])
        else:
            # ensure at least one context line for empty old file
            body.append("-")
        if new_lines:
            body.extend(["+" + ln for ln in new_lines])
        else:
            body.append("+")
        return "\n".join(header + body) + "\n"


def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False) -> str:
    run_id = (input_dict or {}).get("run_id", os.getenv("RUN_ID", str(uuid.uuid4())))
    print(f"[V6][MAIN] run_id={run_id} repo_dir={repo_dir}")
    if repo_dir and os.path.exists(repo_dir):
        try:
            os.chdir(repo_dir)
            print(f"[V6][MAIN] Changed cwd to {os.getcwd()}")
        except Exception as e:
            print(f"[V6][MAIN] Failed to chdir to repo: {e}")
    ensure_git_initialized()
    try:
        reorder_models((input_dict or {}).get("problem_statement", ""))
    except Exception as e:
        print(f"[V6][MAIN] reorder_models failed: {e}")
    problem = (input_dict or {}).get("problem_statement", "")
    problem_category = (input_dict or {}).get("problem_category", None)
    if problem_category not in ("spec_only", "tests_available"):
        problem_category = "tests_available" if os.path.exists("tests.py") else "spec_only"
    print(f"[V6][MAIN] problem_len={len(problem)} mode={problem_category}")
    agent = AgentV6(problem, mode=problem_category, top_k=30)
    # Run generate tests => test => fix loop three times, stopping only after 3 successful cycles
    final_patch = ""
    success_cycles = 0
    for cycle in range(3):
        print(f"[V6][MAIN] Starting solve cycle {cycle+1}/3")
        patch = agent.propose_patch(f"{run_id}-cycle{cycle+1}")
        print(f"[V6][MAIN] patch_len={len(patch) if patch else 0}")
        if not patch:
            print(f"[V6][MAIN] Cycle {cycle+1} did not produce a fully passing patch; stopping")
            break
        final_patch = patch
        # Apply the patch to the working tree so the next cycle starts from the updated code
        try:
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".diff") as tf:
                tf.write(patch)
                temp_path = tf.name
            # Use git apply; repository is initialized earlier
            print(f"[V6][MAIN] Applying patch for cycle {cycle+1}")
            applied = False
            last_err = ""
            for cmd in (["git", "apply", "--whitespace=nowarn", "-p1", temp_path],
                        ["git", "apply", "--whitespace=nowarn", temp_path]):
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if res.returncode == 0:
                    applied = True
                    break
                last_err = res.stderr.strip()
            if not applied:
                print(f"[V6][MAIN] Failed to apply patch for cycle {cycle+1}: {last_err}")
                break
            print(f"[V6][MAIN] Patch applied for cycle {cycle+1}")
            success_cycles += 1
        except Exception as e:
            print(f"[V6][MAIN] Exception applying patch for cycle {cycle+1}: {e}")
            break

    print(f"[V6][MAIN] Completed {success_cycles} successful cycle(s)")
    return final_patch or ""


