# New minimal Cursor-like agent (from scratch)


from __future__ import annotations


import os
import re
import json
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple


import requests



# --------------------------------------------------------------------------------
# Configuration and whitelisted models (must match current repo constraints)
# --------------------------------------------------------------------------------
DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")


GLM_MODEL_NAME = "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AGENT_MODELS = [GLM_MODEL_NAME, QWEN_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME]
AGENT_ID = "new-agent/1.0"



# --------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------
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
    # Strip code fences and language tags if present
    t = patch.strip()
    if t.startswith("```") and t.endswith("```"):
        t = t.strip("`")
        t = re.sub(r"^\w+\n", "", t)
    # Keep only unified-diff friendly lines
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
    
    # Try to fix missing headers by adding them if we have content but no headers
    has_git_header = any("diff --git" in ln for ln in cleaned)
    has_at_sign = any(ln.startswith("@@") for ln in cleaned)
    has_file_headers = any(ln.startswith(("--- ", "+++ ")) for ln in cleaned)
    
    # If we have @@ but missing other headers, try to reconstruct
    if has_at_sign and not has_git_header:
        # Look for file references in the content
        file_paths = []
        for ln in cleaned:
            if ln.startswith(("--- ", "+++ ")):
                file_paths.append(ln.split()[-1])
        if not file_paths:
            for ln in cleaned:
                if "/" in ln and ("main.py" in ln or "tests.py" in ln):
                    file_paths.append(ln.strip("/").split()[-1] if ln.strip().startswith("/") else ln.split()[-1])
        
        # Insert headers if we found file paths
        if file_paths:
            new_lines = ["diff --git a/" + file_paths[0] + " b/" + file_paths[0],
                        "index 0000000..1111111 100644"]
            if not has_file_headers and len(file_paths) >= 2:
                new_lines.append(f"--- a/{file_paths[0]}")
                new_lines.append(f"+++ b/{file_paths[1] if len(file_paths) > 1 else file_paths[0]}")
            cleaned = new_lines + cleaned
    
    # Validate patch has proper structure
    if not any("diff --git" in ln for ln in cleaned):
        print("[SANITIZE] Missing 'diff --git' header")
        return ""
    
    if not any(ln.startswith("@@") for ln in cleaned):
        print("[SANITIZE] Missing '@@' hunk markers")
        return ""
    
    # Check that the patch appears complete (has file headers)
    has_file_headers = any(ln.startswith("--- ") or ln.startswith("+++ ") for ln in cleaned)
    if not has_file_headers:
        print("[SANITIZE] Missing file headers (--- / +++ )")
        # Try to add them anyway
        file_refs = []
        for ln in cleaned:
            if "diff --git" in ln:
                parts = ln.split()
                if len(parts) >= 3 and parts[2].startswith("b/"):
                    file_name = parts[2][2:]  # remove "b/" prefix
                    if not file_refs:
                        cleaned.insert(1, f"index 0000000..1111111 100644")
                        cleaned.insert(2, f"--- a/{file_name}")
                        cleaned.insert(3, f"+++ b/{file_name}")
                    break
    
    # Ensure a final newline
    result = ("\n".join(cleaned) + "\n")
    
    # Final sanity check: make sure it's not suspiciously short
    if len(result.split('\n')) < 5:
        print("[SANITIZE] Patch seems too short")
        return ""
    
    return result



def extract_code_blocks(response: str) -> Dict[str, str]:
    """Extract file:code mappings from LLM response"""
    
    file_implementations = {}
    
    print(f"[EXTRACT] Extracting code blocks from response (length: {len(response)})")
    
    # Look for code blocks with file names in comments
    # Pattern: ```python\n# filename.py\n[code]\n```
    code_block_pattern = r'```python\s*\n#\s*([^\n]+\.py)\s*\n(.*?)\n```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    
    print(f"[EXTRACT] Found {len(matches)} code block matches")
    
    for filename, code in matches:
        # Clean up the code (remove extra whitespace)
        clean_code = code.strip()
        if clean_code:
            file_implementations[filename] = clean_code
            print(f"[EXTRACT] Extracted {filename}: {len(clean_code)} characters")
    
    # Also look for standalone file sections without code fences
    # Pattern: # filename.py\n[code]
    file_section_pattern = r'#\s*([^\n]+\.py)\s*\n(.*?)(?=\n#\s*[^\n]+\.py|\Z)'
    matches = re.findall(file_section_pattern, response, re.DOTALL)
    
    print(f"[EXTRACT] Found {len(matches)} file section matches")
    
    for filename, code in matches:
        clean_code = code.strip()
        if clean_code and filename not in file_implementations:
            file_implementations[filename] = clean_code
            print(f"[EXTRACT] Extracted {filename}: {len(clean_code)} characters")
    
    # If no code blocks found, try to extract from the entire response
    if not file_implementations:
        print("[EXTRACT] No code blocks found, trying to extract from entire response")
        # Look for any Python code in the response
        python_code_pattern = r'```python\s*\n(.*?)\n```'
        matches = re.findall(python_code_pattern, response, re.DOTALL)
        
        for i, code in enumerate(matches):
            clean_code = code.strip()
            if clean_code:
                filename = f"main.py" if i == 0 else f"file_{i}.py"
                file_implementations[filename] = clean_code
                print(f"[EXTRACT] Extracted {filename}: {len(clean_code)} characters")
    
    print(f"[EXTRACT] Total extracted files: {len(file_implementations)}")
    return file_implementations



def generate_diff_from_implementations(file_implementations: Dict[str, str]) -> str:
    """Generate unified diff from complete file implementations"""
    import tempfile
    
    if not file_implementations:
        print("[DIFF_GEN] No file implementations provided")
        return ""
    
    print(f"[DIFF_GEN] Generating diff for {len(file_implementations)} files")
    
    try:
        # Create a temporary directory for git operations
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"[DIFF_GEN] Using temp directory: {temp_dir}")
            
            # Initialize git repo
            result = subprocess.run(["git", "init"], cwd=temp_dir, capture_output=True, text=True, check=True)
            print(f"[DIFF_GEN] Git init successful")
            
            # Configure git user identity
            subprocess.run(["git", "config", "user.email", "agent@example.com"], cwd=temp_dir, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "Agent"], cwd=temp_dir, capture_output=True, text=True)
            print(f"[DIFF_GEN] Git user configured")
            
            # Copy current files to temp directory
            copied_files = 0
            for root, dirs, files in os.walk("."):
                # Skip hidden directories and __pycache__
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
                            with open(dst_path, 'w', encoding='utf-8') as g:
                                g.write(content)
                            copied_files += 1
                        except Exception as e:
                            print(f"[DIFF_GEN] Warning: Could not copy {src_path}: {e}")
            
            print(f"[DIFF_GEN] Copied {copied_files} files to temp directory")
            
            # Add all files to git
            result = subprocess.run(["git", "add", "."], cwd=temp_dir, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[DIFF_GEN] Git add failed: {result.stderr}")
                return ""
            
            result = subprocess.run(["git", "commit", "-m", "original"], cwd=temp_dir, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[DIFF_GEN] Git commit failed: {result.stderr}")
                return ""
            
            print(f"[DIFF_GEN] Git commit successful")
            
            # Apply new implementations
            for filename, new_content in file_implementations.items():
                file_path = os.path.join(temp_dir, filename)
                print(f"[DIFF_GEN] Writing {filename} ({len(new_content)} chars)")
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Write new content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            
            # Generate diff
            result = subprocess.run(
                ["git", "diff", "--no-color", "--unified=3"],
                cwd=temp_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"[DIFF_GEN] Git diff failed: {result.stderr}")
                return ""
            
            diff_output = result.stdout or ""
            print(f"[DIFF_GEN] Generated diff: {len(diff_output)} characters")
            
            # Validate the diff
            if diff_output and "diff --git" in diff_output:
                print(f"[DIFF_GEN] Diff validation passed")
                return diff_output
            else:
                print(f"[DIFF_GEN] Diff validation failed - no valid diff generated")
                return ""
            
    except Exception as e:
        print(f"[DIFF_GEN] Error generating diff: {e}")
        import traceback
        print(f"[DIFF_GEN] Traceback: {traceback.format_exc()}")
        return ""



def dry_run_patch(patch_text: str) -> Tuple[bool, str]:
    """Lightweight applicability check only (does not apply the patch)."""
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
    """Apply the patch, run Python syntax checks on modified/untracked files, then hard reset.


    Always restores the working tree to pre-apply state.
    """
    if not patch_text.strip():
        return False, "empty patch"
    try:
        # Clean state
        subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
        with open(".temp_patch", "w", encoding="utf-8") as f:
            f.write(patch_text)
        res_apply = subprocess.run(["git", "apply", ".temp_patch"], capture_output=True, text=True, timeout=45)
        if res_apply.returncode != 0:
            return False, f"patch apply failed: {res_apply.stderr.strip()}"


        # Gather modified/untracked files
        mod = subprocess.run(["git", "ls-files", "-m"], capture_output=True, text=True, timeout=20)
        untracked = subprocess.run(["git", "ls-files", "-o", "--exclude-standard"], capture_output=True, text=True, timeout=20)
        changed = set((mod.stdout or "").splitlines()) | set((untracked.stdout or "").splitlines())
        py_changed = [p for p in changed if p.endswith(".py")]


        # Syntax check
        for path in py_changed:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    src = f.read()
                compile(src, path, "exec")
            except SyntaxError as se:
                return False, f"SyntaxError in {path}:{se.lineno}: {se.msg}"
            except Exception as e:
                # Non-syntax read/IO errors: surface as failure
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



# --------------------------------------------------------------------------------
# Core minimal agent
# --------------------------------------------------------------------------------
class MinimalCursorAgent:
    OUTPUT_RULES = (
        "You MUST return complete Python file implementations for each file that needs changes.\n"
        "Format your response as:\n"
        "```python\n# main.py\n[complete Python code for main.py]\n```\n"
        "```python\n# other_file.py\n[complete Python code for other_file.py]\n```\n"
        "Return ONLY the complete file contents, not diffs or patches."
    )


    def __init__(self, problem_statement: str, top_k: int = 20):
        self.problem_statement = problem_statement or ""
        self.top_k = top_k


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
        # Always include main.py and tests.py first if present
        for must in ["main.py", "tests.py"]:
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
            body = content[:2000]  # Reduced to avoid timeouts
            parts.append(f"### {fp}\n```python\n{body}\n```")
        return "\n\n".join(parts)


    def _build_messages(self, repo_summary: str) -> List[Dict[str, str]]:
        sys = (
            "You are an autonomous senior software engineer.\n"
            "Analyze the problem and repository summary.\n"
            "Implement or fix only the code under test so that all tests in tests.py pass.\n"
            "Do not modify tests.py. Do not return placeholders (no 'pass' or 'return None').\n"
            "Pay special attention to exact error messages - they must match test expectations exactly.\n"
            "Read the test assertions carefully to understand expected behavior and error messages.\n"
            "For error messages, copy the exact text from test assertions including punctuation.\n"
            "CRITICAL: When tests call methods or access attributes, ensure they exist in your implementation.\n"
            "Do not leave incomplete implementations - implement ALL methods referenced in tests.\n"
            "CRITICAL: Pay attention to return types - if tests expect a list, return a list; if they expect a string, return a string.\n"
            "CRITICAL: Look at the test assertions to understand the exact expected return format.\n"
            "CRITICAL: If tests use self.assertEqual(function(), [item1, item2]), return a LIST.\n"
            "CRITICAL: If tests use self.assertEqual(function(), \"string\"), return a STRING.\n"
            "CRITICAL: Never return a string when tests expect a list, or vice versa.\n"
            "CRITICAL: If tests import specific names from your module, ensure those names are defined and exported.\n"
            "CRITICAL: Check what the tests are trying to import and make sure those names exist in your implementation.\n"
            "CRITICAL: For complex problems, break them down into smaller, manageable functions.\n"
            "CRITICAL: Always provide complete, working implementations - no partial solutions.\n"
            "IMPORTANT: Make sure your code handles edge cases and error conditions properly.\n"
            "IMPORTANT: If the problem involves algorithms, implement them correctly with proper logic.\n"
            "IMPORTANT: If the problem involves data structures, use appropriate Python data types.\n"
            "IMPORTANT: Always return the correct data type that the tests expect.\n"
            "Return complete Python code for each file that needs changes.\n"
            + self.OUTPUT_RULES
        )
        test_reqs = self._analyze_test_requirements()
        print(f"[AGENT] Test requirements analysis: {test_reqs}")
        user = (
            f"Problem Statement:\n{self.problem_statement}\n\n"
            f"Repository summary (top files):\n\n{repo_summary}\n\n"
            f"{test_reqs}\n\n"
            "Return complete Python implementations for each file that needs changes."
        )
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ]


    def _analyze_test_requirements(self) -> str:
        """Analyze tests.py to extract specific requirements like error messages and return types"""
        try:
            with open("tests.py", "r", encoding="utf-8") as f:
                content = f.read()
            
            requirements = []
            
            # Look for specific error message assertions
            error_messages = re.findall(r'self\.assertEqual\(err\.exception\.args\[0\], "([^"]+)"\)', content)
            if error_messages:
                requirements.append("IMPORTANT: The tests expect these exact error messages:")
                requirements.extend(f"- '{msg}'" for msg in error_messages)
            
            # Look for import statements to understand what the tests expect
            import_lines = re.findall(r'from\s+\w+\s+import\s+([^,\n]+)', content)
            if import_lines:
                requirements.append("CRITICAL: The tests are trying to import these names from your module:")
                for imports in import_lines:
                    names = [name.strip() for name in imports.split(',')]
                    requirements.extend(f"- {name}" for name in names)
                requirements.append("Make sure ALL these names are defined in your implementation!")
            
            # Look for return type patterns - more comprehensive analysis
            if 'assertEqual' in content:
                # Simple check: look for list patterns in assertEqual
                has_list_assertions = False
                has_string_assertions = False
                
                # Check for list patterns - look for [ in assertEqual context
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'self.assertEqual' in line:
                        # Check if this assertEqual has a list
                        context = ' '.join(lines[max(0, i-2):i+3])  # Get context around the line
                        if '[' in context and ']' in context:
                            has_list_assertions = True
                            break
                
                # Check for string patterns
                for i, line in enumerate(lines):
                    if 'self.assertEqual' in line:
                        context = ' '.join(lines[max(0, i-2):i+3])
                        if '"' in context and '[' not in context:
                            has_string_assertions = True
                            break
                
                if has_list_assertions:
                    requirements.append("CRITICAL: The tests expect the function to return a LIST, not a string!")
                    requirements.append("Look for patterns like self.assertEqual(function(), [item1, item2, ...])")
                    requirements.append("Your function should return a list of strings, not a single string!")
                    requirements.append("Example: return [\"item1\", \"item2\"] not return \"item1\\nitem2\"")
                elif has_string_assertions:
                    requirements.append("CRITICAL: The tests expect the function to return a STRING, not a list!")
                    requirements.append("Your function should return a single string, not a list!")
            
            # Look for specific test patterns that indicate expected behavior
            if 'assertRaises' in content:
                requirements.append("IMPORTANT: The tests expect certain exceptions to be raised!")
                requirements.append("Make sure your implementation raises the correct exceptions when expected.")
            
            # Look for specific function calls in tests
            function_calls = re.findall(r'(\w+)\s*\(', content)
            if function_calls:
                unique_calls = list(set(function_calls))
                if len(unique_calls) > 1:  # More than just 'self'
                    requirements.append(f"IMPORTANT: The tests call these functions: {', '.join(unique_calls[:5])}")
                    requirements.append("Make sure all these functions are implemented in your code!")
            
            # Look for specific data types or patterns
            if 'list(' in content or '[]' in content:
                requirements.append("IMPORTANT: The tests work with lists - make sure your functions handle lists correctly!")
            
            if 'str(' in content or 'string' in content.lower():
                requirements.append("IMPORTANT: The tests work with strings - make sure your functions handle strings correctly!")
            
            if requirements:
                return "\n\n" + "\n".join(requirements)
        except Exception as e:
            print(f"[ANALYZE] Error analyzing test requirements: {e}")
        return ""


    def _call_llm(self, messages: List[Dict[str, str]], run_id: str, attempt: int) -> str:
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
        headers = {"Content-Type": "application/json"}
        model = AGENT_MODELS[attempt % len(AGENT_MODELS)]
        body = {
            "run_id": run_id,
            "messages": messages,
            "temperature": 0.0,
            "agent_id": AGENT_ID,
            "model": model,
        }
        max_retries = 3  # Increased from 2 to 3
        for retry in range(max_retries):
            try:
                print(f"[AGENT] LLM attempt {attempt + 1}/4 using model: {model} (retry {retry + 1}/{max_retries})")
                # Increase timeout for complex problems
                timeout = 180 if attempt > 1 else 120
                resp = requests.post(url, json=body, timeout=timeout, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and data.get("choices") and data["choices"][0].get("message"):
                    content = data["choices"][0]["message"]["content"] or ""
                    if content and len(content.strip()) > 50:  # Ensure meaningful response
                        print(f"[AGENT] LLM response received: {len(content)} characters")
                        return content
                    else:
                        print(f"[AGENT] Empty or too short response from LLM, retrying...")
                if isinstance(data, str):
                    if data and len(data.strip()) > 50:
                        print(f"[AGENT] LLM response received: {len(data)} characters")
                        return data
                return json.dumps(data)
            except requests.exceptions.Timeout:
                print(f"[AGENT] Timeout on attempt {attempt + 1}, retry {retry + 1}/{max_retries}")
                if retry < max_retries - 1:
                    time.sleep(3 ** retry)  # Longer exponential backoff
            except requests.exceptions.ConnectionError:
                print(f"[AGENT] Connection error on attempt {attempt + 1}, retry {retry + 1}/{max_retries}")
                if retry < max_retries - 1:
                    time.sleep(5 + retry * 2)  # Longer wait for connection issues
            except Exception as e:
                print(f"[AGENT] Error on attempt {attempt + 1}, retry {retry + 1}: {e}")
                if retry < max_retries - 1:
                    time.sleep(2 + retry)
        
        print(f"[AGENT] All retries failed for attempt {attempt + 1}")
        return ""


    def propose_patch(self, run_id: str) -> str:
        print(f"[AGENT] Starting code-first patch generation for problem: {self.problem_statement[:100]}...")
        files = self._collect_python_files(".")
        top = self._score_files(files)
        print(f"[AGENT] Found {len(files)} Python files, selected top {len(top)} for analysis")
        summary = self._build_repo_summary(top)
        messages = self._build_messages(summary)


        raw = ""
        for attempt in range(4):
            print(f"[AGENT] LLM attempt {attempt + 1}/4 using model: {AGENT_MODELS[attempt % len(AGENT_MODELS)]}")
            raw = self._call_llm(messages, run_id, attempt)
            if raw:
                break


        if not raw:
            print("[AGENT] No response from LLM")
            return ""
        
        if len(raw.strip()) < 100:
            print(f"[AGENT] Response too short ({len(raw)} chars), likely incomplete")
            return ""
        
        # Check for common failure patterns in LLM response
        if "I cannot" in raw or "I'm unable" in raw or "I don't know" in raw:
            print("[AGENT] LLM response indicates inability to solve problem")
            return ""
        
        # Extract code implementations from LLM response
        file_implementations = extract_code_blocks(raw)
        print(f"[AGENT] Extracted {len(file_implementations)} file implementations: {list(file_implementations.keys())}")
        
        if not file_implementations:
            print("[AGENT] No valid code implementations found in LLM response")
            print(f"[AGENT] Raw response preview: {raw[:500]}...")
            return ""
        
        # Validate syntax of each file implementation
        valid_implementations = {}
        
        for filename, code in file_implementations.items():
            try:
                compile(code, filename, 'exec')
                print(f"[AGENT] Syntax validation passed for {filename}")
                valid_implementations[filename] = code
            except SyntaxError as e:
                print(f"[AGENT] Syntax error in {filename}: {e}")
        
        if not valid_implementations:
            print("[AGENT] No valid syntax implementations found")
            return ""
        
        # Generate unified diff from implementations
        patch = generate_diff_from_implementations(valid_implementations)
        print(f"[AGENT] Generated patch length: {len(patch)} characters")
        
        if not patch:
            print("[AGENT] Failed to generate diff from implementations")
            return ""
        
        # Validate the patch format
        if not patch.startswith("diff --git"):
            print("[AGENT] Generated patch does not start with 'diff --git'")
            return ""
        
        # Check patch validity
        ok, err = dry_run_patch(patch)
        if ok:
            ok2, err2 = apply_and_syntax_check(patch)
            if ok2:
                print(f"[AGENT] Patch validation passed!")
                return patch
            err = err2 or err
        
        print(f"[AGENT] Initial patch validation failed: {err}")
        
        # Iterative refinement loop - try up to 3 times with syntax error feedback
        best_patch = patch
        current_messages = messages.copy()
        
        for refinement_round in range(3):
            if refinement_round == 0:
                # First refinement attempt
                refinement_msg = (
                    f"Your previous code implementations had issues: {err or 'unknown'}\n"
                    "Please return corrected complete Python implementations for each file that needs changes.\n"
                    "Make sure the code has valid Python syntax and implements all required functionality.\n"
                    "Pay special attention to:\n"
                    "- Correct return types (list vs string)\n"
                    "- Proper error handling\n"
                    "- Complete implementation of all required functions\n"
                    "- Edge case handling"
                )
            else:
                # Additional refinement attempts
                refinement_msg = (
                    f"Your previous code implementations still had issues: {err or 'unknown'}\n"
                    "Please return corrected complete Python implementations for each file that needs changes.\n"
                    "Focus on fixing the specific errors mentioned above.\n"
                    "Make sure to:\n"
                    "- Fix all syntax errors\n"
                    "- Implement all missing functions\n"
                    "- Handle all edge cases\n"
                    "- Return the correct data types"
                )
            
            current_messages.append({"role": "assistant", "content": raw})
            current_messages.append({"role": "user", "content": refinement_msg})
            
            raw_refinement = self._call_llm(current_messages, run_id, 4 + refinement_round)
            if not raw_refinement:
                print(f"[AGENT] Round {refinement_round + 1}: LLM returned empty, stopping refinement")
                break
            
            # Extract new implementations
            new_implementations = extract_code_blocks(raw_refinement)
            print(f"[AGENT] Round {refinement_round + 1}: Extracted {len(new_implementations)} file implementations")
            
            if not new_implementations:
                print(f"[AGENT] Round {refinement_round + 1}: No valid implementations found")
                break
            
            # Validate syntax of new implementations
            valid_new_implementations = {}
            
            for filename, code in new_implementations.items():
                try:
                    compile(code, filename, 'exec')
                    print(f"[AGENT] Round {refinement_round + 1}: Syntax validation passed for {filename}")
                    valid_new_implementations[filename] = code
                except SyntaxError as e:
                    print(f"[AGENT] Round {refinement_round + 1}: Syntax error in {filename}: {e}")
            
            if not valid_new_implementations:
                print(f"[AGENT] Round {refinement_round + 1}: No valid syntax implementations found")
                break
            
            # Generate new patch
            new_patch = generate_diff_from_implementations(valid_new_implementations)
            print(f"[AGENT] Round {refinement_round + 1}: Generated patch length: {len(new_patch)} characters")
            
            if not new_patch:
                print(f"[AGENT] Round {refinement_round + 1}: Failed to generate diff")
                break
            
            # Validate the patch format
            if not new_patch.startswith("diff --git"):
                print(f"[AGENT] Round {refinement_round + 1}: Generated patch does not start with 'diff --git'")
                break
            
            # Check new patch validity
            ok, err = dry_run_patch(new_patch)
            if ok:
                ok2, err2 = apply_and_syntax_check(new_patch)
                if ok2:
                    print(f"[AGENT] Round {refinement_round + 1}: Patch validation passed!")
                    return new_patch
                err = err2 or err
            
            print(f"[AGENT] Round {refinement_round + 1}: Patch validation failed: {err}")
            best_patch = new_patch
            raw = raw_refinement
        
        # Return best patch we got (or empty if none)
        if best_patch:
            print("[AGENT] Returning best patch despite validation failures")
            return best_patch
        
        print("[AGENT] No valid patch generated")
        return ""



# --------------------------------------------------------------------------------
# Entry point – must return a string unified diff (to satisfy the runner)
# --------------------------------------------------------------------------------
def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _generate_diff_for_main_py(new_content: str) -> str:
    try:
        # Clean any prior changes
        subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
        original = _read("main.py")
        with open("main.py", "w", encoding="utf-8") as f:
            f.write(new_content)
        subprocess.run(["git", "add", "main.py"], capture_output=True, text=True, timeout=20)
        diff = subprocess.run(["git", "diff", "--cached", "--no-color", "--unified=3"], capture_output=True, text=True, timeout=30)
        patch_text = diff.stdout or ""
        return patch_text
    except Exception:
        return ""
    finally:
        try:
            subprocess.run(["git", "reset", "--hard"], capture_output=True, text=True, timeout=30)
        except Exception:
            pass



def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False) -> str:
    run_id = (input_dict or {}).get("run_id", os.getenv("RUN_ID", "nocache-1"))


    # Position inside repo if present
    if repo_dir and os.path.exists(repo_dir):
        try:
            os.chdir(repo_dir)
        except Exception:
            pass


    ensure_git_initialized()


    # Prefer a first-choice model based on problem text (rotation preserved per-call)
    try:
        reorder_models((input_dict or {}).get("problem_statement", ""))
    except Exception:
        pass


    problem = (input_dict or {}).get("problem_statement", "")


    agent = MinimalCursorAgent(problem, top_k=30)
    patch = agent.propose_patch(run_id)
    return patch or ""



