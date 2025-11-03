#V3.4

from __future__ import annotations
import ast
import json
import os
# Improvement v1 - 2025-10-24 23:35:24
from functools import partial
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.messages import ToolCallExecutionEvent,FunctionExecutionResult,ToolCallSummaryMessage
import requests
from typing import Union, get_args, get_origin

import urllib.request as _urlreq
from ast import literal_eval
import math
import pytest
from typing import NamedTuple
from collections import Counter

AGENT_ID="m-3.4"




import subprocess
import ast, sys
from typing import Callable, Dict, List, Optional, Set, Tuple
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from json import JSONDecodeError
import re
import inspect
import random
from enum import Enum
import json
import csv
import logging
import concurrent.futures
import threading
from collections import defaultdict
import shutil

import logging



from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import UserMessage
import json
import uuid
import sys
from autogen_agentchat.agents import AssistantAgent
import time
from autogen_core.models import ModelFamily
from autogen_agentchat.ui import Console
from pydantic import BaseModel
from autogen_core.models import SystemMessage
from autogen_core.models import CreateResult
import importlib
httpx=importlib.import_module("httpx")
import logging
import re
from autogen_agentchat import EVENT_LOGGER_NAME, TRACE_LOGGER_NAME
import asyncio
import threading
from queue import Queue


from typing import List, Literal, Optional

## logging setup -------------------------------------------------------->>
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Remove any existing handlers to avoid duplicates or default ones
for h in list(logger.handlers):
    logger.removeHandler(h)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')

# Stream handler (stdout)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# File handler
log_file = "final_agent.log"
file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
sys.stdout = open("agent_flow.log", "w", encoding="utf-8")


#-------------------------------------------------------------------------------------->>

## all global variables here -------------------------------------------------------->>
DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2200"))
MAX_TEST_PATCH_TIMEOUT = int(os.getenv("MAX_STEPS_TEST_PATCH_FIND", "400"))

GLM_MODEL_NAME = "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
#QWEN_MODEL_NAME=GLM_MODEL_NAME
AGENT_MODELS=[GLM_MODEL_NAME,QWEN_MODEL_NAME,KIMI_MODEL_NAME,DEEPSEEK_MODEL_NAME]
RUN_ID="nocache-1"
JSON_LLM_USED=0
JSON_LITERAL_USED=0
DEBUG_MODE=True
MARKDOWN_FAILED=0
TOOL_CALL_FAILED=0
MAX_EMBED_TOKENS=128000
MAX_EMBED_CHARS=MAX_EMBED_TOKENS*4
IS_SOLUTION_APPROVED=False
DISABLE_TEST_FILE_REMOVAL=False
TOO_MANY_SECTIONS_FOUND=0

# Token budgets for different tasks to control quota usage
TOKEN_BUDGETS = {
    "initial_solution": 2000,
    "test_generation": 1500,
    "fix_task": 3000,
    "problem_classification": 1000
}

#--------------------------------------------------------->>

class AgentPool:
    """Pool for reusing agents to reduce quota usage"""
    def __init__(self):
        self._agents = {}
        self._lock = threading.Lock()
    
    def get_agent(self, agent_type: str, model_name: str, system_message: str = None, agent_name: str = None):
        """Get or create an agent from the pool"""
        key = f"{agent_type}_{model_name}_{agent_name or 'default'}"
        
        with self._lock:
            if key not in self._agents:
                self._agents[key] = CustomAssistantAgent(
                    agent_name=agent_name or f"{agent_type}_agent",
                    model_name=model_name,
                    system_message=system_message
                )
            return self._agents[key]
    
    def clear_pool(self):
        """Clear the agent pool"""
        with self._lock:
            self._agents.clear()

# Global agent pool instance
_agent_pool = AgentPool()

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False):
    """Legacy interface wrapper for backwards compatibility."""
    global DEFAULT_PROXY_URL, REPO_DIR, DEFAULT_TIMEOUT, MAX_TEST_PATCH_TIMEOUT, RUN_ID
    RUN_ID = os.getenv("RUN_ID", "nocache-1")
    REPO_DIR = repo_dir
    sys.path.insert(0, repo_dir)

    if os.path.exists(repo_dir):
        os.chdir(repo_dir)

    ensure_git_initialized()

    set_env_for_agent()

    # Check problem type first
    problem_type = asyncio.run(ProblemTypeClassifierAgent.check_problem_type(input_dict.get("problem_statement")))
    
    if problem_type == ProblemTypeClassifierAgent.PROBLEM_TYPE_FIX:
       fix_prb_task=BugFixSolver(input_dict.get("problem_statement")).solve_problem()
       try:
           result=asyncio.run(asyncio.wait_for(fix_prb_task, timeout=2280))
       except asyncio.TimeoutError as e:
           logger.error(f"Timed out after 2280 seconds..")
           result=FixTaskEnhancedToolManager.get_final_git_patch(initial_checkpoint="initial_commit")
           if not DISABLE_TEST_FILE_REMOVAL:
               FixTaskEnhancedToolManager.remove_any_generated_test_files()
       except Exception as e:
           logger.error(f"Error: {e}")
           result=FixTaskEnhancedToolManager.get_final_git_patch(initial_checkpoint="initial_commit")
           if not DISABLE_TEST_FILE_REMOVAL:
               FixTaskEnhancedToolManager.remove_any_generated_test_files()
    else:
        # Use traditional approach for CREATE tasks
        solve_prb_task=CreateProblemSolver(input_dict.get("problem_statement")).solve_problem()
        try:
            result=asyncio.run(asyncio.wait_for(solve_prb_task, timeout=2280))
        except asyncio.TimeoutError as e:
            logger.error(f"Timed out after 60 seconds..")
            result=FixTaskEnhancedToolManager.get_final_git_patch()
            
        except Exception as e:
            logger.error(f"Error: {e}")
            result=FixTaskEnhancedToolManager.get_final_git_patch()
            
        

    if not DISABLE_TEST_FILE_REMOVAL:
        os.system("git reset --hard")
    logger.info("patch returned: {}".format(result))
    logger.info("JSON_LLM_USED: {}".format(JSON_LLM_USED))
    logger.info("JSON_LITERAL_USED: {}".format(JSON_LITERAL_USED))
    logger.info("MARKDOWN_FAILED: {}".format(MARKDOWN_FAILED))
    logger.info("TOO_MANY_SECTIONS_FOUND: {}".format(TOO_MANY_SECTIONS_FOUND))
    return result
    
 

class CreateProblemSolver:
    
    SYSTEM_PROMPT_INITIAL_SOLUTION_EVAL=textwrap.dedent("""
    You are an expert Python developer. You will be given a problem statement and a python solution. You need to evaluate if the solution is correct or not as per the problem statement.
    
    WorkFlow:-
    - **Plan:** After understanding the problem statement, create a initial list of all the requirements mentioned in problem statement that you need to evaluate.
    - **Evaluate:** Begin evaluating the solution for each of those cases. Create test cases to confirm if the solution is correct.
    - **Adapt:** As you discover new information or encounter obstacles, update your plan.
    - **Verify (Tests):** Check test_cases.txt file for additional scenerios you can test.
    - **Comprehensive Testing:** Think about all the edge cases. Ensure the solution handles all of them. Run comprehensive test to ensure solution fully satisfies all the requirements.
    - **Finish:** Call finish once the solution fully satisfies all the requirements.


    Tool Usage:-
    - Use run_code to create and run unittests.
    - Use apply_code_edit to fix the solution if it fails.
    - Use apply_code_edit to fix the test case if they are not as per the problem statement.
    - Use finish to finish the task.
    
    
    Rules:-
    1. Test code must always import functionality from the repository—never duplicate or reimplement the code within the test itself.
    2. Use verbosity level 2 while running the tests to ensure you see the full output.
    3. If run_code tool throws syntax error, check if last assistant response was truncated. If yes, then skip last couple of test cases and try again.
    4. Must ensure you have tested **ALL scenarios** listed in test_cases.txt file. Even if some  scenarios are not mentioned in problem statement, you must test them.
    
    Here are the tools you have access to:-
    {tools_docs}
    
    You must follow the following response format.
    {format_prompt}
    """
    )
    
    INSTANCE_PROMPT_INITIAL_SOLUTION_EVAL=textwrap.dedent("""
    Problem Statement:
    {problem_statement}
    
    Key Python Files:
    {initial_solution}
    
    """
    )
    
    SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Python developer. Your task is to generate a complete, working Python solution for the given problem statement.

    Strict Requirements:
    1. Output the full content of Python files along with their file names.
    2. Do not include explanations, comments, or markdown formatting.
    3. Use only standard Python (no external libraries).
    4. Implement all required classes and functions exactly with the same names as in the initial code stub.
    5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
    6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
    7. The solution must be executable as-is with no placeholders or TODOs.
    
    """
    )
    
    INSTANCE_PROMPT=textwrap.dedent("""Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\n\nGenerate the complete and correct implementation in python files.""")
    
    RESPONSE_FORMAT="""Return only the final python files code.

    Response Examples:
    ```python
    a.py
    {content}

    b.py
    {content}
    ```"""
    
    RESPONSE_FORMAT_JSON="""Return only the final python files code in JSON format.
    Response Examples:
    [{"file_name":"a.py","code":"contents of a.py"},{"file_name":"b.py","content":"contents of b.py"}]
    """
    
    RESPONSE_FORMAT_SOLUTION_EVAL="""
    Your response must not contain multiple THOUGHT or TOOL_CALL sections. You must respond in the following format. You must not add anything before THOUGHT section.
    ======THOUGHT
    <<your detailed thought process>>
    ======TOOL_CALL
    {"name":"<tool_name>","arguments":{...}}
    """
    RESPONSE_FORMAT_SOLUTION_EVAL_2="""
    Your response must not contain multiple TOOL_CALL sections. You must add your detailed analysis before TOOL_CALL section. You must respond in the following format.
    <<your detailed thought process>>
    ======TOOL_CALL
    {"name":"<tool_name>","arguments":{...}}
    """
    
    TEST_CASE_GENERATOR_SYSTEM_PROMPT=textwrap.dedent("""You are an expert Python unittest testcase developer. 
    Important points:-
    - you have generation limit of 2048 tokens. Hence you must stop generating more test cases when you are near the limit.
    - If you get syntax error, check if last assistant response was truncated. If yes, then skip last couple of test cases to fit in.
    
    You must respond directly with the test cases in the following format. 
    =========TEST_CASES
    <<test cases>>
    Do not include anything else. For Example:
    =========TEST_CASES
    # These tests are auto-generated with test data from:
    # https://github.com/xxxx.json
    # File last updated on 2023-07-20
    import unittest
    from main_module import (
        main_func
    )

    class TestFuncA(unittest.TestCase):
        def test_main_func(self):
            self.assertEqual(main_func(), "expected_output")

    if __name__ == "__main__":
        unittest.main()
    """)
    
    TEST_CASES_GEN_INSTANCE_PROMPT=textwrap.dedent("""Problem Statement:\n{problem_statement}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the complete and correct testcases.""")
    
    TESTCASES_CHECK_PROMPT = textwrap.dedent(
    """
    You are an expert testcases reviewer specializing in invalid testcases detection and prevention. Your task is to analyze the generated test code if it's all valid for the problem statement.
    WorfFlow:-
    1. Read the problem statement carefully and note down all the requirements/edge cases/worflows/mathmatical formulas/data/etc.
    2. Derive expected output for each test case one by one. You must include working, reflection, reflection2, reflection3 and final expected output in your response for each test case.
    3. Reply with 'DONE' once you have verfied all the test cases.
    
    
    For Example:
    Generated Test Case Code:
    def test_add_two_numbers(self):
        self.assertEqual(add_two_numbers(1, 2), 3)
    Output:
    Test case 1:test_add_two_numbers:
    - Working - I need to add two numbers and check output. As per the problem statement, all arguments needs to be positive integers.
    Both 1 and 2 are integers. So 1+2=3. So the expected output is 3.
    - Reflection - Let me double check my work above. I checked if both are positive integers and then i checked the calculation. I am confident about my work. 1+2=3. So the expected output is 3.
    - Reflection2 - <<double check your work on reflection, must recheck all the steps and ensure you have not missed anything>>
    - Reflection3 - <<double check your work on reflection2, must recheck all the steps and ensure you have not missed anything>>
    - Final Expected Output:- 3
    
    Now no more test cases to check, let me end my work here.
    DONE
    """)
    
    INSTANCE_TESTCASES_CHECK_PROMPT=textwrap.dedent(""""Problem statement: {problem_statement}\n\nCode skeleton: \n{code_skeleton}\n\nGenerated Test Code:\n{testcode_response}
                                 
    Now check all the test cases above. Please note you must check each test case carefully one by one as per the plan you created.
                                     
    """)
    
    FINAL_OUTPUT_TEST_CHECK=textwrap.dedent("""
    Now share the final code in the following format.
    =========TEST_CASES
    <<final code here>>
    For example:
    # These tests are auto-generated with test data from:
    # https://github.com/xxxx.json
    # File last updated on 2023-07-20
    import unittest
    from main_module import (
        main_func
    )

    class TestFuncA(unittest.TestCase):
        def test_main_func(self):
            self.assertEqual(main_func(), "expected_output")

    if __name__ == "__main__":
        unittest.main()       
    """)
    
    class ResponseValidator:
        def check_syntax_error(content:str,raw_response:str)->str:
            try:
                if "```python" in content:
                    return """Do not include any markups like ```python, ```, etc. in the response.
                Follow the response format strictly like below:-
                =======TEST_CASES
                import unittest
                from main_module import (
                    main_func
                )

                class TestFuncA(unittest.TestCase):
                    def test_main_func(self):
                        self.assertEqual(main_func(), "expected_output")

                if __name__ == "__main__":
                    unittest.main()
                """
                ast.parse(content)
                return "success"
            except Exception as e:
                logger.error(f"Syntax error: {e}")
                logger.error(f"Content: {content}")
                # check the length of content and if high advise to reduce the number of test cases.
                no_tokens=len(content.split(" "))
                if "unittest.main()" not in content:
                    return "Generation limit reached. Response truncated.. Skip last couple of test cases from your last response.."
                return f"Syntax error: {e}\n If the syntax error is due to the response getting truncated skip last couple of test cases and try again."
    
    def process_response(self,response):
        resp=None
        next_tool_name=None
        global TOOL_CALL_FAILED
        #logger.info(f"response: {response}")
        if response is None:
            logger.error("response NONE received..")
            return None
        if type(response) is tuple and len(response)==2:
            _,next_tool_name=response
        elif "{" in response:
            next_tool_name=response
        if next_tool_name:
            json_obj,_,error=CustomOpenAIModelClient.Utils.parse_response(next_tool_name)
            if error:
                resp=error
            elif json_obj:
                try:
                    json_obj=json.loads(json_obj)
                    #logger.info("calling tool: "+json_obj.get("name")+" with arguments: "+json.dumps(json_obj.get("arguments")))
                    resp=self.tool_map[json_obj.get("name")](**json_obj.get("arguments"))
                except Exception as e:
                    TOOL_CALL_FAILED+=1
                    logger.error(f"Error calling tool: {e}")
                    resp=f"Error: {e}"
        return resp
    
    def _sanity_check_code(self,code:str)->[bool,str]:
        try:
            # check for syntax errors
            tree = ast.parse(code)
        except Exception as e:
            return False,str(e)
        # build parent map to determine enclosing classes for functions
        parent_map = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_map[child] = parent

        for node in ast.walk(tree):
            
            if isinstance(node, ast.FunctionDef):
                # find enclosing class name if any
                class_name = None
                current_parent = parent_map.get(node)
                while current_parent is not None:
                    if isinstance(current_parent, ast.ClassDef):
                        class_name = current_parent.name
                        break
                    current_parent = parent_map.get(current_parent)
                #logger.info(f"node: {node}, name: {node.name}, type: {type(node)}, class: {class_name}")
                body = list(node.body)
                # drop leading docstring
                if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
                    body = body[1:]
                #logger.info(f"body: {body}, type: {type(body)}, len: {len(body)},type of body[0]: {type(body[0])}")
                
                if not body or (len(body) == 1 and isinstance(body[0], ast.Pass)):
                    return False, f"function {node.name} has empty body"
        return True,None
    
    def check_code_for_common_errors(self,response:str,raw_response:str)->str:
        if isinstance(response, list):
            for r in response:
                if not r.get("code"):
                    return "'code' key is missing in the response"
                is_success,error_message=self._sanity_check_code(r.get("code"))
                logger.info(f"sanity check code for {r.get('file_name')}: {is_success} {error_message}")
                if not is_success:
                    return error_message
        return "success"
                
    def __init__(self, problem_statement: str):
        self.problem_statement = problem_statement
        self.problem_statement = self.post_process_instruction()
        self.code_skeleton = self.get_code_skeleton()
        global IS_SOLUTION_APPROVED
        IS_SOLUTION_APPROVED=True
        self.tools=[FixTaskEnhancedToolManager.run_code,FixTaskEnhancedToolManager.apply_code_edit,FixTaskEnhancedToolManager.finish,FixTaskEnhancedToolManager.get_file_content,FixTaskEnhancedToolManager.search_in_specified_file_v2,FixTaskEnhancedToolManager.run_python_file]
        
        self.tool_map={tool.__name__:tool for tool in self.tools}
        self.agent_initial_solution_eval=CustomAssistantAgent(system_message=CreateProblemSolver.SYSTEM_PROMPT_INITIAL_SOLUTION_EVAL.format(tools_docs=CustomAssistantAgent.Utils.get_tool_docs(self.tools),format_prompt=self.RESPONSE_FORMAT_SOLUTION_EVAL_2),model_name=QWEN_MODEL_NAME)
        
    
    
    def get_code_skeleton(self) -> str:
        # Initialize the result string
        result = ""
        
        # Walk through the current directory
        for root, _, files in os.walk("."):
            for file in files:
                # Check if the file is a Python file
                if file.endswith(".py") and "solution" not in file:
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        content = f.read()
                    # Concatenate the file name and content
                    result += f"{file}\n{{\n{content}\n}}\n\n"
        
        return result

    def post_process_instruction(self):
        def apply_markup(text_block: str) -> str:
            """
            Apply markup to make whitespaces and empty lines explicit to make llm not confusing and ignoring them.
            For example, if the text block is:

            ```text
            This is a test.

            This is another test!
            ```text

            Then the text block should be:

            ```
            This is a test.
            [EMPTY_LINE]
            This is another test!
            ```
            """
            lines = text_block.split('\n')
            processed_lines = []
            
            should_apply_markup = True
            for line in lines:
                if line.strip() == '':
                    should_apply_markup = True
                    break
                if line[-1] != "." and line[-1] != "!":
                    should_apply_markup = False
                    break
                
            if should_apply_markup == False:
                return text_block

            for i, line in enumerate(lines):
                if line.strip() == '':                
                    processed_line = '[EMPTY_LINE]'
                else:
                    # Mark trailing and leading spaces
                    leading_spaces = len(line) - len(line.lstrip(' '))
                    trailing_spaces = len(line) - len(line.rstrip(' '))
                    
                    processed_line = line
                    if leading_spaces > 0:
                        processed_line = f'[{leading_spaces}_LEADING_SPACES]' + line.lstrip(' ')
                    if trailing_spaces > 0:
                        processed_line = processed_line.rstrip(' ') + f'[{trailing_spaces}_TRAILING_SPACES]'
                
                processed_lines.append(f"\"{processed_line}\"")
            
            return "[\n    " + ",\n    ".join(processed_lines) + "\n]"
                
        # Pattern to match ```text...``` blocks
        pattern = r'```text\n(.*?)\n```'
        
        def replace_text_block(match):
            text_content = match.group(1)
            processed_content = apply_markup(text_content)
            
            return f'```text\n{processed_content}\n```'
        
        # Replace all text blocks with processed versions
        processed_instruction = re.sub(pattern, replace_text_block, self.problem_statement, flags=re.DOTALL)
        logger.info(f"Processed instruction: {processed_instruction}")
        return processed_instruction
    
    def generate_initial_solution(self):
        
        pass
    
    
    def extract_and_write_files(self,initial_solution: str, base_dir: str = ".") -> list:
        import os
# Improvement v1 - 2025-10-24 23:35:24
        
        created_files = []
        
        if not initial_solution.strip():
            logger.info("No solution content to process")
            return created_files
        
        lines = initial_solution.split('\n')
        current_filename = None
        current_content = []
        
        for line in lines:
            # Check if this line is just a Python filename (*.py pattern)
            stripped_line = line.strip()
            
            # Pattern: ends with .py and looks like a filename (no spaces, reasonable length)
            if (stripped_line.endswith('.py') and 
                ' ' not in stripped_line and 
                len(stripped_line) > 3 and 
                '/' not in stripped_line.replace('/', '') and  # Allow subdirectories
                not stripped_line.startswith('#')):  # Not a comment
                
                # Write the previous file if we have one
                if current_filename and current_content:
                    file_path = os.path.join(base_dir, current_filename)
                    # Create directory if needed (for subdirectories)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                    # Join content and remove empty lines at start/end
                    content = '\n'.join(current_content).strip()
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    created_files.append(file_path)
                    print(f"Created file: {file_path}")
                
                # Start new file
                current_filename = stripped_line
                current_content = []
            else:
                # This line is content for the current file
                if current_filename and not line.startswith('{') and not line.startswith('}'):  # Only collect content if we have a filename
                    current_content.append(line)
        
        # Write the last file
        if current_filename and current_content:
            file_path = os.path.join(base_dir, current_filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            content = '\n'.join(current_content).strip()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            created_files.append(file_path)
            print(f"Created file: {file_path}")
        
        return created_files
    
    
    def remove_duplicate_solutions(self,results:list[str])->list[str]:
        req_unq=[]
        key_unq=[]
        for r in results:
            # tries to remove all comments and empty lines and spaces
            if re.sub("#[^\n]*\n","\n",r).replace("\n","").replace(" ","") not in key_unq:
                key_unq.append(re.sub("#[^\n]*\n","\n",r).replace("\n","").replace(" ",""))
                req_unq.append(r)
        return req_unq
    
    async def generate_initial_solution(self):
        logger.info("Generating initial solution")
        agent=_agent_pool.get_agent("initial_solution", QWEN_MODEL_NAME, self.SYSTEM_PROMPT.format())
        
        response=await agent.solve_task(self.INSTANCE_PROMPT.format(problem_statement=self.problem_statement,code_skeleton=self.code_skeleton),response_format=self.RESPONSE_FORMAT_JSON,is_json=True,regex=None,post_process_func=self.check_code_for_common_errors,max_attempts=3,is_parallel=False,disable_reset=False)
        
        if response is None:
            logger.info("Failed to generate initial solution")
            return None
        
        logger.info("Initial solution generated successfully")
        logger.info(response)
        initial_solution="\n".join([r["file_name"]+"\n"+r["code"] for r in response])
        return initial_solution
    
    async def generate_test_cases(self):
        #generate test cases.
       
        agent=_agent_pool.get_agent("test_case_generator", QWEN_MODEL_NAME, self.TEST_CASE_GENERATOR_SYSTEM_PROMPT.format())
        response=await agent.solve_task(self.TEST_CASES_GEN_INSTANCE_PROMPT.format(problem_statement=self.problem_statement,code_skeleton=self.code_skeleton),response_format="",is_json=False,regex=None,post_process_func=self.ResponseValidator.check_syntax_error,max_attempts=3,is_parallel=False,disable_reset=False,return_type=str)
        
        if response is None:
            logger.info("Failed to generate test cases")
            return None
        logger.info(response)
        if False:
            logger.info("Now verifying the test cases...")
            agent=CustomAssistantAgent(system_message=self.TESTCASES_CHECK_PROMPT.format(),model_name=QWEN_MODEL_NAME)
            no_steps=10
            response=await agent.solve_task(self.INSTANCE_TESTCASES_CHECK_PROMPT.format(problem_statement=self.problem_statement,code_skeleton=self.code_skeleton,testcode_response=response),response_format="",is_json=False,regex=None,post_process_func=None,max_attempts=3,is_parallel=False,disable_reset=True,return_type=str)
            no_of_rechecks=1
            while True:
                while "DONE" not in response:
                    response=await agent.solve_task("continue for other test cases",response_format="",is_json=False,regex=None,post_process_func=None,max_attempts=3,is_parallel=False,disable_reset=True,return_type=str)
                no_of_rechecks+=1
                if no_of_rechecks<=1:
                    response=await agent.solve_task(f"recheck all your test cases once again.. This is your attempt %{no_of_rechecks}th attempt",response_format="",is_json=False,regex=None,post_process_func=None,max_attempts=3,is_parallel=False,disable_reset=True,return_type=str)
                else:
                    break
            response=await agent.solve_task(self.FINAL_OUTPUT_TEST_CHECK,response_format="",is_json=False,regex=None,post_process_func=None,max_attempts=3,is_parallel=False,disable_reset=True,return_type=str)
        
            if response is None or self.ResponseValidator.check_syntax_error(response,None)!="success":
                logger.info("Failed to verify test cases")
                return None
    
        all_test_cases=[]
        for line in response.split("\n"):
            if "def test_" in line.strip() and "(" in line.strip():
                all_test_cases.append(line.strip().split("def test_")[1].split("(")[0].strip())
        with open("test_cases.txt", "w") as f:
            f.write("\n".join(all_test_cases))
        logger.info("test_cases.txt file created successfully")
            
        return response
    async def solve_problem(self):
        start_time = time.time()
        
        initial_solution=await self.generate_initial_solution()
        if initial_solution is None:
            return None
        initial_solutions=[self.generate_initial_solution() for _ in range(2)]
        initial_solutions=await asyncio.gather(*initial_solutions)
        logger.info(Counter(initial_solutions))
        initial_solution=max(initial_solutions, key=initial_solutions.count)
        self.extract_and_write_files(initial_solution)
        await self.generate_test_cases()
        
        response=await self.agent_initial_solution_eval.solve_task(CreateProblemSolver.INSTANCE_PROMPT_INITIAL_SOLUTION_EVAL.format(problem_statement=self.problem_statement,initial_solution=initial_solution),response_format="",is_json=False,regex=None,post_process_func=partial(CustomAssistantAgent.ResponseValidator.check_tool_call_section,correct_format=CreateProblemSolver.RESPONSE_FORMAT_SOLUTION_EVAL_2),max_attempts=3,is_parallel=False,disable_reset=True,return_type=str)
        response=self.process_response(response)
        finish_called_earlier=True
        while True:
            if response!="finish":
                response=await self.agent_initial_solution_eval.solve_task(response,response_format="",is_json=False,regex=None,post_process_func=partial(CustomAssistantAgent.ResponseValidator.check_tool_call_section,correct_format=CreateProblemSolver.RESPONSE_FORMAT_SOLUTION_EVAL_2),max_attempts=3,is_parallel=False,disable_reset=True,return_type=str)
                response=self.process_response(response)
            else:
                if not finish_called_earlier:
                    
                    response=await self.agent_initial_solution_eval.solve_task("Check the problem statement and find out the cases which have not been tested yet. You must check all the mentioned scenarios (outputs, any edge cases, errors, any workflows). Create those test cases and test your solution.",response_format="",is_json=False,regex=None,post_process_func=None,max_attempts=3,is_parallel=False,disable_reset=True,return_type=str)
                    response=self.process_response(response)
                    finish_called_earlier=True
                    continue
                break
        logger.info(f"total time taken: {time.time()-start_time} seconds")
        patch=FixTaskEnhancedToolManager.get_final_git_patch()
        return patch
    
    def get_final_git_patch(self) -> str:
        '''
        Generates git diff patch containing all modifications in working directory
        Useful for capturing comprehensive change summary before finalization
        '''
        try:
            # Update to include cfg, txt, and toml files along with py files
            # Check whether ignore_files is a property of this clas
            command = f"""
            shopt -s globstar

            cp .gitignore .gitignore.backup 2>/dev/null || true
            echo 'src/agent.py' >> .gitignore
            echo 'src/agent_runner.py' >> .gitignore

            git add **/*.py 2>/dev/null || true
            git add **/*.toml 2>/dev/null || true
            git add **/*.cfg 2>/dev/null || true
            git add **/*.txt 2>/dev/null || true

            git diff --cached > .patch.txt
            cat .patch.txt

            mv .gitignore.backup .gitignore 2>/dev/null || true
            """
            print("Generating git patch...")
            output = subprocess.run(["bash", "-c", command], timeout=30, capture_output=True, text=True)
            
            # output = output.stdout.decode("utf-8") + '\n' + output.stderr.decode("utf-8")
            return output.stdout
        except Exception as e:
            logger.error(f"Error generating git patch: {e}")
            return f"Error generating git patch: {e}"

class BugFixSolver:
    
    FIX_TASK_SYSTEM_PROMPT = textwrap.dedent("""
    # Hey there! You're a Coding Assistant 🚀. I have uploaded all files of a python repository. Your current working directory is at the root of that repo. You will be provided with a problem statement and you need to make the necessary changes to fix the issue.

    ## Follow these steps to fix the issue:
    1. As a first step, find the relevant files in the repo to work on.
    2. Localise the code causing the issue.
    3. Edit the sourcecode of the repo to resolve the issue.
    4. Think about edgecases and make sure the fix handles them as well.
    5. Code must always be backward compatible unless explicitly mentioned otherwise in the problem statement.
    6. Thoroughly check the entire code base to ensure the changes made are exhaustive and does not break any other functionality.
    7. Thoroughly check the entire code base to ensure the changes user requested are only limited to the ones you have identified.
    8. Never edit/update the existing test files directly when validating a hypothesis. Instead, when you need a new or focused test to reproduce or protect the fix, use the dedicated test generation tool.
    9. Do not create any new files or directories unless absolutely necessary for the fix. Generated tests are allowed but are excluded from the final patch automatically.
    10. Always check all the test cases which will be impacted with your change and ensure they don't fail.
    11. You need to propose at least 2 meaningfully different and accurate solutions to the problem to the user for approval.
    12. After fixing the issue, you can test the changes by running the run_code tool.
    13. If you find that the error while running the run_code or run_repo_tests tool due to missing dependencies, do not try to solve it as you don't have any internet access.
    14. Call the finish tool to finish the task after you have fixed the issue.

    ## Multi-file awareness (critical):
    - Tests and patch contexts may span multiple files. Do not stop after the first similar match or applied fix.
    - Keep searching the repository after each match and apply consistent changes to every relevant file before finishing.
    - Prefer using `search_in_all_files_content` to enumerate matches across the codebase and `search_in_specified_file_v2` to drill into each file; iterate until no applicable occurrences remain.
    - Re-run tests only after covering all discovered occurrences to avoid partial fixes.

    ## Test generation guidance:
    - Use `generate_test_function(file_path, test_function_code, position)` after discovering the most relevant existing test file.
    - Prefer `position="append"` which appends the test function to the end of the file.
    - Generated tests (new files or appended functions) are tracked and excluded from the final patch automatically, so they must not show up in the final diff.
    - Keep generated tests minimal and focused on the bug and its edge cases.
    - Note that current test functions should be passed originally and generated test function is FAIL_TO_PASS.
    
    Here are the tools you have access to:
    {tools_docs}
    
    
    Here is the response format you need to follow. Your response must not contain multiple THOUGHT or TOOL_CALL sections:
    {format_prompt}
    """)
    
    RESPONSE_FORMAT="""
    ======THOUGHT
    <<your detailed thought process>>
    ======TOOL_CALL
    {"name":"<tool_name>","arguments":{...}}
    """

    FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
    # Now let's start. Here is the problem statement:
    {problem_statement}
    """)
    
    ONESHOT_SYSTEM_PROMPT = """
    You are an autonomous programmer. The user will provide a bug report or 
    feature request (the "problem") plus a compact summary of the most 
    relevant repository files.  Your job is to return ONE *valid* unified 
    diff patch that fixes the problem. If you have any questions, do not ask the user. 
    Instead, solve it to the best of your ability with the knowledge you have.

    You will be provided with a summary of the repository files that are most relevant to the problem.
    Your patch must be valid and apply cleanly when run from the repository root.

    STRICT FORMAT RULES
    1. Return *only* the diff – no prose, no Markdown back-ticks.
    2. The diff must start with 'diff --git a/<path> b/<path>' followed by 
    the standard "--- a/<path>" and "+++ b/<path>" headers.
    3. Use -u style context hunks that begin with lines like @@ -N,M +N,M @@.
    4. Every changed file needs its own header block as in rule 2.
    5. End the patch with a trailing newline.

    Be exact: if the diff is syntactically malformed or wrapped in extra 
    text the automated patch tool will fail.
    """
    TIMEOUT=900
    MAX_FIX_TASK_STEPS=100
    
    class ResponseValidator:
        def _check_syntax_and_format(resp:dict,raw_response:str)->str:
            if type(resp) is dict and resp.get("code_response",None) is not None:
                patch_text = resp.get("code_response",None)
                ok, dry_out = BugFixSolver.Utils._dry_run_patch(patch_text)
                if ok:
                    return "success"
                else:
                    logger.info(f"[agent] Patch failed to apply. {dry_out}")
                    return "Patch failed to apply. Please reply with a corrected unified diff only in the CODE_RESPONSE section."
            return "Invalid response format"
        
        def _check_syntax_and_format_md(resp:str)->str:
            if resp:
                patch_text = resp
                logger.info(f"[agent] Patch text: {patch_text}")
                ok, dry_out = BugFixSolver.Utils._dry_run_patch(patch_text)
                if ok:
                    return "success"
                else:
                    logger.info(f"[agent] Patch failed to apply. {dry_out}")
                    return "Patch failed to apply. Please reply with a corrected unified diff only."
            return "Invalid response format"
        
            
        
            
    
    def __init__(self,problem_statement:str,top_k:int=30):
        self.problem_statement=problem_statement
        self.top_k=top_k
        self.tools=[FixTaskEnhancedToolManager.get_file_content,FixTaskEnhancedToolManager.save_file,FixTaskEnhancedToolManager.get_approval_for_solution,FixTaskEnhancedToolManager.get_functions,FixTaskEnhancedToolManager.get_classes,FixTaskEnhancedToolManager.search_in_all_files_content,
               FixTaskEnhancedToolManager.search_in_specified_file_v2,FixTaskEnhancedToolManager.start_over,FixTaskEnhancedToolManager.run_code,FixTaskEnhancedToolManager.apply_code_edit,FixTaskEnhancedToolManager.generate_test_function,FixTaskEnhancedToolManager.finish]
        self.tool_map={tool.__name__:tool for tool in self.tools}
        
        self.agent=CustomAssistantAgent(system_message=self.FIX_TASK_SYSTEM_PROMPT.format(tools_docs=CustomAssistantAgent.Utils.get_tool_docs(self.tools),format_prompt=self.RESPONSE_FORMAT),model_name=GLM_MODEL_NAME)
        #logger.info("system message: "+self.agent.system_message)
    
    def process_response(self,response):
        resp=None
        next_tool_name=None
        global TOOL_CALL_FAILED
        #logger.info(f"response: {response}")
        if response is None:
            logger.error("response NONE received..")
            return None
        if type(response) is tuple and len(response)==2:
            _,next_tool_name=response
        elif "{" in response:
            next_tool_name=response
        if next_tool_name:
            json_obj,_,error=CustomOpenAIModelClient.Utils.parse_response(next_tool_name)
            if error:
                resp=error
            elif json_obj:
                try:
                    json_obj=json.loads(json_obj)
                    #logger.info("calling tool: "+json_obj.get("name")+" with arguments: "+json.dumps(json_obj.get("arguments")))
                    resp=self.tool_map[json_obj.get("name")](**json_obj.get("arguments"))
                except Exception as e:
                    TOOL_CALL_FAILED+=1
                    logger.error(f"Error calling tool: {e}")
                    resp=f"Error: {e}"
        return resp
    
    async def find_relevant_test_files(self):
        response=await self.agent.solve_task("Help me with the existing test files (not generated by you) which are most relevant to the problem statement based on the test files you have explored so far. You must respond with the information you have till now without any further exploration. Strictly follow the response format given below.",response_format="=======THOUGHT\n<<your thought>>\n=======TEST_FILES\n<test_file1>,<test_file2>,...",is_json=False,regex=None,post_process_func=None,max_attempts=3,is_parallel=False,disable_reset=True,return_type=Union[tuple[str,str],str])
        logger.info(f"response from find_relevant_test_files: {response}")
        if isinstance(response,tuple) and len(response)==2 and isinstance(response[1],str):
            files_to_test_for_p2p=response[1].split(",")
            files_to_test_for_p2p=[os.path.normpath(f.strip()) for f in files_to_test_for_p2p if f.strip() and os.path.exists(f.strip())]
        else:
            files_to_test_for_p2p=[]
        return files_to_test_for_p2p
    async def solve_problem(self):
        
        
        logger.info(f"Starting main agent execution...")
        
        instance_prompt = self.FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=self.problem_statement)
        # save initial state for comparision later...
        st=create_checkpoint(".","initial_commit")
        if st.get("status")=="success":
            logger.info(f"initial commit created")
        else:
            logger.error(f"initial commit creation failed")
            
        start_time = time.time()
        logs: List[str] = []
        logs.append(f"cwd: {os.getcwd()}")
        response=await self.agent.solve_task(instance_prompt,response_format="",is_json=False,regex=None,post_process_func=None,max_attempts=3,is_parallel=False,disable_reset=True,return_type=tuple[str,str])
        success_count = 0
        for step in range(self.MAX_FIX_TASK_STEPS):
            resp=self.process_response(response)
            if resp!="finish":
                logger.info(f"Execution step {step + 1}/{self.MAX_FIX_TASK_STEPS}")
                response=await self.agent.solve_task(resp,response_format="",is_json=False,regex=None,post_process_func=None,max_attempts=3,is_parallel=False,disable_reset=True,return_type=tuple[str,str])
                
                # Early termination: if we've had 3 consecutive successful steps, we can exit early
                if "success" in resp.lower() or "completed" in resp.lower():
                    success_count += 1
                    if success_count >= 3:
                        logger.info("Early termination: 3 consecutive successful steps detected")
                        break
                else:
                    success_count = 0
            else:
                # one final check to see if no pass_to_pass test are failing..
                files_to_test=FixTaskEnhancedToolManager.generated_test_files
                files_to_test=[f for f in files_to_test if "/" in f]
                if not files_to_test:
                    logger.info(f"generated test files are empty, finding relevant test files..")
                    files_to_test=await self.find_relevant_test_files()
                    files_to_test=[f for f in files_to_test if "/" in f] if files_to_test else []
                    logger.info(f"relevant test files: {files_to_test} found..")
                if files_to_test:
                    # Normalize paths and deduplicate to avoid treating ./tests/abc.py and tests/abc.py as different
                    repo_test_response=FixTaskEnhancedToolManager.run_repo_tests(list(set(os.path.normpath(f) for f in files_to_test)))
                    st=switch_checkpoint(".","initial_commit",True)
                    if st.get("status")=="success":
                        logger.info(f"switched to initial_commit")
                    else:
                        logger.error(f"initial commit switching failed {st}")
                    initial_test_response=FixTaskEnhancedToolManager.run_repo_tests(list(set(os.path.normpath(f) for f in files_to_test)))
                    st=restore_stashed_changes(".",0,False)
                    if st.get("status")=="success":
                        logger.info(f"restored to working state..")
                    else:
                        logger.error(f"stashed changes restoration failed {st}")
                    failures=FixTaskEnhancedToolManager.parse_run_repo_tests_response(repo_test_response,initial_test_response)
                    
                    logger.info(f"files_to_test: {files_to_test}")
                    logger.info(f"repo_test_response: {repo_test_response}")
                    logger.info(f"failures: {failures}")
                    if len(failures)>0:
                        logger.info(f"total {len(failures)} failed tests detected...")
                        repo_failures="Your fix has broken some tests. Please fix them and then call the finish tool to finish the task.\n"+("\n\n===============================================\n\n".join(failures))
                        response=await self.agent.solve_task(repo_failures,response_format="",is_json=False,regex=None,post_process_func=None,max_attempts=10,is_parallel=False,disable_reset=True,return_type=Union[tuple[str,str],str])
                        continue
                break
            
        final_patch=FixTaskEnhancedToolManager.get_final_git_patch(initial_checkpoint="initial_commit")
        if not DISABLE_TEST_FILE_REMOVAL:
            FixTaskEnhancedToolManager.remove_any_generated_test_files()
        
        logger.info(f"Final patch: {final_patch}")
        return final_patch
    async def solve_problem_one_go(self):
        code_chunks = self._collect_code_chunks()
        chunk_texts=[c.text for c in code_chunks]
        PRE_FILTER_TOP = int(os.getenv("PREFILTER_TOP", "50"))  # Reduced from 200 for faster processing

        if len(chunk_texts) > PRE_FILTER_TOP:
            # Simple TF-IDF scoring
            problem_words = set(self.problem_statement.lower().split())
            chunk_scores = []
            
            for chunk_text in chunk_texts:
                chunk_words = set(chunk_text.lower().split())
                common_words = problem_words.intersection(chunk_words)
                score = len(common_words) / max(len(problem_words), 1)
                chunk_scores.append(score)
            
            # Keep top chunks by TF-IDF score
            sorted_indices = sorted(range(len(chunk_scores)), key=lambda i: -chunk_scores[i])
            top_indices = sorted_indices[:PRE_FILTER_TOP]
            
            code_chunks = [code_chunks[i] for i in top_indices]
            chunk_texts = [chunk_texts[i] for i in top_indices]

            # --------------------------------------------------------------------
            # Embed problem text and all chunks ----------------------------------
            # --------------------------------------------------------------------

            query_vec = Network._remote_embed(self.problem_statement)
            chunk_vecs: List[List[float]] = [None] * len(chunk_texts)  # type: ignore

            MAX_WORKERS = min(8, int(os.getenv("EMBED_CONCURRENCY", "8")))  # Increased concurrency

            print(f"[Agent] Embedding {len(chunk_texts)} chunks with {MAX_WORKERS} workers")
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                fut_to_idx = {pool.submit(Network.safe_remote_embed, txt): idx for idx, txt in enumerate(chunk_texts)}

                for fut in concurrent.futures.as_completed(fut_to_idx):
                    idx = fut_to_idx[fut]
                    try:
                        chunk_vecs[idx] = fut.result()
                    except Exception as exc:
                        # Log and keep zero vector; retrieval will simply rank it low.
                        print(f"[agent] embedding error (chunk {idx}): {exc}")
                        chunk_vecs[idx] = Network.ZERO_VEC

            sims = [ self.Utils._cosine(vec, query_vec) for vec in chunk_vecs ]

            # --------------------------------------------------------------------
            # Light path-based bonus if filename is mentioned in the problem text --
            # --------------------------------------------------------------------
            prob_lower = self.problem_statement.lower()
            for idx, ch in enumerate(code_chunks):
                base = os.path.basename(ch.file).lower()
                if base in prob_lower or base.split(".")[0] in prob_lower:
                    sims[idx] += 0.2

            sorted_idx = sorted(range(len(sims)), key=lambda i: -sims[i])
            logger.info(f"[Agent] Sorted indices: {sorted_idx}")
            TARGET_TOKENS = 6_000  # Reduced from 12_000 for faster processing
            token_budget = int(TARGET_TOKENS * 0.85)
            token_total = 0
            top_idx: list[int] = []
            for idx in sorted_idx:
                tok = self.Utils._guess_tokens(chunk_texts[idx])
                logger.info(f"[Agent] Token: {tok}")
                if token_total + tok > token_budget:
                    break
                token_total += tok
                top_idx.append(idx)

            # Fallback to at most top_k if budget yields too many
            if len(top_idx) > self.top_k:
                top_idx = top_idx[:self.top_k]
            logger.info(f"[Agent] Top indices: {top_idx}")
            summary_parts: list[str] = []
            for idx in top_idx:
                ch = code_chunks[idx]
                body = ch.text[:5000]
                tag = self.Utils._lang_tag(ch.file)
                header = f"### {ch.file}:L{ch.start_line}-{ch.end_line}"
                summary_parts.append(f"{header}\n```{tag}\n{body}\n```")

            repo_summary = "\n\n".join(summary_parts)

            # --------------------------------------------------------------------
            # Build initial conversation messages.
            # --------------------------------------------------------------------

            print(f"[Agent] Repository summary: {len(repo_summary)} chars, {len(top_idx)} chunks, {token_total} tokens")
            print(f"[Agent] repo_summary:\n{repo_summary}")

            #"""OUTPUT RULES (VERY IMPORTANT, STRICT)
            #• You MUST end your reply with a *raw* JSON object with "code_response" property – nothing else.
            #• Must hold the unified diff from rules 1-5 *verbatim*.
            #Example: {"code_response": "diff --git a/foo.py b/foo.py\n..."}"""
            #You must directly return the diff in the code_response section.
            #                                        Must hold the unified diff from rules 1-5 *verbatim*.
            #                                        Strictly follow the response format given below.
            #                                       Do not add any other text before or after the CODE_RESPONSE section.
            #                                        OUTPUT:
            #                                        ========CODE_RESPONSE
            #                                        diff --git a/foo.py b/foo.py
            #                                        ...
            proxy_resp = await self.agent.solve_task(self.problem_statement+"\n\n"+"Repository summary (top files):\n\n"+repo_summary, response_format="""OUTPUT RULES (VERY IMPORTANT, STRICT)
            • You MUST end your reply with a *raw* JSON object with "thought" and "code_response" property – nothing else.
            • Must hold the unified diff from rules 1-5 *verbatim*.
            • Do not include any other text other than the JSON object.
            Example: {"thought":"your thought process","code_response": "diff --git a/foo.py b/foo.py\n..."}""", is_json=True, regex=None, post_process_func=self.ResponseValidator._check_syntax_and_format, max_attempts=3, is_parallel=False, disable_reset=False)

            logger.info(f"[agent] Proxy response received: {proxy_resp}")
            code_resp = proxy_resp.get("code_response",None)
            if not code_resp:
                return None
            
            patch_text = self.Utils._sanitize_patch(code_resp)
            logger.info(f"[agent] Sanitized patch : {patch_text}")

            result = self.Utils._apply_patch(patch_text)
            return patch_text

    def _collect_code_chunks(self,root: str = ".") -> List[Chunk]:
        """Collect function-level code chunks from Python files."""
        chunks: List[Chunk] = []
        
        for root, _, files in os.walk(root):
            # Skip hidden directories and common non-source dirs
            if any(part.startswith('.') for part in Path(root).parts):
                continue
            if root.endswith(('__pycache__', 'node_modules', '.git')):
                continue
                
            for file in files:
                if not file.endswith('.py'):
                    continue
                    
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except (UnicodeDecodeError, PermissionError):
                    continue
                    
                try:
                    tree = ast.parse(content, filename=file_path)
                except SyntaxError:
                    continue
                    
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        start_line = getattr(node, 'lineno', 1)
                        end_line = getattr(node, 'end_lineno', start_line)
                        
                        if end_line is None:
                            # Fallback: count lines in the node
                            lines = content.split('\n')
                            if start_line <= len(lines):
                                node_text = '\n'.join(lines[start_line-1:])
                                end_line = start_line + node_text.count('\n')
                            else:
                                end_line = start_line
                        
                        # Extract the actual source lines
                        lines = content.split('\n')
                        if start_line <= len(lines) and end_line <= len(lines):
                            chunk_text = '\n'.join(lines[start_line-1:end_line])
                            
                            # Skip if chunk is too large
                            if len(chunk_text) > MAX_EMBED_CHARS:
                                continue
                                
                            chunks.append(Chunk(
                                file=file_path,
                                start_line=start_line,
                                end_line=end_line,
                                text=chunk_text
                            ))
                            
                # Also add the entire file as a chunk if it's reasonably sized
                if len(content) <= MAX_EMBED_CHARS and content.strip():
                    chunks.append(Chunk(
                        file=file_path,
                        start_line=1,
                        end_line=content.count('\n') + 1,
                        text=content
                    ))
        
        return chunks
    
    class Utils:
        @classmethod
        def _lang_tag(cls,path: str) -> str:
            """Return language tag for syntax highlighting."""
            ext = os.path.splitext(path)[1].lower()
            lang_map = {
                '.py': 'python',
                '.js': 'javascript',
                '.ts': 'typescript',
                '.java': 'java',
                '.cpp': 'cpp',
                '.c': 'c',
                '.h': 'c',
                '.hpp': 'cpp',
                '.cs': 'csharp',
                '.go': 'go',
                '.rs': 'rust',
                '.php': 'php',
                '.rb': 'ruby',
                '.swift': 'swift',
                '.kt': 'kotlin',
                '.scala': 'scala',
                '.r': 'r',
                '.m': 'matlab',
                '.sh': 'bash',
                '.sql': 'sql',
                '.html': 'html',
                '.css': 'css',
                '.xml': 'xml',
                '.yaml': 'yaml',
                '.yml': 'yaml',
                '.json': 'json',
                '.md': 'markdown',
                '.txt': 'text',
            }
            return lang_map.get(ext, 'text')
        def _cosine(u: List[float], v: List[float]) -> float:
            """Compute cosine similarity between two vectors."""
            nu = math.sqrt(sum(x * x for x in u))
            nv = math.sqrt(sum(x * x for x in v))

            if nu == 0 or nv == 0:
                return 0.0

            return sum(x * y for x, y in zip(u, v)) / (nu * nv)
        
        def _guess_tokens(text: str) -> int:
            """Rough token count estimate: ~0.75 tokens per word."""
            return int(len(text.split()) * 0.75)
        
        
        def _sanitize_patch(patch: str) -> str:
            """Return *patch* stripped of markdown/code-fence chatter.

            The LLM sometimes wraps the unified diff in Markdown fences or appends
            explanatory text (e.g. `DISCUSSION`, `EOF`).  This helper keeps only the
            lines that are valid in a unified diff so downstream tools (`patch`,
            `git apply`) succeed and the resulting patch file is clean.
            """
            # Stop at the first closing code fence (```), if present.
            stop_idx = patch.find("\n```")
            if stop_idx != -1:
                patch = patch[:stop_idx]

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
            cleaned_lines: list[str] = []
            for line in patch.splitlines():
                # Early exit markers – anything after is junk.
                if line.strip() in {"DISCUSSION", "EOF"}:
                    break
                if line.startswith(allowed_prefixes):
                    cleaned_lines.append(line)
                    continue
                # Hunk content lines start with space, plus or minus.
                if line.startswith(("+", "-", " ")):
                    cleaned_lines.append(line)
                    continue
                # Ignore everything else (markdown, commentary, empty lines outside hunks).

            # Post-process header lines: ensure 'a/' and 'b/' prefixes so a default
            # `patch -p1` invocation resolves correctly when run from the repository
            # root.  We intentionally *do not* touch paths pointing to /dev/null.
            for idx, ln in enumerate(cleaned_lines):
                if ln.startswith("--- ") and not ln.startswith("--- a/") and not ln.startswith("--- /dev/null"):
                    cleaned_lines[idx] = "--- a/" + ln[4:]
                elif ln.startswith("+++ ") and not ln.startswith("+++ b/") and not ln.startswith("+++ /dev/null"):
                    cleaned_lines[idx] = "+++ b/" + ln[4:]

            return "\n".join(cleaned_lines) + "\n"
        
        
        def _dry_run_patch(patch: str) -> tuple[bool, str]:
            """Validate patch without applying it."""
            try:
                # Sanitize the patch first
                sanitized_patch = BugFixSolver.Utils._sanitize_patch(patch)
                
                if not sanitized_patch.strip():
                    return False, "Empty patch"
                
                # Write to temporary file
                with open(".temp_patch", "w") as f:
                    f.write(sanitized_patch)
                
                # Try to apply with --check flag
                result = subprocess.run(
                    ["git", "apply", "--check", ".temp_patch"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                return result.returncode == 0, result.stderr
                
            except Exception as e:
                return False, str(e)
            finally:
                # Clean up
                try:
                    os.remove(".temp_patch")
                except:
                    pass
                
        def _apply_patch(patch: str) -> str:
            """Apply a git patch and return the result."""
            try:
                # Write patch to temporary file
                with open(".temp_patch", "w") as f:
                    f.write(patch)
                
                # Apply the patch
                result = subprocess.run(
                    ["git", "apply", ".temp_patch"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    # Generate final patch
                    final_patch = subprocess.run(
                        ["git", "diff"],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    return final_patch.stdout
                else:
                    print(f"Patch application failed: {result.stderr}")
                    return ""
                    
            except Exception as e:
                print(f"Error applying patch: {e}")
                return ""
            finally:
                # Clean up temporary file
                try:
                    os.remove(".temp_patch")
                except:
                    pass
    


class ProblemTypeClassifierAgent:
    
    PROBLEM_TYPE_CREATE="CREATE"
    PROBLEM_TYPE_FIX="FIX"
    
    
    SYSTEM_PROMPT='''
    You are the problem type checker that will categories problem type into:

    1. CREATE: If the problem statement is about creating a new functionality from scratch.
    2. FIX: If the problem statement is about fixing a bug, creating a new functionality or improving the existing codebase.

    Only respond with the "FIX" or "CREATE".
    '''
    
    @classmethod
    def get_directory_tree(cls,start_path: str = '.') -> str:

        tree_lines = []
        
        def add_directory_tree(path: str, prefix: str = "", is_last: bool = True, is_root: bool = False):
            """Recursively build the tree structure"""
            try:
                # Get the directory name
                dir_name = os.path.basename(path) if path != '.' else os.path.basename(os.getcwd())
                
                # Add current directory to tree (skip for root directory)
                if not is_root:
                    connector = "└── " if is_last else "├── "
                    tree_lines.append(f"{prefix}{connector}{dir_name}/")
                
                # Get all items in directory
                try:
                    items = os.listdir(path)
                    # Filter out hidden directories and files starting with '.'
                    items = [item for item in items if not item.startswith('.')]
                    items.sort()
                    
                    # Separate directories and files
                    dirs = []
                    files = []
                    for item in items:
                        item_path = os.path.join(path, item)
                        if os.path.isdir(item_path):
                            dirs.append(item)
                        else:
                            files.append(item)
                    
                    # Process directories first
                    for i, dir_name in enumerate(dirs):
                        dir_path = os.path.join(path, dir_name)
                        is_last_dir = (i == len(dirs) - 1) and len(files) == 0
                        new_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
                        add_directory_tree(dir_path, new_prefix, is_last_dir, False)
                    
                    # Then process files
                    for i, file_name in enumerate(files):
                        is_last_file = i == len(files) - 1
                        connector = "└── " if is_last_file else "├── "
                        tree_lines.append(f"{prefix}{'' if is_root else ('    ' if is_last else '│   ')}{connector}{file_name}")
                        
                except PermissionError:
                    # Handle directories we can't read
                    error_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
                    tree_lines.append(f"{error_prefix}└── [Permission Denied]")
                    
            except Exception as e:
                tree_lines.append(f"{prefix}└── [Error: {str(e)}]")
    
        add_directory_tree(start_path, is_root=True)
        return "\n".join(tree_lines)
    
    @classmethod
    async def check_problem_type(cls,problem_statement:str):
        system_message='''
            You are the problem type checker that will categories problem type into:

            1. CREATE: If the problem statement is about creating a new functionality from scratch. The codebase shared would be very small with no more than few files.
            2. FIX: If the problem statement is about fixing a bug, creating a new functionality or improving the existing codebase. Codebase for this **MUST contain multiple files and directories**.

            Only respond with the "FIX" or "CREATE". Your response cannot contain multiple THOUGHT or TOOL_CALL sections.
            '''
        instance_prompt=f"{problem_statement}\n# Project Tree Structure: \n{cls.get_directory_tree()}"
        agent=_agent_pool.get_agent("problem_classifier", GLM_MODEL_NAME, system_message, "problem_type_classifier_agent")
        response=await agent.solve_task(instance_prompt,response_format="=======THOUGHT\n<<your thought>>\n=======PROBLEM_TYPE\n<<problem type>>",is_json=False,regex=None,post_process_func=None,max_attempts=3,is_parallel=False,disable_reset=True,return_type=Union[tuple[str,str],str])
        logger.info("classifier response: {}".format(response))
        while True:
            
            if isinstance(response,tuple) and len(response)==2 and isinstance(response[1],str):
                if response[1].strip()=="FIX":
                    return cls.PROBLEM_TYPE_FIX
                elif response[1].strip()=="CREATE":
                    return cls.PROBLEM_TYPE_CREATE
            elif isinstance(response,str):
                if response.strip()=="FIX":
                    return cls.PROBLEM_TYPE_FIX
                elif response.strip()=="CREATE":
                    return cls.PROBLEM_TYPE_CREATE
            response=await agent.solve_task("Invalid response, please respond problem_type with the 'FIX' or 'CREATE'.",response_format="=======THOUGHT\n<<your thought>>\n=======PROBLEM_TYPE\n<<problem type>>",is_json=False,regex=None,post_process_func=None,max_attempts=3,is_parallel=False,disable_reset=True,return_type=Union[tuple[str,str],str])
            logger.info("classifier response: {}".format(response))


class CustomAssistantAgent(AssistantAgent):
    
    class ResponseValidator:
        def check_tool_call_section(response:str,raw_response:str,correct_format:str)->str:
            if not("TOOL_CALL" in raw_response and re.search("^=+\s*[A-Z_]+$",raw_response,re.MULTILINE)):
                return "Invalid response, please respond in correct format: {}".format(correct_format)
            return "success"
    
    class Utils:
        
        def tool_parsing(fn):
            tool_schemas = None
            name = fn.__name__
            doc_fn = fn.__doc__ or ""
            # remove parameters section from here to be put in args section
            doc=doc_fn.split("Arguments:")[0]
            output_description=doc_fn.split("Output:")
            if len(output_description)>1:
                output_description="Output: "+output_description[1].strip()
                doc=doc+"\n\n"+output_description
            sig = inspect.signature(fn)
            properties = {}
            required = []
            for param in sig.parameters.values():
                if param.name == 'self':
                    continue
                if param.default is param.empty and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                    required.append(param.name)
                type_hint = str(param.annotation) if param.annotation != param.empty else "string"
                param_description=re.search(f"{param.name}:([^\n]+)",doc_fn)
                if param_description:
                    param_description=param_description.group(1)
                else:
                    raise ValueError(f"Parameter description not found for {param.name} in {doc_fn}: tool name: {name}")
                # Special handling for list[str] / List[str] annotations so that the
                # generated JSON schema correctly represents an array of strings.
                if ("list" in type_hint.lower()) and ("str" in type_hint):
                    properties[param.name] = {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": param_description
                    }
                    continue
                elif 'str' in type_hint:
                    json_type = "string"
                elif 'int' in type_hint:
                    json_type = "integer"
                elif 'float' in type_hint:
                    json_type = "number"
                elif 'bool' in type_hint:
                    json_type = "boolean"
                else:
                    json_type = "string"
                properties[param.name] = {
                    "type": json_type,
                    "description": param_description
                }
            parameters = {
                "type": "object",
                "properties": properties,
                "required": required
            }
            tool_schemas={
                "name": name,
                "description": doc.strip(),
                "input_schema": parameters
            }
            
            return tool_schemas
        def get_tool_docs(tool_list:list)->str:
            return '\n\n'.join([json.dumps(CustomAssistantAgent.Utils.tool_parsing(tool), ensure_ascii=False) for tool in tool_list])
    def __init__(self,agent_name="assistant",model_name:str=QWEN_MODEL_NAME,system_message:str=None,tools:list=None):
        self.semaphore=asyncio.Semaphore(3)
        self.agent_idx=0
        self.agent_name=agent_name
        self.model_name=model_name
        self.system_message=system_message
        #self.register_hook("process_all_messages_before_reply",CustomAssistantAgent.Utils.drop_last_assistant)
        self.model_client = CustomOpenAIModelClient(model_name=self.model_name, api_key="", base_url=DEFAULT_PROXY_URL,agent_prefix="test_generator")
        if not tools:
            self.agent:AssistantAgent=AssistantAgent(
                name=self.agent_name,
                model_client=self.model_client,
                reflect_on_tool_use=False,
                system_message=self.system_message
            )
        else:
            self.agent:AssistantAgent=AssistantAgent(
                name=self.agent_name,
                model_client=self.model_client,
                reflect_on_tool_use=False,
                system_message=self.system_message,
                tools=tools
            )
    
    def parse_markdown(self,text:str,return_type:type=None)->Any:
        """Parse markdown text with ==== section headers into a tuple based on return_type."""
        global MARKDOWN_FAILED
        global TOO_MANY_SECTIONS_FOUND
        
        if get_origin(return_type) is Union:
            # determine what we have..
            args=get_args(return_type)
            no_sections=re.findall(f"^=+\s*[A-Z_]+$",text,re.MULTILINE)
            if len(no_sections)<=1 and str in args:
                return_type=str
            
        if return_type==str:
            start_line=[idx for idx,t in enumerate(text.split("\n")) if re.search(f"^=+\s*[A-Z_]+$",t)]
            if start_line:
                start_line=start_line[0]
            else:
                start_line=0
            return True,None,"\n".join([t for t in text.split("\n")[start_line:] if (not re.search(f"^=+\s*[A-Z_]+$",t) and not re.search("^={3,}$",t))])
        if return_type==list:
            return True,None,[t for t in text.split("\n") if not re.search(f"^=+\s*[A-Z_]+$",t)]
        if not return_type:
            return True,None,text
        if return_type==tuple or True:
            sections = []
            current_content = []
            
            lines = text.split("\n")
            start_line=0
            #skipping the start section which does not belong to any section..
            for idx,line in enumerate(lines):
                if re.search(f"^=+\s*[A-Z_]+$",line):
                    start_line=idx
                    break
            for line in lines[start_line:]:
                if re.search(f"^=+\s*[A-Z_]+$",line):
                    # If we have accumulated content, save it
                    if current_content:
                        sections.append("\n".join(current_content).strip())
                        current_content = []
                    # Extract section name from the header line
                    match = re.search(r"([A-Z]+)", line)
                    if match:
                        # Section header found, reset content accumulator
                        current_content = []
                else:
                    # Accumulate content for current section
                    current_content.append(line)
            
            # Don't forget the last section
            if current_content:
                sections.append("\n".join(current_content).strip())
            
            # Check if return_type is a tuple and get expected length
        
            # Try to extract tuple length from return_type hints like tuple[str, str]
            if hasattr(return_type, '__args__'):
                expected_length = len(return_type.__args__)
            else:
                # If no hints, just return tuple of all sections
                return True,None,tuple(sections)
            
            if len(sections) != expected_length:
                MARKDOWN_FAILED+=1
                return False, f"Expected {expected_length} markdown sections but found {len(sections)}", None
            else:
                return True,None,tuple(sections)
        
        else:
            MARKDOWN_FAILED+=1
            raise ValueError(f"Invalid return type: {return_type}")
                    
        
    async def solve_task(self,task:str,response_format:str,is_json:bool,regex:str,post_process_func:Callable=None,max_attempts:int=3,is_parallel:bool=False,disable_reset:bool=False,return_type=None):
        # write code here to make the request to assistant agent , fetch response, check the response format matches with regex shared. If not it must through an error and ask to regenerate. It must then convert the reponse to json if is_json is True else return. Also, if there is any error in parsing json it must send an error to LLM and ask to regerate. The LLM must have less than 3 attempts to generate the response successfully.
        
        async with self.semaphore:
            if is_parallel:
                logger.info("Creating new agent..{}".format(self.agent_idx))
                self.agent_idx+=1
                agent=AssistantAgent(
                    name=self.agent_name,
                    model_client=CustomOpenAIModelClient(model_name=self.model_name, api_key="", base_url=DEFAULT_PROXY_URL,agent_prefix=self.agent_name),
                    reflect_on_tool_use=False,
                    system_message=self.system_message
                )
            else:
                agent=self.agent
            
            if not disable_reset:
                await agent.on_reset(None)
                
            attempts = 0
            
            full_task = (
                    f"{task}\n\n"
                    f"\n{response_format}\n\n"
                )
            while attempts < max_attempts:
                
                logger.info(f"assistant response attempt {attempts}..")
                attempts += 1
                try:
                    result: TaskResult=await Console(agent.run_stream(task=full_task))
                except Exception as e:
                    logger.info(f"Agent call failed: {type(e)}:{e}, sleeping for 2 seconds before retrying..")
                    time.sleep(2)
                    continue
                
                # Find the last non-summary message
                last_message = None
                try:
                    for m in result.messages[::-1]:
                        if isinstance(m, ToolCallSummaryMessage):
                            continue
                        last_message = m
                        break
                except Exception:
                    last_message = None
                
                if not last_message:
                    last_error = "No response message returned by assistant."
                    logger.error("No response message returned by assistant. This should not happen..")
                    continue
                
                candidate_text = CustomOpenAIModelClient.Utils._extract_text_from_message(last_message).strip()
                
                candidate_text,_,error=CustomOpenAIModelClient.Utils.parse_response(candidate_text)
                
                if error:
                    full_task=error
                    logger.info(f"assistant attempt {attempts} error: {error}")
                    continue
                    
                    
                if CustomOpenAIModelClient.Utils.is_empty_response(candidate_text) or CustomOpenAIModelClient.Utils.is_network_error(candidate_text):
                    full_task="network error. please try again."
                    continue
                # Regex validation
                if regex and not re.search(regex, candidate_text, flags=re.DOTALL):
                    full_task=f"Response did not match the required response format. You need to respond with this format: {response_format}"
                    logger.info(f"assistant attempt {attempts} failed regex. Text: {candidate_text[:2000]}")
                    continue
                
                #if "Error: " in candidate_text:
                #    full_task=candidate_text
                #    logger.info(f"assistant attempt {attempts} error: {candidate_text}")
                #    continue
                
                if not is_json:
                    cleaned=self.model_client.Utils._strip_code_fences(candidate_text)
                    is_success,error_message,cleaned=self.parse_markdown(cleaned,return_type)
                    if not is_success:
                        if error_message and "expected" in error_message.lower():  
                            logger.info(f"context length before rejection: {len(agent.model_context._messages)}")
                            logger.info("removing the last assistant messages from context")
                            agent.model_context._messages=[m for m in agent.model_context._messages if  isinstance(last_message,TextMessage) and m.content!=last_message.content]
                            logger.info(f"context length after rejection: {len(agent.model_context._messages)}")
                            sections=re.findall(f"^=+\s*[A-Z_]+$",candidate_text,re.MULTILINE)
                            if sections:
                                sections=[s.replace("=","").strip() for s in sections if s.strip()]
                                logger.info(f"len(sections): {len(sections)}, return_type: {return_type}")
                                if len(sections)>2:
                                    sections=list(set(sections))
                                    full_task="Respond in correct format. You must not have multiple sections of {}".format(",".join(sections))
                                elif return_type==tuple[str,str] and len(sections)<2:
                                   full_task="Respond in the correct format. You are missing a section in your response. Check if THOUGHT or any other section is missing."
                                else:
                                    full_task=None
                            else:
                                full_task=None  
                        else:
                            full_task=error_message
                        logger.info(f"assistant attempt {attempts} error: {error_message}")
                        continue
                    if post_process_func:
                        resp_post_process=post_process_func(cleaned,candidate_text)
                        if resp_post_process!="success":        
                            full_task=f"Invalid response:{resp_post_process}"
                            logger.info(f"assistant attempt {attempts} invalid response: {resp_post_process}")
                            continue
                    return cleaned
                
                # Parse JSON with best-effort cleanup
                try:
                    cleaned = CustomOpenAIModelClient.Utils._strip_code_fences(candidate_text)
                    #cleaned = _coerce_json_slice(cleaned)
                    
                
                    parsed = json.loads(cleaned)
                    if post_process_func:
                        resp_post_process=post_process_func(parsed,candidate_text)
                        if resp_post_process!="success":
                            full_task=f"Invalid response:{resp_post_process}"
                            logger.info(f"assistant attempt {attempts} invalid response: {resp_post_process}")
                            continue
                    return parsed
                except Exception as e:
                    full_task=f"Invalid JSON: {e}. Please respond with the exact same format as the response format: {response_format}"
                    logger.info(f"Unexpected JSON format: {e}")
                    logger.info(f"json tried: {cleaned}")
                    continue
            
            return None




class CustomOpenAIModelClient(OpenAIChatCompletionClient):
    class Utils:
        
        def is_json_string(raw_text:str)->bool:
            return ("{" in raw_text[:10] and "}" in raw_text[len(raw_text)-10:]) or ("[" in raw_text[:10] and "]" in raw_text[len(raw_text)-10:])
        def parse_response(raw_text:str):
            global JSON_LLM_USED,JSON_LITERAL_USED
            raw_text2=raw_text
            #logger.info("raw_text:{}".format(raw_text))
            raw_text=CustomOpenAIModelClient.Utils._strip_code_fences(raw_text)
            try:
                if CustomOpenAIModelClient.Utils.is_json_string(raw_text):
                    raw_text=json.loads(raw_text)
                    if isinstance(raw_text, str): # sometimes server returns leading quotes.
                        raw_text=json.loads(raw_text)
            except Exception as e:
                try:
                    with open("raw_text.txt", "w") as f:
                        f.write(raw_text)
                    with open("raw_text2.txt", "w") as f:
                        f.write(raw_text2)
                    JSON_LITERAL_USED+=1
                    raw_text=literal_eval(raw_text)
                    if isinstance(raw_text, str):
                        raw_text=json.loads(raw_text)
                        if isinstance(raw_text, str):
                            raw_text=json.loads(raw_text)
                except Exception as e:
                    if isinstance(raw_text, str):
                        JSON_LLM_USED+=1
                        logger.info("trying to fix json string with llm")
                        logger.info(raw_text)
                        raw_text_n=Network.fix_json_string_with_llm(raw_text)
                        if raw_text_n:
                            raw_text=raw_text_n
                        else:
                            logger.info("json load failed")
                            error="Invalid JSON: "+str(e)
            
            content_text = ""
            tool_calls = None
            error=""
            if isinstance(raw_text, (dict,list)):
               
                if type(raw_text) == dict and raw_text.get("response_type")=="tool":
                    if raw_text.get("tool_calls") is not None and isinstance(raw_text.get("tool_calls"), list) and len(raw_text.get("tool_calls"))>0:
                        
                        tool_calls=raw_text.get("tool_calls")
                        try:
                            #logger.info("found tool calls")
                            tool_calls=[{"id":stable_tool_call_id(call.get("name"),call.get("arguments")),"type":"function","function":{"name":call.get("name"),"arguments":json.dumps(call.get("arguments") if isinstance(call.get("arguments"), (dict, list)) else {"input":call.get("arguments")})}} for call in tool_calls]
                            content_text=""
                        except Exception as e:
                            error="Invalid tool_calls arguments."
                            logger.error(f"cannot fix tool_calls arguments: {e}")
                            content_text=json.dumps(raw_text)
                            tool_calls=None
                        
                    else:
                        logger.info("found no tool calls, invalid tool_calls arguments.")
                        error="Invalid tool_calls arguments."
                        content_text=json.dumps(raw_text)
                    
                else:
                    #logger.info("json but not a tool call")
                    #logger.info("json but not a tool call, json load succeeded")
                    content_text=json.dumps(raw_text)
            else:
                if (raw_text[0]=="\"" or raw_text[0]=="'") and (raw_text[-1]=="\"" or raw_text[-1]=="'"):
                    try:
                        raw_text=literal_eval(raw_text)
                    except Exception as e:
                        logger.error("literal eval failed..")
                        pass
                content_text=raw_text
            
            return content_text,tool_calls,error

        def is_empty_response(response:str)->bool:
            return not response or response=="null" or response.strip()==""
        

        def is_network_error(response:str)->bool:
            return  "<|reserved_token_" in response or "API request failed with status 429" in response or "Read timed out" in response or "Network unreachable" in response or "Connection refused" in response
        
        

        def _strip_code_fences(text: str) -> str:
            if re.search("^=+\s*[A-Z_]+$",text,re.MULTILINE): # ignore if its a markdown text #^=+\s*[A-Z_]+$
                return text
            if text:
                text=text.strip().strip("\n")
            fenced = re.search(r"```(?:json)\s*(.*?)```$", text, flags=re.DOTALL | re.IGNORECASE)
            if fenced:
                fenced=fenced.group(1).strip().strip("\n")
                return fenced
            m = re.search(r"[^`]*```python\r?\n(.*?)\n?```.*", text, flags=re.DOTALL|re.IGNORECASE)
            text = m.group(1) if m else text
              
            return text.strip()

        def _extract_text_from_message(message):
            try:
                if isinstance(message, TextMessage):
                    return message.content
                if isinstance(message, FunctionExecutionResult):
                    return message.content
                if isinstance(message, ToolCallExecutionEvent):
                    # Tool calls sometimes return a list of chunks
                    try:
                        return message.content[0].content
                    except Exception:
                        return str(message.content)
                # Fallback: try generic .content
                content = getattr(message, "content", None)
                if isinstance(content, list):
                    return "".join([getattr(c, "content", str(c)) for c in content])
                if content is not None:
                    return content
                return str(message)
            except Exception as e:
                return str(message)
    
    def __init__(self, model_name:str, api_key:str, base_url:str,agent_prefix:str):
        self.model_name = model_name
        super().__init__(model=model_name,
        api_key=api_key,
        base_url=base_url,
        model_info={
                    "vision": False,
                    "function_calling": True,
                    "json_output": True,
                    "family": ModelFamily.UNKNOWN,
                    "structured_output": False,
                },timeout=180)
        self.parsing_error=None
        self.agent_prefix=agent_prefix
        self._client._client._event_hooks['request'] = [self.request_modify]
        self._client._client._event_hooks['response'] = [self.response_modify]
    
    
    
    
    async def request_modify(self,request):
        
        self.parsing_error=None

        await request.aread()

        
        try:
            raw = request.content.decode('utf-8') if request.content else '{}'
            body_data = json.loads(raw)
        except Exception as e:
            logger.error(f"Error parsing request content: {e}")
            body_data = {}
        messages=[]
 
        for m in body_data.get("messages"):
            if m.get("content")!=None:
                messages.append(m)
            else:
                content=""
                for k in m.keys():
                    if k not in ["role","content"]:
                        content+=f"{k}: {m[k]}\n"
                messages.append({"role":m.get("role"),"content":content})
        # Build body with only model, messages, and run_id

        new_body = {
            "model": self.model_name,
            "messages": messages,
            "run_id": RUN_ID, 
            "temperature": 0.0,
            "agent_id": self.agent_prefix+":"+AGENT_ID
        }
        new_bytes = json.dumps(new_body).encode('utf-8')
 
        # Update URL and replace stream safely
        request.url = request.url.copy_with(path="/api/inference")
        request.headers["content-type"] = "application/json"
        request.headers["content-length"] = str(len(new_bytes))
        request._content = new_bytes
        request.stream = httpx.ByteStream(new_bytes)  # provide body bytes without httpx
 
        return request

    async def response_modify(self,response1):
        data = await response1.aread()
        #raw_text=response1
        
        raw_text = data.decode('utf-8') if data else ""
        raw_text=raw_text.strip()
        if ("<think>" in raw_text or "</think>" in raw_text) and False:
            raise Exception("think tag found in response")

        content_text,tool_calls,_=CustomOpenAIModelClient.Utils.parse_response(raw_text)
        logger.info(f"Content text: {content_text}")
        
        message = {
            "role": "assistant",
            "content": content_text
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
            logger.info(message)
        oai_response = {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop" if message.get("tool_calls") is None else "tool_calls",
                    "message": message,
                }
            ],
            "usage": {
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
        }

        new_bytes = json.dumps(oai_response).encode("utf-8")
        response1.headers["content-type"] = "application/json"
        response1.headers["content-length"] = str(len(new_bytes))
        response1._content = new_bytes
        response1.stream = httpx.ByteStream(new_bytes)
        return response1
    

def set_env_for_agent():
    
    if os.getcwd() not in os.environ.get("PYTHONPATH",""):
        os.environ["PYTHONPATH"]=os.environ.get("PYTHONPATH","")+":"+os.getcwd()
    if Path(os.getcwd()+"/lib").exists() and os.getcwd()+"/lib" not in os.environ.get("PYTHONPATH",""):
        os.environ["PYTHONPATH"]=os.environ["PYTHONPATH"]+":"+os.getcwd()+"/lib"
        

def stable_tool_call_id(name: str, args: dict | list | str) -> str:
    key = f"{name}:{json.dumps(args, sort_keys=True)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

def ensure_git_initialized():
    """Initialize git repository if not already initialized, with temporary config."""
    print("[DEBUG] Starting git initialization check...")
    
    work_dir = os.getcwd()
    original_cwd = os.getcwd()
    
    try:
        print(f"[DEBUG] Work directory: {work_dir}")
        print(f"[DEBUG] Before chdir - pwd shows: {subprocess.run(['pwd'], capture_output=True, text=True).stdout.strip()}")
        
        os.chdir(work_dir)
        print(f"[DEBUG] After chdir - pwd shows: {subprocess.run(['pwd'], capture_output=True, text=True).stdout.strip()}")
        
        # Initialize git repo if not already initialized
        if not os.path.exists(".git"):
            print("[DEBUG] Initializing git repository...")
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
            
            # Verify .git was created in current directory
            print(f"[DEBUG] .git exists: {os.path.exists('.git')}")
            print(f"[DEBUG] Files in current dir: {os.listdir('.')[:10]}")  # Show first 10 files
            
            # Set local git config (only for this repo)
            print("[DEBUG] Setting git config...")
            subprocess.run(["git", "config", "--global", "user.email", "agent@sandbox.local"], check=True)
            subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], check=True)

            # Add all files
            print("[DEBUG] Adding all files...")
            subprocess.run(["git", "add", "."], check=True)
            
            # Commit (ignore error if nothing to commit)
            print("[DEBUG] Creating initial commit...")
            result = subprocess.run(["git", "commit", "-m", "Initial commit"], check=False, capture_output=True, text=True)
            if result.returncode == 0:
                print("[DEBUG] Initial commit created successfully")
            else:
                print(f"[DEBUG] Commit result: {result.stderr.strip()}")
                
            print("[DEBUG] Git initialization completed successfully")
        else:
            print("[DEBUG] Git repository already exists")
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
        
    except Exception as e:
        print(f"[DEBUG] ERROR: Could not initialize git repository: {e}")
    finally:
        os.chdir(original_cwd)




class Chunk(NamedTuple):
    file: str
    start_line: int
    end_line: int
    text: str

class Network:
    _EMBED_CACHE: Dict[str, List[float]] = {}
    ZERO_VEC: List[float] = [0.0] * 1024  # embedding for empty input
    class ErrorType(Enum):
        EMPTY_RESPONSE=1
        RESERVED_TOKEN_PRESENT=2
        RATE_LIMIT_EXCEEDED=3
        INVALID_RESPONSE_FORMAT=4
        TIMEOUT=5
        UNKNOWN=6
        NETWORK_ERROR=7
        AUTHENTICATION_ERROR=8
        RESOURCE_EXHAUSTED=9
    
    @classmethod
    def is_valid_response(cls,raw_text:str)->bool:
        if type(raw_text) is dict and raw_text.get("error",None) is not None and raw_text.get("error")!="":
            return False,cls.ErrorType.EMPTY_RESPONSE.name
        if len(raw_text)==0:
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if "<|reserved_token_" in raw_text:
            return False, cls.ErrorType.RESERVED_TOKEN_PRESENT.name
        if 'API request failed with status 429' in raw_text:
            return False, cls.ErrorType.RATE_LIMIT_EXCEEDED.name
        if 'Read timed out' in raw_text:
            return False, cls.ErrorType.TIMEOUT.name
        if 'Network unreachable' in raw_text or 'Connection refused' in raw_text:
            return False, cls.ErrorType.NETWORK_ERROR.name
        return True, None
    @classmethod
    def get_error_counter(cls)->dict[str,int]:
        return {
            k:0 for k in cls.ErrorType.__members__
        }   
    @classmethod
    def fix_json_string_with_llm(cls,json_string:str,attempt:int=0)->dict:
        messages=[
            {"role":"system", "content":"Fix the json string sent by the user.  Reply only with the json string and nothing else."},
            {"role":"user", "content":json_string}
        ]
        
        try:
            response=cls.make_request(messages)
            response=response.replace('```json','').strip('```')
            response=json.loads(response)
            return response
        except JSONDecodeError as e:
            logger.error(f"Error fixing json string: {e},trying again..")
            logger.error(f"json string is :{json_string}")
            logger.error(f"LLM response is :{response}")
            attempt+=1
            if attempt>2:
                return None
            return cls.fix_json_string_with_llm(json_string,attempt)
        except Exception as e:
            logger.error(f"Error fixing json string: {e},trying again..")
            attempt+=1
            if attempt>2:
                return None
            return cls.fix_json_string_with_llm(json_string,attempt)
            
            
            
    @classmethod
    def make_request(cls,messages:list,attempt:int=0)->str:
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
        
        # Cache miss - make the actual request
        request_data = {
                "run_id": RUN_ID,
                "messages": messages,
                "temperature": 0.0,
                "agent_id": AGENT_ID
            }

        headers = {
            "Content-Type": "application/json"
        }
        request_data['model']=AGENT_MODELS[attempt%len(AGENT_MODELS)]
        response = requests.post(url, json=request_data, timeout=120, headers=headers)
        logger.info(f"[agent] HTTP {response.status_code} from {url} ({len(response.content)} bytes)")
        
        response.raise_for_status()
        response_json = response.json()
        is_oai_interface= type(response_json) is dict and response_json.get('choices') is not None and len(response_json.get('choices'))>0 and response_json.get('choices')[0].get('message') is not None
        if is_oai_interface:
            raw_text=response_json['choices'][0]['message']['content']
        else:
            if type(response_json) is str:
                raw_text=response_json.strip("\n").strip()
            else:
                raw_text=response_json
        if type(raw_text) is not dict:
            raw_text=raw_text.lstrip()
        return raw_text
    
    @classmethod
    def _request_next_action_with_retry(cls, messages: dict, 
                            max_retries: int = 10, 
                            base_delay: float = 2.0) -> str:
        
        raw_text='not defined'
        error_counter=cls.get_error_counter()
        next_thought, next_tool_name, next_tool_args = None, None, None
        total_attempts=0
        for attempt in range(max_retries):
            try:
                total_attempts+=1
                raw_text=cls.make_request(messages,attempt=attempt)
                is_valid,error_msg=cls.is_valid_response(raw_text)
                if not(is_valid):
                    logger.error("--------------------------------")
                    logger.error(f"raw_text: {raw_text}")
                    logger.error("--------------------------------")
                    raise Exception(error_msg)
                    
                next_thought, next_tool_name, next_tool_args,error_msg = cls.parse_response(raw_text)
                if error_msg:
                    raise Exception(error_msg)
                break  # Success, exit retry loop
            except Exception as e:
                error_body = str(e)
                logger.error(f"Error: {error_body}")
                if attempt < max_retries:
                    delay = base_delay
                    logger.info(error_body)
                    logger.error("--------------------------------")
                    logger.error(f"response: {raw_text}")
                    logger.error("--------------------------------")
                    logger.info(f"[agent] Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})") 
                    if "This may indicate API issues or malformed streaming response." in raw_text:
                        raise RuntimeError("API issues or malformed streaming response")

                    if "RATE_LIMIT_EXCEEDED" in error_body:
                        error_counter[cls.ErrorType.RATE_LIMIT_EXCEEDED.name]+=1
                    elif "RESERVED_TOKEN_PRESENT" in error_body:
                        error_counter[cls.ErrorType.RESERVED_TOKEN_PRESENT.name]+=1
                    elif "EMPTY_RESPONSE" in error_body:
                        error_counter[cls.ErrorType.EMPTY_RESPONSE.name]+=1
                    elif "TIMEOUT" in error_body:
                        error_counter[cls.ErrorType.TIMEOUT.name]+=1
                    elif "Invalid JSON" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name]+=1
                    elif "Invalid response" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name]+=1
                    else:
                        error_counter[cls.ErrorType.UNKNOWN.name]+=1
                    if "RATE_LIMIT_EXCEEDED" not in error_body and "RESERVED_TOKEN_PRESENT" not in error_body and "EMPTY_RESPONSE" not in error_body and  "TIMEOUT" not in error_body:
                        messages.append({"role":"assistant","content":raw_text})
                        messages.append({"role":"user","content":"observation: "+error_body})
                    time.sleep(random.uniform(2*delay, 2.2*delay))
                    continue
                else:
                    error_counter[cls.ErrorType.TIMEOUT.name]+=1
                    # Last attempt failed, raise the error
                    raise RuntimeError(error_body)
        
        return next_thought, next_tool_name, next_tool_args,raw_text,total_attempts,error_counter,messages
    
    @classmethod
    def parse_malformed_json(cls,arguments:list[str], json_string:str)->dict | str:    
        # pattern of general json string with unescaped " in values keys from keys list
        pattern = ''
        for i, k in enumerate(arguments):
            pattern += f'"{k}": (.*)'
            if i != len(arguments) - 1:
                pattern += r',\s*'

        match=re.search(pattern, json_string)

        if not match:
            return f"Error: {json_string} can not match pattern {pattern}"
        
        result_json={}
        for i in range(len(arguments)):
            value=match.group(i+1)
            value=value.strip()
            if value.startswith('"') and value.endswith('"'):
                value=value[1:-1]
            #value=value.replace('"', '\\"')
            value=value.replace('\\n','\n')
            result_json[arguments[i]]=value
        return result_json
    
    
    

    
    @classmethod
    def inference(cls, messages: List[Dict[str, Any]], run_id: str = "1",return_json:bool=False) -> dict:
        """Prod inference with caching"""
        # Build request data
        cleaned_msgs: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue  # skip anything non-standard
            content = m.get("content", "")

            # Ignore assistant placeholders that only carry the internal
            # ``tool_call`` and have no visible content.
            if role == "assistant" and not content.strip():
                continue

            cleaned_msgs.append({"role": role, "content": content})

        if not cleaned_msgs:
            raise RuntimeError("No valid messages to send to proxy.")

        next_thought,next_tool_name,next_tool_args,raw_text,total_attempts,error_counter,messages = cls._request_next_action_with_retry(cleaned_msgs)
        
        return next_thought,next_tool_name,next_tool_args,raw_text,total_attempts,error_counter,messages
    
    @classmethod
    def sanitise_text_resp(cls,text_resp:str)->str:
        # remove all leading and trailing quotes
        text_resp=re.sub("[\'\"]*next_thought[\'\"]*:","next_thought:",text_resp)
        text_resp=re.sub("[\'\"]*next_tool_name[\'\"]*:","next_tool_name:",text_resp)
        text_resp=re.sub("[\'\"]*next_tool_args[\'\"]*:","next_tool_args:",text_resp)
        text_resp=re.sub("[\'\"]*observation[\'\"]*:","observation:",text_resp)
        if "next_thought" not in text_resp and "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_tool_name:")<text_resp.find("next_tool_args:") and text_resp.find("next_tool_name:")>10:
            logger.info(f"next_thought not found in {text_resp[:50]}, adding it")
            text_resp="next_thought: "+text_resp
        if "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_tool_name:")<text_resp.find("next_tool_args:"):
            # remove all leading and trailing quotes in tool_name
            next_tool_name=text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n").strip("\'").strip("\"").strip()
            logger.info(text_resp)
            text_resp=re.sub(f"next_tool_name:[\'\" ]*{next_tool_name}[\'\" ]*","next_tool_name: "+next_tool_name,text_resp)
        
        return text_resp
    
    @classmethod
    def parse_response(cls,text_resp: str)->tuple[str, str, dict]:
        error_msg=None
        text_resp = text_resp.strip()
        text_resp=text_resp.split("observation:")[0]
        text_resp=text_resp.strip().strip("\n")
        text_resp=cls.sanitise_text_resp(text_resp)
        if "next_thought:" in text_resp and "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_thought:")<text_resp.find("next_tool_name:") and text_resp.find("next_tool_name:")<text_resp.find("next_tool_args:"):
            next_thought=text_resp.split("next_thought:")[1].split("next_tool_name:")[0].strip().strip("\n")
            next_tool_name=text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n")
            next_tool_args=text_resp.split("next_tool_args:")[1].strip().split("next_thought:")[0].strip().strip("\n")
            try:
                next_tool_args=cls.parse_next_tool_args(next_tool_name, next_tool_args)
            except JSONDecodeError as e:
                error_msg=f"Invalid JSON: {str(e)}"
                Utils.log_to_failed_messages(text_resp)
                
        else:
            if "next_thought:" not in text_resp:
                error_msg="Invalid response. next_thought not found"
            elif "next_tool_name:" not in text_resp:
                error_msg="Invalid response. next_tool_name not found"
            elif "next_tool_args:" not in text_resp:
                error_msg="Invalid response. next_tool_args not found"
            elif text_resp.find("next_thought:")>text_resp.find("next_tool_name:"):
                error_msg="Invalid response. next_thought is after next_tool_name"
            elif text_resp.find("next_tool_name:")>text_resp.find("next_tool_args:"):
                error_msg="Invalid response. next_tool_name is after next_tool_args"
            else:
                logger.error(f"We have no clue why parsing failed. Please check this \n{text_resp}\n")
                error_msg=f"Invalid response. Please follow the response format "
            return None,None,None,error_msg

        return next_thought, next_tool_name, next_tool_args,error_msg
    
    @classmethod
    def _remote_embed(cls,text: str) -> List[float]:
        """Return embedding vector for *text* via the proxy /api/embedding endpoint.

        Caches results in-memory to avoid duplicate HTTP calls.
        """
        print(f"[Agent] Embedding request: {len(text)} chars")
        # Short-circuit empty or whitespace-only inputs.
        if not text.strip():
            return cls._EMBED_CACHE.setdefault("", [0.0] * 1024)

        # Retry–shrink loop to satisfy 512-token limit
        attempt_text = text
        for _ in range(2):  # original + 1 retry after halving
            tokens = attempt_text.split()
            if len(tokens) > MAX_EMBED_TOKENS:
                attempt_text = " ".join(tokens[:MAX_EMBED_TOKENS])

            url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/embedding"
            req = _urlreq.Request(
                url,
                data=json.dumps({"input": attempt_text, "run_id": RUN_ID}, ensure_ascii=False).encode(),
                method="POST",
                headers={"Content-Type": "application/json"},
            )

            try:
                with _urlreq.urlopen(req, timeout=60) as resp:  # Reduced timeout from 300 to 60
                    data_raw = resp.read()
                    data = json.loads(data_raw.decode())
                    print(f"[Agent] Embedding response: {len(data)} bytes")

                    if isinstance(data, list):
                        vec = data[0] if (len(data) == 1 and isinstance(data[0], list)) else data
                        cls._EMBED_CACHE[text] = vec
                        return vec
                    if isinstance(data, dict) and "embedding" in data:
                        vec = data["embedding"]
                        cls._EMBED_CACHE[text] = vec
                        return vec

                    # If we received a validation error about tokens, halve and retry
                    if isinstance(data, dict) and data.get("error_type") == "Validation":
                        attempt_text = " ".join(tokens[: len(tokens) // 2])
                        continue

                    return cls.ZERO_VEC
            except Exception as e:
                print(f"[Agent] Embedding error: {e}")
                return cls.ZERO_VEC

        return cls.ZERO_VEC
    
    @classmethod
    def safe_remote_embed(cls,text, max_retries=3):  # Reduced retries from 5 to 3
        for attempt in range(max_retries):
            try:
                return cls._remote_embed(text)
            except Exception as e:
                sleep_time = 2  # Reduced sleep time from 10 to 2 seconds
                print(f"Rate limited, retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        return cls.ZERO_VEC


class FixTaskEnhancedToolManager:
    generated_test_files=[]
    class Utils:
        @staticmethod
        def limit_strings(strings: str, n=1000)->str:
            '''
            Limit the number of strings to 1000
            '''
            strings_list=strings.split("\n")
            if len(strings_list)>n:
                return "\n".join(strings_list[:n])+"\n..." + f"({len(strings_list)-n} more lines)"
            else:
                return strings
    
    @staticmethod
    def check_syntax_error(content:str,file_path:str="<unknown>")->bool:
            try:
                ast.parse(content, filename=file_path)
                return False, None
            except SyntaxError as e:
                logger.error(f"Syntax error: {e}")
                return True, "Syntax error. "+str(e)

    @staticmethod
    def _get_file_content(file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None,limit:int=5000)->str:
        if search_term is not None and search_term!="":
            logger.debug(f"search_term specified: {search_term}, searching in v2")
            return FixTaskEnhancedToolManager.search_in_specified_file_v2(file_path, search_term)
            
        # check if start and end line are not between a function..
        func_ranges=FixTaskEnhancedToolManager.get_function_ranges(file_path)
        if search_start_line!=None:
            for start, end, name in func_ranges:
                if start<=search_start_line<=end:
                    if start<search_start_line:
                        logger.debug(f"search start line {search_start_line} is between a function {start}-{end} for function {name}, setting to {start}")
                        search_start_line=start
        if search_end_line!=None:
            for start, end, name in func_ranges:
                if start<=search_end_line<=end:
                    if end>search_end_line:
                        logger.debug(f"search end line {search_end_line} is between a function {start}-{end} for function {name}, setting to {end}")
                        search_end_line=end
        logger.debug(f"search start line: {search_start_line}, search end line: {search_end_line}")
        with open(file_path, "r") as f:
            if search_start_line is not None or search_end_line is not None:
                lines = f.readlines()
                start = max(0, (search_start_line or 1) - 1)  # Convert to 0-based
                end = min(len(lines), search_end_line or len(lines))
                content = ''.join(lines[start:end])
                return f"Lines {start+1}-{end} of {file_path}:\n{content}"
            else:
                content = f.read()

        return FixTaskEnhancedToolManager.Utils.limit_strings(content, n=limit) if limit!=-1  else content
    
    def get_file_content(file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None)->str:
       
        '''
        Retrieves file contents with optional filtering based on search term and line numbers
        Arguments:
            file_path: filesystem path to target file. This file must be python file.
            search_start_line: optional start line number to begin extraction (1-indexed)
            search_end_line: optional end line number to end extraction (1-indexed)
            search_term: optional text pattern to filter matching lines
        '''
        return FixTaskEnhancedToolManager._get_file_content(file_path,search_start_line,search_end_line,search_term,limit=5000)
    def save_test_file(file_path:str,content:str)->str:
        '''
        Writes text content to specified filesystem location. If there are any syntax errors in the code, it rejects the edit with an error message.
        Arguments:
            file_path: target filesystem path
            content: text data to write
        '''
        is_error, error = FixTaskEnhancedToolManager.check_syntax_error(content)
        if is_error:
            return f"Error: {error}"
        if "__main__" not in content or "unittest" not in content:
            return """Incorrect test file. You must include __main__ and unittest in the file. Follow the format as below:-
        import unittest
        from main_module import (
            main_func
        )

        class TestFuncA(unittest.TestCase):
            def test_main_func(self):
                self.assertEqual(main_func(), "expected_output")

        if __name__ == "__main__":
            unittest.main()
        """
        return FixTaskEnhancedToolManager._save(file_path, content)
        
            
    def save_file(file_path: str, content: str)->str:
        '''
        Writes text content to specified filesystem location. If there are any syntax errors in the code, it rejects the edit with an error message. Do not use this tool to create test or files to reproduce the error.
        Arguments:
            file_path: target filesystem path
            content: text data to write
        '''
        file_name=os.path.basename(file_path)
        if ("test" in file_name.lower() or "reproduce" in file_name.lower()) and False:
            logger.error(f"Error: You cannot use this tool to create test or files to reproduce the error.")
            return "Error: You cannot use this tool to create test or files to reproduce the error."
        return FixTaskEnhancedToolManager._save(file_path, content)
    
    def get_approval_for_solution(solutions:list[str],selected_solution:int,reason_for_selection:str)->str:
        '''
        This tool is used to get approval for your proposed solution. You need to propose at least 2 meaningfully different and elegant solutions to the problem.
        While all the solutions proposed needs to be accurate, but following are guidelines for selecting the best solution:
        1. Expected output should be closest to the most relevant test case.
        Arguments:
            solutions: list of solutions proposed by you. Here each solution individually should be very detailed and then must explain why they are better than the other solutions.
            selected_solution: Index of the solution you think is the best.
            reason_for_selection: Reason for selecting the solution over other solutions.
            
        Output:
            approval: approved/not approved. If approved, you can go ahead and implement the solution.
        '''
        global IS_SOLUTION_APPROVED
        
        logger.info(f"len(solutions): {len(solutions)}, solutions: {solutions}")
        logger.info(f"selected_solution: {selected_solution}")
        logger.info(f"reason_for_selection: {reason_for_selection}")
        parsed_solutions = []
        for solution in solutions:
            sols = re.split(r"(Solution \d+:)", solution)
            logger.info(f"sols: {sols}")
            sols = [f"{sols[i]}{sols[i+1]}" for i in range(1, len(sols), 2)]  # Combine the split parts correctly
            parsed_solutions.extend(sols)
        if len(parsed_solutions)>=2:
            solutions = parsed_solutions

        if type(solutions) is not list or len(solutions)<2:
            return "Error: solutions must be a list with length at least 2."

        IS_SOLUTION_APPROVED = True

        return "Approved"
          
    def _save(file_path: str, content: str)->str:
        is_syntax_error, error = FixTaskEnhancedToolManager.check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            logger.info(f"File {file_path} saved successfully")
            return f"File {file_path} saved successfully"
        else:
            logger.info(f"Error saving file. {error}")
            logger.error(f"Error saving file. {error}")
            return f"Error saving file. {error}"
 
    def get_functions(function_paths: List[str]) -> Dict[str, str]:
        '''
        Get functions from a list of function paths.
        Arguments:
            function_paths: list of function paths (e.g. ["folder1/file1.py::class1::function1", "folder2/file2.py::class2::function2"])
        Output:
            dictionary of functions with function paths as keys and function bodies as values
        '''
        functions = {}
        for function_path in function_paths:
            parts = function_path.split("::")
            file_path = parts[0]
            function_name = "::".join(parts[1:])
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content, filename=file_path)
                visitor = FunctionVisitor(content)
                visitor.visit(tree)
                
                if function_name in visitor.functions:
                    functions[function_path] = visitor.functions[function_name].get("body", "")
                else:
                    functions[function_path] = f"Function {function_name} not found in {file_path}"
            except FileNotFoundError:
                functions[function_path] = f"File {file_path} not found"
            except Exception as e:
                functions[function_path] = f"Error processing {file_path}: {str(e)}"

        return functions

    def get_classes(class_paths: List[str])->Dict[str, str]:
        '''
        Get classes from a list of class paths.
        Arguments:
            class_paths: list of class paths (e.g. ["folder1/file1.py::class1", "folder2/file2.py::class2"])
        Output:
            dictionary of classes with class paths as keys and class bodies as values
        '''
        classes = {}
        for class_path in class_paths:
            parts = class_path.split("::")
            file_path = parts[0]
            class_name = "::".join(parts[1:])
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content, filename=file_path)
                visitor = ClassVisitor(content)
                visitor.visit(tree)
                if class_name in visitor.classes:
                    classes[class_path] = visitor.classes[class_name].get("body", "")
                else:
                    classes[class_path] = f"Class {class_name} not found in {file_path}"
            except FileNotFoundError:
                classes[class_path] = f"File {file_path} not found"
            except Exception as e:
                classes[class_path] = f"Error processing {file_path}: {str(e)}"

        return classes

    def search_in_all_files_content(search_term: str, case_sensitive: bool = False) -> str:
        '''
        Search for a text pattern across all .py files in the project, excluding any file with "test" in its path.
        Use at the beginning of the workflow to locate all possible references to a function, class, or variable.
        If more context is needed (e.g., surrounding functions, classes, etc.), follow up with get_classes or get_functions.

        Arguments:
            search_term: text pattern to locate (e.g., "def test_function", "*SomeClass*")
            case_sensitive: flag to determine if the search should be case-sensitive
        Output:
            locations where pattern was found with file paths and line numbers
        '''
        logger.info("tool called: search_in_all_files_content")
        output = []
        search_flags = 0 if case_sensitive else re.IGNORECASE

        # Walk through all directories and find Python files
        for root, _, files in os.walk("."):
            # Skip .git and docs directories
            if ".git" in root or "docs" in root:
                continue

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)

                    # Always check if search term is in the file name
                    if re.search(search_term, file_path, search_flags):
                        output.append(f"{file_path} | Filename match")

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        if not re.search(search_term, content, search_flags):
                            continue

                        # Parse the file content using AST
                        tree = ast.parse(content, filename=file_path)
                        visitor = FunctionVisitor(content)
                        visitor.visit(tree)

                        for function_name, function_info in visitor.functions.items():
                            body = function_info["body"]
                            if re.search(search_term, body, search_flags):
                                lines = body.split("\n")
                                for idx, line in enumerate(lines):
                                    if re.search(search_term, line, search_flags):
                                        line_number = function_info["line_number"] + idx
                                        output.append(f"{file_path}:{line_number} | {function_name} | {line.rstrip()}")
                    except Exception as e:
                        logger.error(f"Error searching in file {file_path} with search term {search_term}: {e}")

        output = FixTaskEnhancedToolManager.Utils.limit_strings("\n".join(output), n=100)
        if not output:
            return f"'{search_term}' not found in the codebase."
        return output

    def get_function_ranges(file_path: str)->list[tuple[int, int, str]]:
        # Try to parse the file to map lines to their enclosing functions.
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            return f"Error reading '{file_path}': {e}"
        try:
            tree = ast.parse("\n".join(source_lines), filename=file_path)
        except SyntaxError as e:
            return f"Error parsing '{file_path}': {e}, {traceback.format_exc()}"

        func_ranges: list[tuple[int, int, str]] = []  # (start, end, name)
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = getattr(node, 'lineno', None)
                    end = getattr(node, 'end_lineno', None)
                    if start is not None and end is not None:
                        func_ranges.append((start, end, node.name))
        return func_ranges

    def _extract_function_matches(file_path: str, search_term: str, *, max_output_lines: int = 1000) -> str:
        '''
        Return the source code of any function definitions that contain `search_term`.
        If a match occurs outside of a function, only that line is returned. The final
        output is truncated with `limit_strings` to avoid excessive verbosity.
        '''
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            logger.error(f"Error reading '{file_path}': {e}")
            return f"Error reading '{file_path}': {e}"

        # Identify all lines that contain the search term.
        match_lines = [idx + 1 for idx, line in enumerate(source_lines) if search_term in line]
        if not match_lines:
            return f"'{search_term}' not found in file '{file_path}'"

        func_ranges=FixTaskEnhancedToolManager.get_function_ranges(file_path)

        def _containing_function(line_no: int):
            for start, end, name in func_ranges:
                if start <= line_no <= end:
                    return (start, end, name)
            return None

        functions_to_return: list[tuple[int, int, str]] = []
        standalone_lines: list[int] = []
        for ln in match_lines:
            info = _containing_function(ln)
            if info and info not in functions_to_return:
                functions_to_return.append(info)
            elif not info:
                standalone_lines.append(ln)

        chunks: list[str] = []
        for start, end, name in functions_to_return:
            func_src = "\n".join(source_lines[start - 1:end])
            chunks.append(f"(lines {start}-{end}):\n{func_src}")

        for ln in standalone_lines:
            chunks.append(f"{ln}:{source_lines[ln - 1]}")

        return FixTaskEnhancedToolManager.Utils.limit_strings("\n\n".join(chunks), n=max_output_lines)

    def search_in_specified_file_v2(file_path: str, search_term: str)->str:
        '''
        Locates text patterns within a specific file
        Arguments:
            file_path: target file for pattern matching. This file must be python file.
            search_term: text pattern to find (e.g., "def test_function", "*SomeClass*")
        Output:
            matching locations with line numbers, or error description
        '''
        if not file_path.endswith(".py"):
            return f"Error: file '{file_path}' is not a python file."
        return FixTaskEnhancedToolManager._extract_function_matches(file_path, search_term)

    def start_over(problem_with_old_approach:str,new_apprach_to_try:str):
        '''
        This will revert any changes made to the codebase and let's you start over. Only use this tool when you have concluded that current changes you made to the codebase are not relevant and you want to start again with new approach.
        Arguments:
            problem_with_old_approach: What you tried and what was the key issues you faced with this approach.
            new_apprach_to_try: What is the new approach you want to try and how it will fix the issues you faced earlier.
        '''    
        logger.info("============Start Over============")
        os.system("git reset --hard")
        logger.info(f"problem_with_old_approach: {problem_with_old_approach}")
        logger.info(f"new_apprach_to_try: {new_apprach_to_try}")
        logger.info("===========================")
        return "Done, codebase reverted to initial state. You can start over with new approach."
        
    def get_final_git_patch(initial_checkpoint:str=None) -> str:
        """
        Generate a clean unified diff (staged changes only) that tools like `patch`
        or `git apply` can consume.
        """
        try:
            # Stage modified/untracked files with desired extensions, excluding agent files.
            exts = (".py", ".ini", ".cfg", ".toml")
            exclude = {"src/agent.py", "src/agent_runner.py"}
            # Exclude any generated test files or files modified via test generation tool
            try:
                for _p in getattr(FixTaskEnhancedToolManager, "generated_test_files", []):
                    # store as relative paths similar to git ls-files output
                    exclude.add(os.path.relpath(_p))
            except Exception:
                pass

            # Discover modified + untracked files
            ls = subprocess.run(
                ["git", "ls-files", "-m", "-o", "--exclude-standard"],
                capture_output=True, text=True, timeout=30, check=True
            ).stdout.splitlines()

            to_add = [f for f in ls if f.endswith(exts) and f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)

            # Produce a clean, parseable patch (no colors; standard unified diff).
            # If initial_checkpoint is provided, diff from that checkpoint to index (HEAD + staged changes)
            if initial_checkpoint:
                diff = subprocess.run(
                    ["git", "diff", "--cached", initial_checkpoint, "--no-color", "--unified=3"],
                    capture_output=True, text=True, timeout=30, check=True
                )
            else:
                diff = subprocess.run(
                    ["git", "diff", "--cached", "--no-color", "--unified=3"],
                    capture_output=True, text=True, timeout=30, check=True
                )

            # Log stderr separately so it never pollutes the patch.
            if diff.stderr:
                logger.error("git diff (stderr): %s", diff.stderr.strip())

            patch_text = diff.stdout or ""
            if not patch_text:
                logger.error("Patch text is empty..")
                logger.error("showing git status..")
                logger.info(os.system("git status"))
                time.sleep(2)
                logger.info("trying it again..")
                diff=subprocess.run(
                    ["git", "diff", "--cached", "--no-color", "--unified=3"],
                    capture_output=True, text=True, timeout=30, check=True
                )
                logger.info(f"diff.stdout: {diff.stdout}")
                logger.info(f"diff.stderr: {diff.stderr}")
                if diff.stdout:
                    patch_text=diff.stdout
            return patch_text
        except Exception as e:
            logger.exception("Error generating git patch")
            return f"Error generating git patch: {e}"
    

    

    def run_code(content:str,file_path:str)->str:
        '''
        Runs any python code. You can use this tool directly to run any test code or bug reproduction code.
        Saves the code at the given file_path and then runs it. Do not use this tool to create test or files to reproduce the error unless user has specifically asked you to create test files as part of problem statement.

        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in. This file should always be in the current working directory.

        Output:
            Returns the stdout/stderr from the executed file.
            Returns error message if there are any third party dependencies.
        '''
        
        is_syntax_error,error=FixTaskEnhancedToolManager.check_syntax_error(content)
        if is_syntax_error:
            logger.error(f"Error: syntax error. {error}")
            return f"Error: syntax error. {error}"
        FixTaskEnhancedToolManager._save(file_path, content)
        # Normalize the path to avoid duplicates like tests/abc.py vs ./tests/abc.py
        FixTaskEnhancedToolManager.generated_test_files.append(os.path.normpath(file_path))
        # Parse the file's AST to collect import statements
        
        with open(file_path, "r") as f:
            tree = ast.parse(f.read(), filename=file_path)

        disallowed_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Use the module specified in 'from x import y' if available;
                # otherwise fall back to the imported name from plain 'import x'
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module.split(".")[0]
                else:
                    mod = node.names[0].name.split(".")[0]

                # Skip if built-in module
                if mod in sys.builtin_module_names:
                    continue

               

                # Skip relative imports ("from . import foo") which have level > 0
                if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                    continue

                # --- Additional check: allow local modules/packages in CWD ---
                cwd = os.getcwd()
                local_file = os.path.join(cwd, f"{mod}.py")
                local_pkg_init = os.path.join(cwd, mod, "__init__.py")
                local_pkg_dir = os.path.join(cwd, mod)
                # Also check inside a conventional 'lib' folder within cwd
                lib_dir = os.path.join(cwd, 'lib')
                lib_file = os.path.join(lib_dir, f"{mod}.py")
                lib_pkg_init = os.path.join(lib_dir, mod, "__init__.py")
                lib_pkg_dir = os.path.join(lib_dir, mod)

                if (
                    os.path.isfile(local_file)
                    or os.path.isfile(local_pkg_init)
                    or os.path.isdir(local_pkg_dir)
                    or os.path.isfile(lib_file)
                    or os.path.isfile(lib_pkg_init)
                    or os.path.isdir(lib_pkg_dir)
                ):
                    # Treat as local dependency, allow it
                    continue

                # Any other module is considered disallowed
                disallowed_modules.add(mod)

        if disallowed_modules and False:
            logger.error(f"Cannot run, third party dependencies detected: {sorted(disallowed_modules)}\n")
            raise ToolManager.Error(ToolManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES.name,f"Error:Cannot run, third party dependencies detected: {sorted(disallowed_modules)}\n")

        
        result = subprocess.run(["python", file_path], capture_output=True, text=True, check=False, timeout=60)
        if result.returncode!=0:
            logger.error(f"Error running code: {result.stderr}\n")
            return f"Error running code: {result.stderr}\n"
        observation = f"{result.stdout}\n"
        if result.stderr:
            observation += f"\nSTDERR: {result.stderr}"
        logger.info(f"output: {observation}")

        return observation
    
    def run_python_file(file_path:str)->str:
        '''
        Runs any python file. 

        Arguments:
            file_path: path of the python file to run. This file should always be in the current working directory.

        Output:
            Returns the stdout/stderr from the executed file.
            Returns error message if file does not exist.
        '''
        
        if not os.path.exists(file_path):
            return f"Error: file '{file_path}' does not exist."
        result = subprocess.run(["python", file_path], capture_output=True, text=True, check=False, timeout=60)
        if result.returncode!=0:
            logger.error(f"Error running code: {result.stderr}\n")
            return f"Error running code: {result.stderr}\n"
        observation = f"{result.stdout}\n"
        if result.stderr:
            observation += f"\nSTDERR: {result.stderr}"
        logger.info(f"output: {observation}")

        return observation
    
    def apply_code_edit(file_path:str, search:str, replace:str)->str:
        '''
        Performs targeted text replacement within source files. If there are any syntax errors in the code, it rejects the edit with an error message. Please note use you can only use this tool after you have approval from user on your proposed solution.
        Arguments:
        file_path: target file for modification
        search: exact text pattern to locate and replace
        replace: new text content to substitute
            
        Output:
            operation status - success confirmation or detailed error with guidance
        '''
        if not IS_SOLUTION_APPROVED:
            logger.error(f"Error: You cannot use this tool before you have approval from user on your proposed solution. Please call get_approval_for_solution tool first with list of proposed solutions.")
            return "Error: You cannot use this tool before you have approval from user on your proposed solution. Please call get_approval_for_solution tool first with list of proposed solutions."
        if not os.path.exists(file_path):
            logger.error(f"file '{file_path}' does not exist.")
            return f"Error: file '{file_path}' does not exist."
        
        original=FixTaskEnhancedToolManager._get_file_content(file_path,limit=-1)

        match original.count(search):
            case 0:
                logger.error(f"search string not found in file {file_path}. You need to share the exact code you want to replace.")
                return f"Error: search string not found in file {file_path}. You need to share the exact code you want to replace."
            case 1:
                
                new_content = original.replace(search, replace)
                try:
                        is_error,error=FixTaskEnhancedToolManager.check_syntax_error(new_content)
                        if not is_error:
                            out=FixTaskEnhancedToolManager.save_file(file_path, new_content)
                            logger.info(f"file saved output: {out}")
                            logger.info(f"ok, code edit applied successfully")
                            return "ok, code edit applied successfully"
                        else:
                            logger.error(f"Error: code edit failed. {error}")
                            return f"Error: code edit failed. {error}"
                except Exception as e:
                    logger.error(f"Error: syntax error in file {file_path}. {e}")
                    return f"Error: syntax error in file {file_path}. {e}"
            case num_hits:
                logger.error(f"search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change.")
                return f"Error: search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change."
    
    def finish(investigation_summary: str):
        '''
        Signals completion of the current workflow execution
        Arguments:
            investigation_summary: Please provide a detailed summary of the findings from your investigation and detailed solution to the problem.Use the following format:
                Problem: <problem_statement>
                Investigation: <investigation_summary>
                Solution: <your solution>
        '''
        qa_response={"is_patch_correct":"yes"}
        if qa_response.get("is_patch_correct","no").lower()=="yes":
            return "finish"
        else: 
            return "Bug reported"
        
    def run_repo_tests(file_paths:List[str])->str:
        '''
        Runs the tests for the repository. This tool will only run the tests for the files provided.
        Arguments:
            file_paths: path of the files to run the tests for.
        Output:
            Returns the stdout/stderr from the executed files.
        '''
        if not TestModeDetector.TEST_RUNNER or not TestModeDetector.TEST_RUNNER_MODE:
            thread=threading.Thread(target=asyncio.run,args=(TestModeDetector.get_test_runner_and_mode(),))
            thread.start()
            thread.join()
        test_runner,test_runner_mode=TestModeDetector.TEST_RUNNER,TestModeDetector.TEST_RUNNER_MODE
        logger.info(f"test_runner: {test_runner}, test_runner_mode: {test_runner_mode}")
        if test_runner == "pytest":
            logger.info("CMD: pytest ", file_paths)
            result = subprocess.run(["pytest"] + file_paths, shell=True, capture_output=True, text=True, timeout=90)
            output = (result.stdout or "") + (result.stderr or "")
        else:
            if test_runner_mode == "MODULE":
                modules = [FixTaskEnhancedToolManager.filepath_to_module(f, os.getcwd(), test_runner) for f in file_paths]
                
                cmd = f"PYTHONPATH=. {test_runner} {' '.join(modules)}"
                logger.info(f"CMD: {cmd}")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
                logger.info(f"output: {output}")
            else:
                files_to_test = [FixTaskEnhancedToolManager.clean_filepath(f, os.getcwd(), test_runner) for f in file_paths]
                cmd = f"{test_runner} {' '.join(files_to_test)}"
                logger.info(f"CMD: {cmd}")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
        return output
    
    def filepath_to_module(file_path: str, repo_path: str, test_runner: str) -> str:
        """Convert file path to Python module notation."""
        root_path = os.path.abspath(repo_path)
        abs_filepath = os.path.abspath(file_path)
        
        # Remove extension and make relative to repo
        module_path = os.path.splitext(abs_filepath)[0]
        if module_path.startswith(root_path):
            module_path = module_path[len(root_path):].lstrip(os.path.sep)

        # Adjust relative to test runner directory if needed
        test_runner_dir = os.path.dirname(test_runner)
        if test_runner_dir and module_path.startswith(test_runner_dir):
            module_path = module_path[len(test_runner_dir):].lstrip(os.path.sep)

        return module_path.replace(os.path.sep, '.')

    def clean_filepath(file_path: str, repo_path: str, test_runner: str) -> str:
        root_path = os.path.abspath(repo_path)
        abs_filepath = os.path.abspath(file_path)
        
        module_path = os.path.splitext(abs_filepath)[0]
        if module_path.startswith(root_path):
            module_path = module_path[len(root_path):].lstrip(os.path.sep)

        test_runner_dir = os.path.dirname(test_runner)
        if test_runner_dir and module_path.startswith(test_runner_dir):
            module_path = module_path[len(test_runner_dir):].lstrip(os.path.sep)

        return module_path
        
    def remove_any_generated_test_files():
        for file in FixTaskEnhancedToolManager.generated_test_files:
            if os.path.exists(file):
                os.remove(file)
        FixTaskEnhancedToolManager.generated_test_files=[]
        
    def generate_test_function(file_path: str, test_function_code: str, position: str = "append") -> str:
        '''
        Create or append a test function to the specified test file. Generated tests are excluded from final patch.
        Arguments:
            file_path: path to the test file to create or modify
            test_function_code: the full test function code to insert
            position: where to place the function: "append", "top", "after_imports", "before_main", or "auto"
        Output:
            Success message or error message
        '''
        if not file_path.endswith('.py'):
            logger.error(f"Error: file '{file_path}' is not a python file.")
            return f"Error: file '{file_path}' is not a python file."

        # Ensure directory exists
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # Normalize newline handling
        #test_fn = (test_function_code or "").strip()
        test_fn = test_function_code or ""
        if not test_fn:
            logger.error(f"Error: test_function_code cannot be empty.")
            return f"Error: test_function_code cannot be empty."

        is_new_file = not os.path.exists(file_path)

        def _insert_after_imports(content: str, block: str) -> str:
            lines = content.splitlines()
            insert_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    insert_idx = i + 1
                elif stripped == "" or stripped.startswith("#"):
                    # allow header comments/blank lines before imports
                    insert_idx = max(insert_idx, i + 1)
                else:
                    break
            lines = lines[:insert_idx] + (["", block, ""] if insert_idx < len(lines) else ["", block]) + lines[insert_idx:]
            return "\n".join(lines).rstrip() + "\n"

        def _insert_before_main(content: str, block: str) -> str:
            marker = "if __name__ == \"__main__\":"
            idx = content.find(marker)
            if idx == -1:
                return None
            return content[:idx].rstrip() + "\n\n" + block + "\n\n" + content[idx:]

        if is_new_file:
            new_content = test_fn + "\n"
            # Validate standalone content before writing
            is_err, err = FixTaskEnhancedToolManager.check_syntax_error(new_content)
            if is_err:
                logger.error(f"Error: generated test function has syntax error: {err}")
                return f"Error: generated test function has syntax error: {err}"
        else:
            original = FixTaskEnhancedToolManager._get_file_content(file_path, limit=-1)
            # Avoid duplicating exact same function text
            if test_fn in original:
                rel = os.path.relpath(file_path)
                if rel not in FixTaskEnhancedToolManager.generated_test_files:
                    FixTaskEnhancedToolManager.generated_test_files.append(rel)
                return f"Test already present in '{rel}', no changes made."

            # Build candidate insertion strategies in order
            candidates = []
            if position == "append":
                candidates = [lambda src: src.rstrip() + "\n\n" + test_fn + "\n"]
            elif position == "top":
                candidates = [lambda src: test_fn + "\n\n" + src]
            elif position == "after_imports":
                candidates = [lambda src: _insert_after_imports(src, test_fn)]
            elif position == "before_main":
                candidates = [lambda src: (_insert_before_main(src, test_fn) or src.rstrip() + "\n\n" + test_fn + "\n")]
            elif position == "auto":
                candidates = [
                    lambda src: (_insert_before_main(src, test_fn) or _insert_after_imports(src, test_fn)),
                    lambda src: src.rstrip() + "\n\n" + test_fn + "\n",
                    lambda src: test_fn + "\n\n" + src,
                ]
            else:
                logger.error(f"Error: invalid position '{position}'. Use 'append', 'top', 'after_imports', 'before_main', or 'auto'.")
                return f"Error: invalid position '{position}'. Use 'append', 'top', 'after_imports', 'before_main', or 'auto'."

            # Try each candidate until one passes syntax check
            new_content = None
            first_error = None
            for idx,builder in enumerate(candidates):
                
                try:
                    candidate = builder(original)
                    is_err, err = FixTaskEnhancedToolManager.check_syntax_error(candidate)
                    if not is_err:
                        new_content = candidate
                        break
                    if first_error is None:
                        first_error = err
                except Exception as e:
                    if first_error is None:
                        first_error = e
                    continue

            if new_content is None:
                logger.error(f"Error: inserting test caused syntax error. First error: {first_error}")
                return f"Error: inserting test caused syntax error. First error: {first_error}"

        FixTaskEnhancedToolManager._save(file_path, new_content)

        # Track for exclusion from final patch
        rel = os.path.relpath(file_path)
        if rel not in FixTaskEnhancedToolManager.generated_test_files:
            FixTaskEnhancedToolManager.generated_test_files.append(rel)

        return f"Test {'created' if is_new_file else 'updated'} in '{rel}' (position={position})."
    
    def parse_run_repo_tests_response(response: str,initial_response: str) -> list[str]:
        if not response:
            return []
        response=response.split("======================================================================")
        initial_response=initial_response.split("======================================================================")
        failures=[]
        for r in response:
            if "FAIL:" in r:
                failures.append(r.strip())
            if "ERROR:" in r and r not in initial_response:
                logger.error(f"found new Error: {r.strip()}")
                failures.append(r.strip())
            else:
                logger.info(f"ignoring {r.strip()} as it is already in the initial commit")
        return failures
        
class FunctionVisitor(ast.NodeVisitor):
    def __init__(self, file_content: str):
        self.functions = {}
        self.current_class = None
        self.class_hierarchy = []
        self.file_content = file_content

    def visit_ClassDef(self, node):
        self.class_hierarchy.append(node.name)
        self.current_class = "::".join(self.class_hierarchy)
        self.generic_visit(node)
        self.class_hierarchy.pop()
        self.current_class = "::".join(self.class_hierarchy) if self.class_hierarchy else None

    def _process_function(self, node):
        full_function_name = f"{self.current_class}::{node.name}" if self.current_class else node.name
        line_number = node.lineno
        if isinstance(node.decorator_list, list) and len(node.decorator_list) > 0:
            line_number = node.decorator_list[0].lineno
        
        end_line_number = line_number
        if isinstance(node.body, list) and len(node.body) > 0:
            end_line_number = node.body[-1].lineno
        
        lines = self.file_content.split("\n")
        body = "\n".join(lines[line_number-1:end_line_number])
        
        self.functions[full_function_name] = {
            "class": self.current_class,
            "body": body,
            "line_number": line_number
        }
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node):
        self._process_function(node)

    def visit_Module(self, node):
        self.current_class = None
        self.generic_visit(node)
        self.current_class = None

class ClassVisitor(ast.NodeVisitor):
    def __init__(self, file_content: str):
        self.classes = {}
        self.file_content = file_content

    def visit_ClassDef(self, node):
        line_number = node.lineno
        if isinstance(node.decorator_list, list) and len(node.decorator_list) > 0:
            line_number = node.decorator_list[0].lineno
        end_line_number = line_number
        if isinstance(node.body, list) and len(node.body) > 0:
            end_line_number = node.body[-1].lineno
        lines = self.file_content.split("\n")
        body = "\n".join(lines[line_number-1:end_line_number])
        self.classes[node.name] = {
            "body": body,
            "line_number": line_number
        }
        self.generic_visit(node)



class TestModeDetector:
    TEST_RUNNER=None
    TEST_RUNNER_MODE=None
    
    FIND_TEST_RUNNER_PROMPT = textwrap.dedent("""\
    You are a helpful assistant that can find the test runner for a given repository.
    - The test runner is the file that can run the individual test files and test cases. (e.g. pytest, unittest, etc.)
    - Do not use the test runner to run test for whole repository or test setup.
    - Read the README file and find the test runner. If there is no test runner, return pytest.
    - Output format should be as the following. No other texts are allowed.
    abc/test.py
    """)

    TEST_RUNNER_MODE_PROMPT = textwrap.dedent("""\
    You are a helpful assistant that determines the mode of the test runner.
    Read the test runner file and determine if it requires a module or a file path to run the test.
    Output should be one of MODULE or FILE, No other texts are allowed.
    - MODULE: When the test runner requires a module path to run the test.
    - FILE: When the test runner requires a file path to run the test (e.g. pytest, unittest, py.test, etc.).
    """)
    
    
    async def get_test_runner_and_mode():
        if TestModeDetector.TEST_RUNNER_MODE and TestModeDetector.TEST_RUNNER:
            return TestModeDetector.TEST_RUNNER, TestModeDetector.TEST_RUNNER_MODE
        
        test_runner = "pytest"
        test_runner_mode = "FILE"
        test_files = []  # Initialize the test_files list
        test_file_path = None
        
        for root, _, files in os.walk('.'):
            for file in files:
                if 'test_' in file and file.endswith('.py'):
                    test_files.append(os.path.join(root, file))
        
        test_files.sort(key=len)

        for path in test_files:
            if TestModeDetector.count_test_cases(path) > 5:
                test_file_path = path
                break

        if not test_file_path:
            print(f"no test file found")
            return "pytest", "FILE"

        print(f"test_file_path: {test_file_path}")
        readme_file_path = TestModeDetector.find_readme(test_file_path, '.')
        if readme_file_path:
            print(f"README found: {readme_file_path}")
            test_runner = await TestModeDetector.find_test_runner(readme_file_path)
            test_runner_mode = await TestModeDetector.get_test_runner_mode(test_runner)
        else:
            print("No README found, using default pytest")

        TestModeDetector.TEST_RUNNER = test_runner
        TestModeDetector.TEST_RUNNER_MODE = test_runner_mode
        return test_runner, test_runner_mode
    
    
    def find_readme(file_path: str, repo_path: str) -> Optional[str]:
        """Find README file by traversing up from the given path."""
        current_dir = os.path.dirname(file_path)
        
        while True:
            for readme_name in ['README.md', 'README.rst']:
                readme_path = os.path.join(current_dir, readme_name)
                if os.path.exists(readme_path):
                    return readme_path
            if current_dir == repo_path:
                break
            current_dir = os.path.dirname(current_dir)

        return None
    def count_test_cases(file_path: str) -> int:
        """Count the number of test cases (functions starting with 'test_') in a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            import re
            test_functions = re.findall(r'^\s*def\s+test_\w+', content, re.MULTILINE)
            return len(test_functions)
        
        except (FileNotFoundError, UnicodeDecodeError):
            return 0
        
    
    async def find_test_runner(readme_file_path: Optional[str] = None):
        if not readme_file_path:
            return "pytest"
        try:
            with open(readme_file_path, "r", encoding='utf-8') as f:
                readme_content = f.read()
            
            agent = _agent_pool.get_agent("test_runner", GLM_MODEL_NAME, TestModeDetector.FIND_TEST_RUNNER_PROMPT, "test_runner_agent")
            response=await agent.solve_task(readme_content,response_format="",is_json=False,regex=None,post_process_func=None,return_type=str)
            attempt=0
            while attempt<5:
                if ".py" in response.strip() and not os.path.exists(response.strip()):
                    logger.error(f"test runner file {response.strip()} not found, retrying..")
                    response=await agent.solve_task(f"the filepath {response.strip()} is not present in the repo, Carefully check the readme filepath and correctly provide the filepath. You are currently at the root of the repo.",response_format="=======THOUGHT\n<<your thought>>\n=======TEST_RUNNER\n<<your test runner filepath>>",is_json=False,regex=None,post_process_func=None,disable_reset=True,return_type=Union[str,tuple[str,str]])
                    if isinstance(response,tuple) and isinstance(response[1],str):
                        response=response[1].strip()
                    attempt+=1
                else:
                    break
            return response.strip() or "pytest"
        except Exception as e:
            logger.error(f"Error finding test runner: {e}")
            return "pytest" 
        
        
    async def get_test_runner_mode(test_runner: str):
        if test_runner == 'pytest':
            return "FILE"

        try:
            with open(test_runner, "r", encoding='utf-8') as f:
                runner_content = f.read()
            agent = _agent_pool.get_agent("test_runner_mode", GLM_MODEL_NAME, TestModeDetector.TEST_RUNNER_MODE_PROMPT, "test_runner_mode_agent")
            response=await agent.solve_task(runner_content,response_format="",is_json=False,regex=None,post_process_func=None,return_type=str)
            return response.strip() or "FILE"
        except Exception as e:
            logger.error(f"Error determining test runner mode: {e}")
            return "FILE"
        

def create_checkpoint(repo_path: str, checkpoint_name: str) -> dict:
    
    import subprocess
    import os
# Improvement v1 - 2025-10-24 23:35:24
    
    try:
        # Validate that the path is a git repository
        if not os.path.exists(os.path.join(repo_path, ".git")):
            return {
                "status": "error",
                "message": f"Not a git repository: {repo_path}"
            }
        
        # Change to repository directory
        original_dir = os.getcwd()
        os.chdir(repo_path)
        
        # Check if checkpoint name already exists
        result = subprocess.run(
            ["git", "tag", "-l", checkpoint_name],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.stdout.strip():
            os.chdir(original_dir)
            return {
                "status": "error",
                "message": f"Checkpoint '{checkpoint_name}' already exists. Use a different name or delete the existing checkpoint first."
            }
        
        # Stage all changes (including untracked files)
        subprocess.run(["git", "add", "-A"], check=True, capture_output=True)
        
        # Check if there are changes to commit
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Create commit with all changes
        if status_result.stdout.strip():
            commit_result = subprocess.run(
                ["git", "commit", "-m", f"Checkpoint: {checkpoint_name}"],
                capture_output=True,
                text=True,
                check=True
            )
        
        # Get current commit hash
        hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        commit_hash = hash_result.stdout.strip()
        
        # Create a tag for this checkpoint
        subprocess.run(
            ["git", "tag", checkpoint_name, commit_hash],
            check=True,
            capture_output=True
        )
        
        os.chdir(original_dir)
        
        return {
            "status": "success",
            "checkpoint_name": checkpoint_name,
            "commit_hash": commit_hash,
            "message": f"Checkpoint '{checkpoint_name}' created successfully at {commit_hash[:8]}"
        }
        
    except subprocess.CalledProcessError as e:
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return {
            "status": "error",
            "message": f"Git command failed: {e.stderr if e.stderr else str(e)}"
        }
    except Exception as e:
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }


def switch_checkpoint(repo_path: str, checkpoint_name: str, save_current: bool = True) -> dict:
   
    import subprocess
    import os
# Improvement v1 - 2025-10-24 23:35:24
    
    try:
        # Validate that the path is a git repository
        if not os.path.exists(os.path.join(repo_path, ".git")):
            return {
                "status": "error",
                "message": f"Not a git repository: {repo_path}"
            }
        
        # Change to repository directory
        original_dir = os.getcwd()
        os.chdir(repo_path)
        
        # Check if checkpoint exists
        tag_result = subprocess.run(
            ["git", "tag", "-l", checkpoint_name],
            capture_output=True,
            text=True,
            check=False
        )
        
        if not tag_result.stdout.strip():
            os.chdir(original_dir)
            return {
                "status": "error",
                "message": f"Checkpoint '{checkpoint_name}' not found"
            }
        
        # Save current state if requested
        if save_current:
            # Check if there are uncommitted changes
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True
            )
            
            if status_result.stdout.strip():
                # Stash current changes
                stash_result = subprocess.run(
                    ["git", "stash", "push", "-u", "-m", f"Auto-stash before switching to {checkpoint_name}"],
                    capture_output=True,
                    text=True,
                    check=True
                )
        
        # Checkout the checkpoint
        checkout_result = subprocess.run(
            ["git", "checkout", checkpoint_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Get current commit hash after checkout
        hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        commit_hash = hash_result.stdout.strip()
        
        os.chdir(original_dir)
        
        return {
            "status": "success",
            "checkpoint_name": checkpoint_name,
            "commit_hash": commit_hash,
            "message": f"Switched to checkpoint '{checkpoint_name}' at {commit_hash[:8]}",
            "stashed": save_current and status_result.stdout.strip()
        }
        
    except subprocess.CalledProcessError as e:
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return {
            "status": "error",
            "message": f"Git command failed: {e.stderr if e.stderr else str(e)}"
        }
    except Exception as e:
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }


def restore_stashed_changes(repo_path: str, stash_index: int = 0, remove_after_apply: bool = True) -> dict:
    
    import subprocess
    import os
# Improvement v1 - 2025-10-24 23:35:24
    
    try:
        # Validate that the path is a git repository
        if not os.path.exists(os.path.join(repo_path, ".git")):
            return {
                "status": "error",
                "message": f"Not a git repository: {repo_path}"
            }
        
        # Change to repository directory
        original_dir = os.getcwd()
        os.chdir(repo_path)
        
        # Check if there are any stashes
        stash_list_result = subprocess.run(
            ["git", "stash", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if not stash_list_result.stdout.strip():
            os.chdir(original_dir)
            return {
                "status": "error",
                "message": "No stashed changes found"
            }
        
        # Count number of stashes
        stash_count = len(stash_list_result.stdout.strip().split('\n'))
        
        if stash_index >= stash_count:
            os.chdir(original_dir)
            return {
                "status": "error",
                "message": f"Stash index {stash_index} out of range. Only {stash_count} stash(es) available."
            }
        
        # Apply or pop the stash
        stash_ref = f"stash@{{{stash_index}}}"
        
        if remove_after_apply:
            # Pop: apply and remove
            command = ["git", "stash", "pop", stash_ref]
            action = "popped"
        else:
            # Apply: apply but keep in stash list
            command = ["git", "stash", "apply", stash_ref]
            action = "applied"
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        
        os.chdir(original_dir)
        
        return {
            "status": "success",
            "message": f"Successfully {action} stash@{{{stash_index}}}",
            "stash_index": stash_index,
            "removed": remove_after_apply
        }
        
    except subprocess.CalledProcessError as e:
        if 'original_dir' in locals():
            os.chdir(original_dir)
        error_msg = e.stderr if e.stderr else str(e)
        return {
            "status": "error",
            "message": f"Git stash command failed: {error_msg}"
        }
    except Exception as e:
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }


def list_stashes(repo_path: str) -> dict:
    """
    List all stashed changes in the repository.
    
    Args:
        repo_path: Path to the git repository
        
    Returns:
        dict: Contains status, count, and list of stashes with their descriptions
        
    Example:
        result = list_stashes("/path/to/repo")
        if result["status"] == "success":
            print(f"Found {result['count']} stash(es)")
            for stash in result['stashes']:
                print(f"  {stash['index']}: {stash['message']}")
    """
    import subprocess
    import os
# Improvement v1 - 2025-10-24 23:35:24
    import re
    
    try:
        # Validate that the path is a git repository
        if not os.path.exists(os.path.join(repo_path, ".git")):
            return {
                "status": "error",
                "message": f"Not a git repository: {repo_path}"
            }
        
        # Change to repository directory
        original_dir = os.getcwd()
        os.chdir(repo_path)
        
        # Get stash list
        result = subprocess.run(
            ["git", "stash", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        os.chdir(original_dir)
        
        if not result.stdout.strip():
            return {
                "status": "success",
                "count": 0,
                "stashes": [],
                "message": "No stashes found"
            }
        
        # Parse stash list
        stashes = []
        for line in result.stdout.strip().split('\n'):
            # Format: stash@{0}: WIP on branch: message
            match = re.match(r'stash@\{(\d+)\}: (.+)', line)
            if match:
                stashes.append({
                    'index': int(match.group(1)),
                    'reference': f"stash@{{{match.group(1)}}}",
                    'message': match.group(2)
                })
        
        return {
            "status": "success",
            "count": len(stashes),
            "stashes": stashes,
            "message": f"Found {len(stashes)} stash(es)"
        }
        
    except subprocess.CalledProcessError as e:
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return {
            "status": "error",
            "message": f"Git command failed: {e.stderr if e.stderr else str(e)}"
        }
    except Exception as e:
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }
    