import sys
import json
import time
import traceback
import importlib.util


def main():
    print("[AGENT_RUNNER] Entered main()")

    time.sleep(3)

    try:
        # Read input.json
        print("[AGENT_RUNNER] Reading input.json")
        with open("/sandbox/input.json", "r") as f:
            input_data = json.load(f)
        print("[AGENT_RUNNER] Read input.json")
        
        # Import agent module
        print("[AGENT_RUNNER] Loading /sandbox/agent.py")
        spec = importlib.util.spec_from_file_location("agent", "/sandbox/agent.py")
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        print("[AGENT_RUNNER] Loaded /sandbox/agent.py")
        
        # Check for the agent_main() function in /sandbox/agent.py
        if hasattr(agent_module, "agent_main"):
            print("[AGENT_RUNNER] agent_main() function found in /sandbox/agent.py")
        else:
            print("[AGENT_RUNNER] agent_main() function not found in /sandbox/agent.py")
            raise Exception("agent_main() function not found in /sandbox/agent.py")
        
        # Invoke agent_main function
        print("[AGENT_RUNNER] Entering agent's agent_main()")
        agent_main_return_value = agent_module.agent_main(input_data)
        print("[AGENT_RUNNER] Exited agent's agent_main()")

        # Handle both string and dict returns (production compatibility)
        if isinstance(agent_main_return_value, str):
            patch = agent_main_return_value
            print("[AGENT_RUNNER] Agent returned string patch")
        elif isinstance(agent_main_return_value, dict) and "patch" in agent_main_return_value:
            patch = agent_main_return_value["patch"]
            print("[AGENT_RUNNER] Agent returned dict with patch key")
        else:
            raise Exception(f"agent_main() function returned invalid value: {type(agent_main_return_value)}. Expected string or dict with 'patch' key.")

        # Ensure patch is a string
        if not isinstance(patch, str):
            raise Exception(f"Patch must be a string, got {type(patch)}")

        output = {
            "success": True,
            "output": patch
        }

        print("[AGENT_RUNNER] Writing output.json")
        with open("/sandbox/output.json", "w") as f:
            json.dump(output, f, indent=2)
        print("[AGENT_RUNNER] Wrote output.json")
        
    except Exception as e:
        print("[AGENT_RUNNER] Exception:")
        traceback.print_exc(file=sys.stdout)
        
        output = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        
        try:
            print("[AGENT_RUNNER] Writing output.json")
            with open("/sandbox/output.json", "w") as f:
                json.dump(output, f, indent=2)
            print("[AGENT_RUNNER] Wrote output.json")
        except:
            print("[AGENT_RUNNER] Failed to write output.json")
            pass

    print("[AGENT_RUNNER] Exiting main()")



if __name__ == "__main__":
    main()