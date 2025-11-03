#!/usr/bin/env python3
"""Simple test script for Cursor Cloud Agent API.

Tests:
- POST /v0/agents (launch-an-agent)
- GET /v0/agents/{id} (check status)
- POST /v0/agents/{id}/followup (add-follow-up)
"""

import os
import sys
import json
import time
import requests


CURSOR_API_URL = os.getenv("CURSOR_API_URL", "https://api.cursor.com")
CURSOR_API_KEY = os.getenv("CURSOR_API_KEY", "key_d52e0c4d44b712e3c13ab15ee2d0713d4e683a8f4900699f210be3673e181fa3")


def test_launch_agent():
    """Test launching a simple Cursor agent."""
    if not CURSOR_API_KEY:
        print("ERROR: CURSOR_API_KEY environment variable not set")
        return None
    
    url = f"{CURSOR_API_URL.rstrip('/')}/v0/agents"
    headers = {
        "Authorization": f"Bearer {CURSOR_API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Get repository URL from git if available
    repo_url = None
    try:
        import subprocess
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            repo_url = result.stdout.strip()
            print(f"[TEST] Found git remote: {repo_url}")
    except Exception:
        pass
    
    if not repo_url:
        print("[TEST] WARNING: No git remote found. Using placeholder.")
        repo_url = "https://github.com/placeholder/repo"  # May need to be a valid repo
    
    # Simple test prompt - source is required
    payload = {
        "prompt": {
            "text": "Create a simple hello_world.py file that prints 'Hello, Cursor API!'"
        },
        "source": {
            "repository": repo_url,
        },"target": {
            "skipReviewerRequest": True,
            # "branchName": "cursor-work"
        }
    }
    
    print(f"\n[TEST] Launching agent via {url}")
    print(f"[TEST] Payload: {json.dumps(payload, indent=2)}")
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        print(f"[TEST] Response status: {resp.status_code}")
        
        # Accept both 200 and 201 as success (201 = Created)
        if resp.status_code not in (200, 201):
            print(f"[TEST] Error response: {resp.text[:500]}")
            return None
        
        data = resp.json()
        print(f"[TEST] Response keys: {list(data.keys())}")
        print(f"[TEST] Full response:\n{json.dumps(data, indent=2)}")
        
        agent_id = data.get("id") or data.get("agent_id") or data.get("agentId")
        if agent_id:
            agent_status = data.get("status", "unknown")
            print(f"[TEST] ✓ Agent launched successfully!")
            print(f"[TEST]   Agent ID: {agent_id}")
            print(f"[TEST]   Status: {agent_status}")
            if data.get("target") and data["target"].get("url"):
                print(f"[TEST]   URL: {data['target']['url']}")
            return agent_id
        else:
            print(f"[TEST] ✗ No agent ID found in response")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"[TEST] ✗ Request failed: {e}")
        return None


def test_get_agent_status(agent_id):
    """Test getting agent status."""
    if not agent_id:
        return
    
    url = f"{CURSOR_API_URL.rstrip('/')}/v0/agents/{agent_id}"
    headers = {
        "Authorization": f"Bearer {CURSOR_API_KEY}",
        "Content-Type": "application/json",
    }
    
    print(f"\n[TEST] Checking agent status via {url}")
    
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        print(f"[TEST] Response status: {resp.status_code}")
        
        if resp.status_code not in (200, 201):
            print(f"[TEST] Error response: {resp.text[:500]}")
            return
        
        data = resp.json()
        print(f"[TEST] Response keys: {list(data.keys())}")
        print(f"[TEST] Full response:\n{json.dumps(data, indent=2)}")
        
        status = data.get("status") or data.get("state") or data.get("status")
        print(f"[TEST] Agent status: {status}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"[TEST] ✗ Request failed: {e}")
        return None


def test_poll_agent(agent_id, max_polls=12):
    """Poll agent until completion."""
    if not agent_id:
        return
    
    print(f"\n[TEST] Polling agent {agent_id} (max {max_polls} polls, 5s interval)")
    
    for i in range(max_polls):
        print(f"\n[TEST] Poll {i+1}/{max_polls}...")
        data = test_get_agent_status(agent_id)
        
        if data:
            status = data.get("status") or data.get("state")
            
            if status in ("completed", "done", "success", "FINISHED"):
                print(f"[TEST] ✓ Agent completed!")
                
                # Try to extract any files/changes
                changes = data.get("changes") or data.get("files") or []
                if changes:
                    print(f"[TEST] Found {len(changes)} file changes")
                    for change in changes[:3]:  # Show first 3
                        print(f"[TEST]   - {json.dumps(change, indent=4)[:200]}")
                
                summary = data.get("summary") or data.get("message") or ""
                if summary:
                    print(f"[TEST] Summary: {summary[:300]}")
                
                return data
            
            if status in ("failed", "error", "cancelled"):
                print(f"[TEST] ✗ Agent failed with status: {status}")
                return data
        
        if i < max_polls - 1:
            time.sleep(5)
    
    print(f"[TEST] Agent did not complete within {max_polls * 5} seconds")
    return None


def test_add_followup(agent_id):
    """Test adding a follow-up to an agent."""
    if not agent_id:
        return
    
    url = f"{CURSOR_API_URL.rstrip('/')}/v0/agents/{agent_id}/followup"
    headers = {
        "Authorization": f"Bearer {CURSOR_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "prompt": {
            "text": "Also add a docstring to the function"
        },
    }
    
    print(f"\n[TEST] Adding follow-up via {url}")
    print(f"[TEST] Payload: {json.dumps(payload, indent=2)}")
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        print(f"[TEST] Response status: {resp.status_code}")
        
        if resp.status_code not in (200, 201):
            print(f"[TEST] Error response: {resp.text[:500]}")
            return None
        
        data = resp.json()
        print(f"[TEST] Response keys: {list(data.keys())}")
        print(f"[TEST] Full response:\n{json.dumps(data, indent=2)}")
        
        print(f"[TEST] ✓ Follow-up added successfully")
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"[TEST] ✗ Request failed: {e}")
        return None


def test_delete_agent(agent_id):
    """Test deleting an agent."""
    if not agent_id:
        return False
    
    url = f"{CURSOR_API_URL.rstrip('/')}/v0/agents/{agent_id}"
    headers = {
        "Authorization": f"Bearer {CURSOR_API_KEY}",
        "Content-Type": "application/json",
    }
    
    print(f"\n[TEST] Deleting agent via DELETE {url}")
    
    try:
        resp = requests.delete(url, headers=headers, timeout=30)
        print(f"[TEST] Response status: {resp.status_code}")
        
        if resp.status_code in (200, 204, 201):
            print(f"[TEST] ✓ Agent deleted successfully")
            if resp.text:
                try:
                    data = resp.json()
                    print(f"[TEST] Response: {json.dumps(data, indent=2)}")
                except:
                    print(f"[TEST] Response: {resp.text[:200]}")
            return True
        else:
            print(f"[TEST] ✗ Delete failed: {resp.text[:500]}")
            return False
        
    except requests.exceptions.RequestException as e:
        print(f"[TEST] ✗ Request failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Cursor Cloud Agent API Test")
    print("=" * 60)
    print(f"API URL: {CURSOR_API_URL}")
    print(f"API Key: {'*' * 20 if CURSOR_API_KEY else 'NOT SET'}")
    
    # Test 1: Launch agent
    agent_id = test_launch_agent()
    
    if not agent_id:
        print("\n[TEST] ✗ Failed to launch agent. Cannot continue with other tests.")
        sys.exit(1)
    
    # Test 2: Check status immediately
    print("\n" + "=" * 60)
    print("Test 2: Get initial status")
    print("=" * 60)
    test_get_agent_status(agent_id)
    
    # Test 3: Poll until completion
    print("\n" + "=" * 60)
    print("Test 3: Poll until completion")
    print("=" * 60)
    final_data = test_poll_agent(agent_id)
    
    # Test 4: Add follow-up (only if agent is still active)
    print("\n" + "=" * 60)
    print("Test 4: Add follow-up")
    print("=" * 60)
    if final_data:
        test_add_followup(agent_id)
        test_poll_agent(agent_id)
       
    # Test 5: Delete agent
    print("\n" + "=" * 60)
    print("Test 5: Delete agent")
    print("=" * 60)
    test_delete_agent(agent_id)
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

