"""Cursor Cloud Agent API Client.

Utility class for interacting with Cursor's Cloud Agent API:
- Launch agents
- Add follow-ups
- Poll status
- Extract results

Reference: https://cursor.com/docs/cloud-agent/api/endpoints
"""

import os
import json
import time
import re
import subprocess
from typing import Dict, Any, List, Optional, Callable
import requests


class CursorAPIClient:
    """Client for Cursor Cloud Agent API operations."""
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_repo_url: Optional[str] = None,
    ):
        """Initialize Cursor API client.
        
        Args:
            api_url: Base URL for Cursor API (defaults to env CURSOR_API_URL or https://api.cursor.com)
            api_key: API key for authentication (defaults to env CURSOR_API_KEY)
            default_repo_url: Default repository URL if not provided in launch requests
        """
        self.api_url = (api_url or os.getenv("CURSOR_API_URL", "https://api.cursor.com")).rstrip("/")
        self.api_key = api_key or os.getenv("CURSOR_API_KEY", "")
        self.default_repo_url = default_repo_url or self._detect_git_remote()
        
        if not self.api_key:
            raise ValueError("CURSOR_API_KEY must be provided or set in environment")
    
    def _detect_git_remote(self, cwd: str = ".") -> Optional[str]:
        """Detect git remote URL from current directory."""
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def _headers(self) -> Dict[str, str]:
        """Get default headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def launch_agent(
        self,
        prompt_text: str,
        repository: Optional[str] = None,
        branch_name: Optional[str] = None,
        skip_reviewer_request: bool = True,
        auto_create_pr: bool = False,
    ) -> Dict[str, Any]:
        """Launch a Cursor agent.
        
        Args:
            prompt_text: The task/prompt for the agent
            repository: Repository URL (defaults to detected git remote or default_repo_url)
            branch_name: Branch name for the agent's work (defaults to auto-generated)
            skip_reviewer_request: Whether to skip reviewer requests
            auto_create_pr: Whether to automatically create a PR when done
            
        Returns:
            Dictionary with agent details including 'id', 'status', etc.
            
        Raises:
            requests.RequestException: If the API request fails
        """
        repo_url = repository or self.default_repo_url
        if not repo_url:
            raise ValueError("Repository URL required (provide or set default_repo_url)")
        
        url = f"{self.api_url}/v0/agents"
        payload = {
            "prompt": {
                "text": prompt_text,
            },
            "source": {
                "repository": repo_url,
            },
            "target": {
                "skipReviewerRequest": skip_reviewer_request,
                "autoCreatePr": auto_create_pr,
            },
        }
        
        if branch_name:
            payload["target"]["branchName"] = branch_name
            payload["source"]["ref"] = branch_name
        
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=60)
        
        if resp.status_code not in (200, 201):
            error_msg = resp.text[:500] if resp.text else "Unknown error"
            raise requests.exceptions.RequestException(
                f"Failed to launch agent: HTTP {resp.status_code} - {error_msg}"
            )
        
        return resp.json()
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get current status of an agent.
        
        Args:
            agent_id: The agent ID
            
        Returns:
            Dictionary with agent status and details
            
        Raises:
            requests.RequestException: If the API request fails
        """
        url = f"{self.api_url}/v0/agents/{agent_id}"
        resp = requests.get(url, headers=self._headers(), timeout=30)
        
        if resp.status_code not in (200, 201):
            error_msg = resp.text[:500] if resp.text else "Unknown error"
            raise requests.exceptions.RequestException(
                f"Failed to get agent status: HTTP {resp.status_code} - {error_msg}"
            )
        
        return resp.json()
    
    def poll_until_complete(
        self,
        agent_id: str,
        max_polls: int = 6000000,
        poll_interval: int = 5,
        status_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        completed_statuses: Optional[List[str]] = None,
        failed_statuses: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Poll agent until completion.
        
        Args:
            agent_id: The agent ID
            max_polls: Maximum number of polls (default: 60)
            poll_interval: Seconds between polls (default: 5)
            status_callback: Optional callback function(status_data) called on each poll
            completed_statuses: List of status strings indicating completion (default: ["FINISHED", "completed", "done", "success"])
            failed_statuses: List of status strings indicating failure (default: ["failed", "error", "cancelled"])
            
        Returns:
            Final agent status data when completed
            
        Raises:
            TimeoutError: If agent doesn't complete within max_polls * poll_interval seconds
            RuntimeError: If agent fails
        """
        if completed_statuses is None:
            completed_statuses = ["FINISHED", "completed", "done", "success"]
        if failed_statuses is None:
            failed_statuses = ["failed", "error", "cancelled"]
        
        for poll_num in range(max_polls):
            # Wait-first: sleep before checking status each iteration
            time.sleep(poll_interval)
            
            status_data = self.get_agent_status(agent_id)
            print(f"[POLL] Agent status: {status_data}")
            status = status_data.get("status") or status_data.get("state")
            
            if status_callback:
                status_callback(status_data)
            
            if status in completed_statuses:
                return status_data
            
            if status in failed_statuses:
                error_msg = status_data.get("error") or status_data.get("message") or "Unknown error"
                raise RuntimeError(f"Agent failed with status '{status}': {error_msg}")
        
        raise TimeoutError(
            f"Agent did not complete within {max_polls * poll_interval} seconds"
        )
    
    def add_followup(
        self,
        agent_id: str,
        followup_text: str,
    ) -> Dict[str, Any]:
        """Add a follow-up instruction to an agent.
        
        Args:
            agent_id: The agent ID
            followup_text: The follow-up instruction text
            
        Returns:
            Response data from the API
            
        Raises:
            requests.RequestException: If the API request fails
        """
        url = f"{self.api_url}/v0/agents/{agent_id}/followup"
        payload = {
            "prompt": {
                "text": followup_text,
            },
        }
        
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=60)
        
        if resp.status_code not in (200, 201):
            error_msg = resp.text[:500] if resp.text else "Unknown error"
            raise requests.exceptions.RequestException(
                f"Failed to add follow-up: HTTP {resp.status_code} - {error_msg}"
            )
        
        return resp.json()
    
    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent.
        
        Args:
            agent_id: The agent ID
            
        Returns:
            True if deletion was successful, False otherwise
        """
        url = f"{self.api_url}/v0/agents/{agent_id}"
        
        try:
            resp = requests.delete(url, headers=self._headers(), timeout=30)
            return resp.status_code in (200, 201, 204)
        except Exception:
            return False
    
    @staticmethod
    def extract_file_content(status_data: Dict[str, Any], file_path: str) -> Optional[str]:
        """Extract file content from agent status data.
        
        Args:
            status_data: Agent status data from get_agent_status or poll_until_complete
            file_path: Path or partial path of the file to extract (e.g., "v7.py" will match "my-agents/v7.py")
            
        Returns:
            File content as string, or None if not found
        """
        # Try changes/files array first
        changes = status_data.get("changes") or status_data.get("files") or []
        for change in changes:
            if isinstance(change, dict):
                path = change.get("path") or change.get("file") or ""
                if file_path in path:
                    content = change.get("content") or change.get("body")
                    if content:
                        return content
        
        # Fallback: try to extract from summary/message using code blocks
        summary = status_data.get("summary") or status_data.get("message") or ""
        if summary:
            # Look for code blocks with the file path
            pattern = rf"```python\s*\n#\s*.*?{re.escape(file_path)}.*?\n([\s\S]*?)\n```"
            m = re.findall(pattern, summary, re.DOTALL)
            if m and m[0].strip():
                return m[0].strip()
            
            # Look for any Python code block if file_path matches
            if file_path.endswith(".py"):
                m2 = re.findall(r"```python\s*\n([\s\S]*?)\n```", summary, re.DOTALL)
                if m2 and m2[0].strip():
                    return m2[0].strip()
        
        return None
    
    def launch_and_wait(
        self,
        prompt_text: str,
        repository: Optional[str] = None,
        branch_name: Optional[str] = None,
        skip_reviewer_request: bool = True,
        auto_create_pr: bool = True,
        max_polls: int = 60,
        poll_interval: int = 5,
        status_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """Launch an agent and wait for completion (convenience method).
        
        Args:
            prompt_text: The task/prompt for the agent
            repository: Repository URL (defaults to detected git remote)
            branch_name: Branch name for the agent's work
            skip_reviewer_request: Whether to skip reviewer requests
            auto_create_pr: Whether to automatically create a PR when done
            max_polls: Maximum number of polls
            poll_interval: Seconds between polls
            status_callback: Optional callback function(status_data) called on each poll
            
        Returns:
            Tuple of (agent_id, final_status_data)
            
        Raises:
            Various exceptions from launch_agent, poll_until_complete
        """
        launch_data = self.launch_agent(
            prompt_text=prompt_text,
            repository=repository,
            branch_name=branch_name,
            skip_reviewer_request=skip_reviewer_request,
            auto_create_pr=auto_create_pr,
        )
        
        agent_id = launch_data.get("id") or launch_data.get("agent_id") or launch_data.get("agentId")
        if not agent_id:
            raise ValueError("No agent ID returned from launch")
        
        final_data = self.poll_until_complete(
            agent_id=agent_id,
            max_polls=max_polls,
            poll_interval=poll_interval,
            status_callback=status_callback,
        )
        
        return agent_id, final_data

