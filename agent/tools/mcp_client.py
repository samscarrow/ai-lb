from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_PROTOCOL_VERSION = "2024-11-05"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    command: List[str]
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None


class MCPClient:
    def __init__(
        self,
        config: MCPServerConfig,
        startup_timeout_secs: float = 10.0,
    ) -> None:
        self._config = config
        self._startup_timeout_secs = startup_timeout_secs
        self._proc: Optional[subprocess.Popen[str]] = None
        self._stdout_queue: "queue.Queue[str]" = queue.Queue()
        self._stderr_queue: "queue.Queue[str]" = queue.Queue()
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._request_id = 0

    def start(self) -> None:
        if self._proc is not None:
            return
        env = os.environ.copy()
        if self._config.env:
            env.update(self._config.env)
        self._proc = subprocess.Popen(
            self._config.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self._config.cwd,
            env=env,
            bufsize=1,
        )
        if not self._proc.stdin or not self._proc.stdout or not self._proc.stderr:
            raise RuntimeError(f"Failed to start MCP server {self._config.name}")
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def initialize(self) -> Dict[str, Any]:
        self.start()
        params = {
            "protocolVersion": DEFAULT_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "hybrid-agent-client", "version": "0.1"},
        }
        response = self._send_request("initialize", params=params, timeout=self._startup_timeout_secs)
        self._send_notification("initialized")
        return response

    def list_tools(self) -> List[ToolSpec]:
        response = self._send_request("tools/list")
        tools = response.get("result", {}).get("tools", [])
        out = []
        for tool in tools:
            out.append(
                ToolSpec(
                    name=tool.get("name", ""),
                    description=tool.get("description", ""),
                    input_schema=tool.get("inputSchema", {}) or {},
                )
            )
        return out

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = {"name": name, "arguments": arguments or {}}
        response = self._send_request("tools/call", params=params)
        return response.get("result", {})

    def close(self) -> None:
        if not self._proc:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
        finally:
            self._proc = None

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("MCP server not started")
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()

    def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Dict[str, Any]:
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("MCP server not started")
        with self._lock:
            req_id = self._next_id()
            payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
            if params is not None:
                payload["params"] = params
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
        return self._wait_for_response(req_id, timeout=timeout)

    def _wait_for_response(self, req_id: int, timeout: float) -> Dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self._stdout_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("id") == req_id:
                return payload
        raise TimeoutError(f"MCP server {self._config.name} timed out waiting for response")

    def _read_stdout(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            self._stdout_queue.put(line.strip())

    def _read_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        for line in self._proc.stderr:
            self._stderr_queue.put(line.strip())


class MCPToolRouter:
    def __init__(self, servers: List[MCPServerConfig], startup_timeout_secs: float = 10.0) -> None:
        self._servers = servers
        self._startup_timeout_secs = startup_timeout_secs
        self._clients: List[MCPClient] = []
        self._tool_map: Dict[str, Tuple[MCPClient, str]] = {}
        self._tool_specs: List[ToolSpec] = []

    @classmethod
    def from_env(cls, startup_timeout_secs: float = 10.0) -> "MCPToolRouter":
        servers: List[MCPServerConfig] = []
        fs_cmd = os.getenv("MCP_FILESYSTEM_CMD")
        shell_cmd = os.getenv("MCP_SHELL_CMD")
        if fs_cmd:
            servers.append(MCPServerConfig(name="filesystem", command=shlex.split(fs_cmd)))
        if shell_cmd:
            servers.append(MCPServerConfig(name="shell", command=shlex.split(shell_cmd)))
        return cls(servers=servers, startup_timeout_secs=startup_timeout_secs)

    def start(self) -> None:
        if self._clients:
            return
        for server in self._servers:
            client = MCPClient(server, startup_timeout_secs=self._startup_timeout_secs)
            client.initialize()
            tools = client.list_tools()
            for tool in tools:
                mapped_name = tool.name
                if mapped_name in self._tool_map:
                    mapped_name = f"{server.name}.{tool.name}"
                self._tool_map[mapped_name] = (client, tool.name)
                self._tool_specs.append(
                    ToolSpec(
                        name=mapped_name,
                        description=tool.description,
                        input_schema=tool.input_schema,
                    )
                )
            self._clients.append(client)

    def tool_catalog(self) -> List[ToolSpec]:
        return list(self._tool_specs)

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if name not in self._tool_map:
            raise KeyError(f"Unknown tool: {name}")
        client, raw_name = self._tool_map[name]
        return client.call_tool(raw_name, arguments=arguments)

    def close(self) -> None:
        for client in self._clients:
            client.close()
        self._clients = []
        self._tool_map = {}
        self._tool_specs = []
