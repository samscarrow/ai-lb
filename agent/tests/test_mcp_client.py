import sys
import textwrap
from pathlib import Path

from agent.tools.mcp_client import MCPClient, MCPServerConfig


def _write_stub_server(path: Path) -> None:
    script = textwrap.dedent(
        """
        import json
        import sys

        def send(payload):
            sys.stdout.write(json.dumps(payload) + "\\n")
            sys.stdout.flush()

        for line in sys.stdin:
            if not line.strip():
                continue
            msg = json.loads(line)
            method = msg.get("method")
            if method == "initialize":
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": msg.get("id"),
                        "result": {"capabilities": {}, "serverInfo": {"name": "stub", "version": "0.1"}},
                    }
                )
            elif method == "tools/list":
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": msg.get("id"),
                        "result": {
                            "tools": [
                                {
                                    "name": "echo",
                                    "description": "Echo text",
                                    "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
                                }
                            ]
                        },
                    }
                )
            elif method == "tools/call":
                args = (msg.get("params") or {}).get("arguments") or {}
                text = args.get("text", "")
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": msg.get("id"),
                        "result": {"content": [{"type": "text", "text": text}]},
                    }
                )
            elif method == "initialized":
                continue
        """
    ).lstrip()
    path.write_text(script, encoding="utf-8")


def test_mcp_client_list_and_call(tmp_path: Path) -> None:
    script_path = tmp_path / "stub_server.py"
    _write_stub_server(script_path)
    config = MCPServerConfig(name="stub", command=[sys.executable, str(script_path)])
    client = MCPClient(config)
    client.initialize()

    tools = client.list_tools()
    assert tools
    assert tools[0].name == "echo"

    result = client.call_tool("echo", {"text": "hi"})
    content = result.get("content")
    assert isinstance(content, list)
    assert content[0]["text"] == "hi"

    client.close()
