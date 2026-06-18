# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class McpRegistry:
    """MCP host: keeps the server configs and opens a stdio session per operation.
    Tool names exposed to the model are prefixed 'server__tool' to avoid collisions."""
    def __init__(self):
        self._servers = {}   # name -> {"command": str, "args": [str]}

    def add(self, name: str, command: str, args=None) -> None:
        self._servers[name] = {"command": command, "args": list(args or [])}

    def remove(self, name: str) -> None:
        self._servers.pop(name, None)

    def list_servers(self) -> list:
        return list(self._servers.keys())

    async def _alist_tools(self):
        out = []
        for name, cfg in self._servers.items():
            params = StdioServerParameters(command=cfg["command"], args=cfg["args"])
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    res = await session.list_tools()
                    for t in res.tools:
                        out.append({
                            "type": "function",
                            "function": {
                                "name": f"{name}__{t.name}",
                                "description": t.description or "",
                                "parameters": t.inputSchema or {"type": "object", "properties": {}},
                            },
                            "_server": name, "_tool": t.name,
                        })
        return out

    def list_tools(self) -> list:
        return asyncio.run(self._alist_tools())

    async def _acall_tool(self, full_name: str, arguments: dict) -> str:
        server, _, tool = full_name.partition("__")
        cfg = self._servers[server]
        params = StdioServerParameters(command=cfg["command"], args=cfg["args"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.call_tool(tool, arguments or {})
                parts = [getattr(c, "text", "") or "" for c in res.content]
                return "\n".join(p for p in parts if p)

    def call_tool(self, full_name: str, arguments: dict) -> str:
        return asyncio.run(self._acall_tool(full_name, arguments))
