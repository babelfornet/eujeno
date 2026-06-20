# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import json
import os


def _home() -> str:
    return os.environ.get("EUJENO_HOME") or os.path.join(os.path.expanduser("~"), ".eujeno")


def _path() -> str:
    return os.path.join(_home(), "mcp.json")


def load_servers() -> dict:
    p = _path()
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("servers", {})
    except Exception:
        return {}


def save_servers(servers: dict) -> None:
    os.makedirs(_home(), exist_ok=True)
    with open(_path(), "w", encoding="utf-8") as f:
        json.dump({"servers": servers}, f, indent=2)


def add_server(name: str, command: str, args=None) -> dict:
    servers = load_servers()
    servers[name] = {"command": command, "args": list(args or [])}
    save_servers(servers)
    return servers


def remove_server(name: str) -> dict:
    servers = load_servers()
    servers.pop(name, None)
    save_servers(servers)
    return servers
