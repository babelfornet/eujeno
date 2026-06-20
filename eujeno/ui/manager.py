# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import subprocess


class NodeManager:
    """Manages the local processes started by the UI (coordinator and/or worker)."""
    def __init__(self):
        self._procs = {}   # role -> {"popen": Popen, "info": dict}

    def start(self, role: str, cmd: list, info: dict) -> None:
        self.stop(role)
        popen = subprocess.Popen(cmd)
        self._procs[role] = {"popen": popen, "info": dict(info)}

    def status(self) -> dict:
        out = {}
        for role, d in self._procs.items():
            running = d["popen"].poll() is None
            out[role] = {"running": running, "pid": d["popen"].pid, **d["info"]}
        return out

    def stop(self, role: str) -> None:
        d = self._procs.pop(role, None)
        if d is not None and d["popen"].poll() is None:
            d["popen"].terminate()
            try:
                d["popen"].wait(timeout=5)
            except Exception:
                d["popen"].kill()

    def stop_all(self) -> None:
        for role in list(self._procs.keys()):
            self.stop(role)
