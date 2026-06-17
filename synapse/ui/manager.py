import subprocess


class NodeManager:
    """Gestisce i processi locali avviati dalla UI (coordinator e/o worker)."""
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
