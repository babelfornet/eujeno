# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Per-node persisted settings + a stable peer identity."""
import json
import os
import tempfile
import uuid

DEFAULTS = {
    "name": "eujeno-node",
    "model": None,
    "layerMode": "manual",
    "maxLayers": 8,
    "maxRam": 16,
    "port": 8001,
    "region": "eu-west",
    "bandwidth": 200,
    "autojoin": True,
    "contribute": True,
    "inbound": True,
    "telemetry": False,
}


class NodeConfig:
    def __init__(self, path=None):
        self.path = path
        self._data = dict(DEFAULTS)
        self._data["peerId"] = None
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    self._data.update(json.load(f))
            except Exception:
                pass
        if not self._data.get("peerId"):
            self._data["peerId"] = f"node·{uuid.uuid4().hex[:8]}·{uuid.uuid4().hex[:8]}"
            self._save()

    @property
    def peer_id(self):
        return self._data["peerId"]

    def get(self):
        return dict(self._data)

    def update(self, partial):
        for k, v in (partial or {}).items():
            if k == "peerId":
                continue
            if k in DEFAULTS:
                self._data[k] = v
        self._save()
        return self.get()

    def _save(self):
        if not self.path:
            return
        try:
            d = os.path.dirname(self.path) or "."
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d)
            with os.fdopen(fd, "w") as f:
                json.dump(self._data, f)
            os.replace(tmp, self.path)
        except Exception:
            pass
