# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import json
import struct


def pack(header: dict, payload: bytes = b"") -> bytes:
    """A frame = uint32 big-endian (JSON header length) + header + payload."""
    hb = json.dumps(header).encode("utf-8")
    return struct.pack(">I", len(hb)) + hb + payload


def unpack(data: bytes):
    """Inverse of pack(). Returns (header: dict, payload: bytes)."""
    n = struct.unpack(">I", data[:4])[0]
    header = json.loads(data[4:4 + n].decode("utf-8"))
    return header, data[4 + n:]
