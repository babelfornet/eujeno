import json
import struct


def pack(header: dict, payload: bytes = b"") -> bytes:
    """Un frame = uint32 big-endian (lunghezza header JSON) + header + payload."""
    hb = json.dumps(header).encode("utf-8")
    return struct.pack(">I", len(hb)) + hb + payload


def unpack(data: bytes):
    """Inverso di pack(). Ritorna (header: dict, payload: bytes)."""
    n = struct.unpack(">I", data[:4])[0]
    header = json.loads(data[4:4 + n].decode("utf-8"))
    return header, data[4 + n:]
