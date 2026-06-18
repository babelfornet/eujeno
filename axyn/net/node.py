# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import asyncio

import websockets

from axyn.net.framing import pack, unpack
from axyn.net.node_exec import handle_request


async def run_node(coordinator_ws_url: str, state):
    """Connects (outbound, NAT-friendly) to the coordinator, announces its stages and serves
    the relayed hops. The torch computation runs in an executor so the loop isn't blocked."""
    async with websockets.connect(coordinator_ws_url, max_size=None) as ws:
        await ws.send(pack({"type": "announce", "stages": state.stages_dict()}))
        loop = asyncio.get_running_loop()
        async for message in ws:
            header, payload = unpack(message)
            resp_header, resp_payload = await loop.run_in_executor(
                None, handle_request, state, header, payload)
            await ws.send(pack({**resp_header, "req_id": header.get("req_id")}, resp_payload))
