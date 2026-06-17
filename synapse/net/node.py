import asyncio

import websockets

from synapse.net.framing import pack, unpack
from synapse.net.node_exec import handle_request


async def run_node(coordinator_ws_url: str, state):
    """Si connette (outbound, NAT-friendly) al coordinator, annuncia gli stage e serve
    gli hop relayati. Il calcolo torch gira in un executor per non bloccare il loop."""
    async with websockets.connect(coordinator_ws_url, max_size=None) as ws:
        await ws.send(pack({"type": "announce", "stages": state.stages_dict()}))
        loop = asyncio.get_event_loop()
        async for message in ws:
            header, payload = unpack(message)
            resp_header, resp_payload = await loop.run_in_executor(
                None, handle_request, state, header, payload)
            await ws.send(pack({**resp_header, "req_id": header.get("req_id")}, resp_payload))
