from __future__ import annotations
import inspect
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4
from engine.runtime.packets import Packet
PacketHandler=Callable[[Packet],Any|Awaitable[Any]]
class PacketBus:
    def __init__(self): self._handlers=defaultdict(list); self.history=[]
    def subscribe(self, packet_type:str, handler:PacketHandler): self._handlers[packet_type].append(handler)
    async def publish(self, packet:Packet, *, trace_id:str|None=None):
        trace_id=trace_id or packet.trace_id or str(uuid4())
        if packet.trace_id!=trace_id: packet=packet.with_trace(trace_id)
        self.history.append(packet); outputs=[]
        for h in list(self._handlers.get(packet.packet_type, [])):
            r=h(packet)
            if inspect.isawaitable(r): r=await r
            outputs.append(r)
        return outputs
    def packets_of_type(self, packet_type:str): return [p for p in self.history if p.packet_type==packet_type]
