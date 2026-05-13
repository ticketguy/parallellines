from uuid import uuid4
from engine.cores import IntuOneCore, MemoryFabricCore, SubmindCore
from engine.organs import DEFAULT_ORGANS
from engine.runtime.packets import IntuOnePacket, PerceptionPacket, SignalPacket
class ParallelLinesEngine:
    def __init__(self, organs=None, memory=None, submind=None, intuone=None):
        self.organs=organs or DEFAULT_ORGANS; self.memory=memory or MemoryFabricCore(); self.submind=submind or SubmindCore(); self.intuone=intuone or IntuOneCore()
    def perceive(self, signals:list[SignalPacket], topic:str, trace_id:str|None=None)->IntuOnePacket:
        trace_id=trace_id or str(uuid4())
        perception=PerceptionPacket(topic=topic,organs={o.name:o.read(signals,topic) for o in self.organs},trace_id=trace_id)
        memory=self.memory.recall(topic=topic,signals=signals,perception=perception,trace_id=trace_id)
        submind=self.submind.fire(topic=topic,perception=perception,memory=memory,signals=signals,trace_id=trace_id)
        return self.intuone.interpret(topic=topic,perception=perception,memory=memory,submind=submind,trace_id=trace_id)
