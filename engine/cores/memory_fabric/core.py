from engine.runtime.packets import IntuOnePacket, MemoryPacket, PerceptionPacket, SignalPacket
class MemoryFabricCore:
    core_id='core.memory_fabric'
    def recall(self, *, topic:str, signals:list[SignalPacket]|None=None, perception:PerceptionPacket|None=None, prior_intuone:list[IntuOnePacket]|None=None, trace_id:str|None=None)->MemoryPacket:
        signals=signals or []; prior_intuone=prior_intuone or []; memories=[]; contradictions=[]
        for s in signals: memories.append({'type':'episodic_signal','source':s.source,'payload':s.payload,'external_id':s.external_id})
        if perception:
            p=perception.organs.get('probability'); c=perception.organs.get('conviction')
            if p and c and abs(p.score-c.score)>0.7: contradictions.append({'type':'probability_conviction_divergence','probability':p.score,'conviction':c.score})
            for n,r in perception.organs.items():
                if r.confidence>0: memories.append({'type':'organ_reading','organ':n,'score':r.score,'confidence':r.confidence})
        for pkt in prior_intuone: memories.append({'type':'prior_intuone_read','perception_index':pkt.perception_index,'confidence':pkt.confidence})
        return MemoryPacket(topic=topic,memories=memories,continuity_score=min(1,0.1*len(memories)),contradictions=contradictions,trace_id=trace_id)
