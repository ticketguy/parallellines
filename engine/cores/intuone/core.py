from engine.runtime.packets import IntuOnePacket, MemoryPacket, PerceptionPacket, SubmindPacket
class IntuOneCore:
    core_id='core.intuone'
    def interpret(self, *, topic:str, perception:PerceptionPacket, memory:MemoryPacket, submind:SubmindPacket, trace_id:str|None=None)->IntuOnePacket:
        weighted=sum(r.score*r.confidence for r in perception.organs.values()); total=sum(r.confidence for r in perception.organs.values()); idx=weighted/total if total else 0; avg=total/max(len(perception.organs),1); conf=min(1,(avg+memory.continuity_score+(1-submind.instability))/3)
        if submind.fracture>0.5: read=f'{topic}: belief geometry is fractured. This is instability, not prediction.'
        elif idx>0.2: read=f'{topic}: perception leans positive with moderate coherence.'
        elif idx<-0.2: read=f'{topic}: perception leans negative with moderate coherence.'
        else: read=f'{topic}: perception is unresolved.'
        state={'organs':{n:r.model_dump() for n,r in perception.organs.items()},'memory':{'continuity_score':memory.continuity_score,'contradiction_count':len(memory.contradictions)},'submind':submind.model_dump()}
        actions=[]
        if submind.fracture>0.5: actions.append('increase_fracture_monitoring')
        if memory.contradictions: actions.append('retrieve_contradiction_history')
        return IntuOnePacket(topic=topic,perception_index=round(idx,4),confidence=round(conf,4),human_read=read,machine_state=state,next_actions=actions,trace_id=trace_id)
