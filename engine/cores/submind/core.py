from engine.runtime.packets import MemoryPacket, PerceptionPacket, SignalPacket, SubmindPacket
class SubmindCore:
    core_id='core.submind'
    def fire(self, *, topic:str, perception:PerceptionPacket, memory:MemoryPacket, signals:list[SignalPacket]|None=None, trace_id:str|None=None)->SubmindPacket:
        signals=signals or []; organs=perception.organs; p=organs.get('probability'); c=organs.get('conviction'); shadow=organs.get('shadow'); fr=organs.get('fracture')
        divergence=min(1,abs(p.score-c.score)) if p and c else 0
        narrative_heat=max([r.confidence*abs(r.score) for n,r in organs.items() if n in {'echo','memory','conviction'}] or [0])
        hidden=shadow.confidence*abs(shadow.score) if shadow else 0; fracture=max(divergence,abs(fr.score) if fr else 0); contradiction=min(1,0.25*len(memory.contradictions)+divergence); instability=min(1,(fracture+contradiction+hidden)/3); pressure=min(1,(narrative_heat+hidden+fracture+0.05*len(signals))/3)
        thoughts=[]
        if divergence>0.5: thoughts.append({'name':'probability_conviction_split','machine_code':f'FRACTURE.PROBABILITY_CONVICTION:{divergence:.2f}'})
        if memory.contradictions: thoughts.append({'name':'memory_conflict_present','machine_code':f'MEMORY.CONFLICT:{len(memory.contradictions)}'})
        return SubmindPacket(topic=topic,pressure=round(pressure,4),instability=round(instability,4),contradiction=round(contradiction,4),hidden_pressure=round(hidden,4),narrative_heat=round(narrative_heat,4),fracture=round(fracture,4),subthoughts=thoughts,trace_id=trace_id)
