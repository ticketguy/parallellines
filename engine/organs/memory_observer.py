from engine.organs.base import PerceptionOrgan
from engine.runtime.packets import OrganReading, SignalPacket
class MemoryObserverOrgan(PerceptionOrgan):
    name='memory'
    def read(self, signals:list[SignalPacket], topic:str)->OrganReading:
        ids={s.external_id for s in signals if s.external_id}; texts=[s for s in signals if s.payload.get('text') or s.payload.get('headline')]
        persistence=min(1,len(ids)*0.2+len(texts)*0.15); conf=min(1,0.2+len(signals)/6) if signals else 0
        return OrganReading(score=round(persistence,4),confidence=round(conf,4),evidence=[f'unique_external_ids={len(ids)}',f'text_signals={len(texts)}'],machine_code='MEMORY.PERSISTENCE')
