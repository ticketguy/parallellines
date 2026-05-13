from collections import Counter
from engine.organs.base import PerceptionOrgan
from engine.runtime.packets import OrganReading, SignalPacket
class EchoOrgan(PerceptionOrgan):
    name='echo'
    def read(self, signals:list[SignalPacket], topic:str)->OrganReading:
        sources=Counter(s.source for s in signals); repeated=sum(c-1 for c in sources.values() if c>1); text_count=sum(1 for s in signals if s.payload.get('text') or s.payload.get('headline'))
        echo=min(1,(repeated+text_count*0.25)/4); conf=min(1,len(signals)/5)
        return OrganReading(score=round(echo,4),confidence=round(conf,4),evidence=[f'signals={len(signals)}',f'repeated_source_events={repeated}'],machine_code='ECHO.REPETITION')
