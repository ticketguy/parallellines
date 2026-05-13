from engine.organs.base import PerceptionOrgan
from engine.runtime.packets import OrganReading, SignalPacket
class ConvictionOrgan(PerceptionOrgan):
    name='conviction'; markers=('must','will','guaranteed','certain','inevitable','never','belief','conviction')
    def read(self, signals:list[SignalPacket], topic:str)->OrganReading:
        hits=0; evidence=[]
        for s in signals:
            text=str(s.payload.get('text') or s.payload.get('headline') or s.payload.get('question') or '').lower()
            h=sum(1 for m in self.markers if m in text)
            if h: hits+=h; evidence.append(f'{s.source} conviction_markers={h}')
        return OrganReading(score=round(min(1,hits/5),4),confidence=round(min(1,0.2+hits/6),4) if hits else 0,evidence=evidence,machine_code='CONVICTION.TEXT')
