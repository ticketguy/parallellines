from engine.organs.base import PerceptionOrgan
from engine.runtime.packets import OrganReading, SignalPacket
class ProbabilityOrgan(PerceptionOrgan):
    name='probability'
    def read(self, signals:list[SignalPacket], topic:str)->OrganReading:
        prices=[]; evidence=[]
        for s in signals:
            p=s.payload.get('yes_price') or s.payload.get('probability')
            if p is not None: prices.append(float(p)); evidence.append(f'{s.source} probability={float(p):.4f}')
        if not prices: return OrganReading(score=0,confidence=0,evidence=[])
        avg=sum(prices)/len(prices)
        return OrganReading(score=round((avg-0.5)*2,4),confidence=min(1,0.4+0.15*len(prices)),evidence=evidence,machine_code='PROBABILITY.READ')
