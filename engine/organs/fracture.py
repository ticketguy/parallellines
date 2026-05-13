from engine.organs.base import PerceptionOrgan
from engine.runtime.packets import OrganReading, SignalPacket
class FractureOrgan(PerceptionOrgan):
    name='fracture'
    def read(self, signals:list[SignalPacket], topic:str)->OrganReading:
        prices=[]; markers=0
        for s in signals:
            p=s.payload.get('yes_price') or s.payload.get('probability')
            if p is not None: prices.append(float(p))
            text=str(s.payload.get('text') or s.payload.get('headline') or s.payload.get('question') or '').lower(); markers+=sum(1 for m in ('will','must','certain','inevitable') if m in text)
        if not prices: return OrganReading(score=0,confidence=0,evidence=[])
        avg=sum(prices)/len(prices); conviction=min(1,markers/4); low_prob=max(0,0.5-avg)*2; fracture=min(1,abs(conviction-low_prob))
        return OrganReading(score=round(fracture,4),confidence=0.7 if fracture else 0,evidence=[f'avg_probability={avg:.4f}',f'conviction_proxy={conviction:.4f}'],machine_code='FRACTURE.DIVERGENCE')
