from abc import ABC, abstractmethod
from engine.runtime.packets import OrganReading, SignalPacket
class PerceptionOrgan(ABC):
    name:str
    @abstractmethod
    def read(self, signals:list[SignalPacket], topic:str)->OrganReading: ...
