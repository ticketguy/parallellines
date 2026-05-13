from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, ClassVar, Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, field_validator
ENGINE_VERSION='0.1.0'
def utc_now(): return datetime.now(timezone.utc)
class PacketBase(BaseModel):
    model_config=ConfigDict(extra='forbid')
    packet_type:str; packet_id:str=Field(default_factory=lambda:str(uuid4())); created_at:datetime=Field(default_factory=utc_now); engine_version:str=ENGINE_VERSION; trace_id:str|None=None
    def with_trace(self, trace_id:str):
        d=self.model_dump(); d['trace_id']=trace_id; return self.__class__(**d)
class OrganReading(BaseModel):
    model_config=ConfigDict(extra='forbid')
    score:float=Field(...,ge=-1,le=1); confidence:float=Field(...,ge=0,le=1); evidence:list[str]=Field(default_factory=list); machine_code:str|None=None
class SignalPacket(PacketBase):
    packet_type:Literal['signal']='signal'; source:str; topic:str; payload:dict[str,Any]; layer_hint:str|None=None; domain:str|None=None; confidence:float=Field(1.0,ge=0,le=1); external_id:str|None=None; url:str|None=None
    @field_validator('source','topic')
    @classmethod
    def non_empty(cls,v):
        v=v.strip()
        if not v: raise ValueError('must not be empty')
        return v
class PerceptionPacket(PacketBase):
    packet_type:Literal['perception']='perception'; topic:str; organs:dict[str,OrganReading]=Field(default_factory=dict)
class MemoryPacket(PacketBase):
    packet_type:Literal['memory']='memory'; topic:str; memories:list[dict[str,Any]]=Field(default_factory=list); continuity_score:float=Field(0,ge=0,le=1); contradictions:list[dict[str,Any]]=Field(default_factory=list)
class SubmindPacket(PacketBase):
    packet_type:Literal['submind']='submind'; topic:str; pressure:float=Field(0,ge=0,le=1); instability:float=Field(0,ge=0,le=1); contradiction:float=Field(0,ge=0,le=1); hidden_pressure:float=Field(0,ge=0,le=1); narrative_heat:float=Field(0,ge=0,le=1); fracture:float=Field(0,ge=0,le=1); subthoughts:list[dict[str,Any]]=Field(default_factory=list)
class IntuOnePacket(PacketBase):
    packet_type:Literal['intuone']='intuone'; topic:str; perception_index:float=Field(...,ge=-1,le=1); confidence:float=Field(...,ge=0,le=1); human_read:str; machine_state:dict[str,Any]=Field(default_factory=dict); next_actions:list[str]=Field(default_factory=list)
class ActionPacket(PacketBase):
    packet_type:Literal['action']='action'; actions:list[dict[str,Any]]=Field(default_factory=list)
Packet=SignalPacket|PerceptionPacket|MemoryPacket|SubmindPacket|IntuOnePacket|ActionPacket
PACKET_TYPES={'signal':SignalPacket,'perception':PerceptionPacket,'memory':MemoryPacket,'submind':SubmindPacket,'intuone':IntuOnePacket,'action':ActionPacket}
def packet_from_dict(data:dict[str,Any])->Packet:
    cls=PACKET_TYPES.get(data.get('packet_type'))
    if cls is None: raise ValueError(f"unknown packet_type: {data.get('packet_type')!r}")
    return cls(**data)
class CorePacketContract(BaseModel):
    model_config=ConfigDict(extra='forbid')
    core_id:str; accepts:list[str]; emits:list[str]; trained_into_weights:bool=True; brain_section:str
    KNOWN_PACKET_NAMES:ClassVar[set[str]]={'SignalPacket','PerceptionPacket','MemoryPacket','SubmindPacket','IntuOnePacket','ActionPacket','RecallPacket','ContradictionPacket','PressurePacket','AnomalyPacket','BriefingPacket'}
    @field_validator('accepts','emits')
    @classmethod
    def known_packet_names(cls, values):
        unknown=[v for v in values if v not in cls.KNOWN_PACKET_NAMES]
        if unknown: raise ValueError(f'unknown packet names: {unknown}')
        return values
