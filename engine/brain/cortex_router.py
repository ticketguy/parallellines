from enum import StrEnum
from pydantic import BaseModel
class CortexMode(StrEnum): MEMORY_FABRIC='memory_fabric'; SUBMIND='submind'; INTUONE='intuone'; TRICORE='tricore'
class CortexRoute(BaseModel): mode:CortexMode; system_token:str; expected_packet_type:str
ROUTES={CortexMode.MEMORY_FABRIC:CortexRoute(mode=CortexMode.MEMORY_FABRIC,system_token='<|core_memory_fabric|>',expected_packet_type='memory'),CortexMode.SUBMIND:CortexRoute(mode=CortexMode.SUBMIND,system_token='<|core_submind|>',expected_packet_type='submind'),CortexMode.INTUONE:CortexRoute(mode=CortexMode.INTUONE,system_token='<|core_intuone|>',expected_packet_type='intuone'),CortexMode.TRICORE:CortexRoute(mode=CortexMode.TRICORE,system_token='<|machine_state|>',expected_packet_type='intuone')}
