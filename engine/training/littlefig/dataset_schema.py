from typing import Literal
from pydantic import BaseModel, Field
from engine.brain.cortex_router import CortexMode
class LittleFigMessage(BaseModel): role:Literal['system','user','assistant']; content:str
class LittleFigTrainingRecord(BaseModel): cortex_mode:CortexMode; topic:str; messages:list[LittleFigMessage]; expected_packet_type:str; quality_score:float=Field(1.0,ge=0,le=1); source:str='parallellines_engine'
