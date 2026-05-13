import json
from engine.brain.cortex_router import CortexMode, ROUTES
from engine.runtime.packets import IntuOnePacket, MemoryPacket, PerceptionPacket, SubmindPacket
from engine.training.littlefig.dataset_schema import LittleFigMessage, LittleFigTrainingRecord
def format_tricore_record(*, topic:str, perception:PerceptionPacket, memory:MemoryPacket, submind:SubmindPacket, target:IntuOnePacket)->LittleFigTrainingRecord:
    route=ROUTES[CortexMode.TRICORE]
    payload={'perception':perception.model_dump(mode='json'),'memory':memory.model_dump(mode='json'),'submind':submind.model_dump(mode='json')}
    return LittleFigTrainingRecord(cortex_mode=CortexMode.TRICORE,topic=topic,expected_packet_type=route.expected_packet_type,messages=[LittleFigMessage(role='system',content=f'{route.system_token} You are the ParallelLines tri-core brain. Return IntuOnePacket JSON.'),LittleFigMessage(role='user',content=json.dumps(payload)),LittleFigMessage(role='assistant',content=target.model_dump_json())])
