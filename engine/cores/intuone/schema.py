from engine.runtime.packets import CorePacketContract
INTUONE_CONTRACT=CorePacketContract(core_id='core.intuone',brain_section='interpreter cortex',trained_into_weights=True,accepts=['PerceptionPacket','MemoryPacket','SubmindPacket'],emits=['IntuOnePacket','BriefingPacket','ActionPacket'])
