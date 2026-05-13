from engine.runtime.packets import CorePacketContract
MEMORY_FABRIC_CONTRACT=CorePacketContract(core_id='core.memory_fabric',brain_section='memory cortex',trained_into_weights=True,accepts=['SignalPacket','PerceptionPacket','IntuOnePacket'],emits=['MemoryPacket','RecallPacket','ContradictionPacket'])
