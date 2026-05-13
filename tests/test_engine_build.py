import json
from engine import ParallelLinesEngine
from engine.brain import CortexMode, PARALLEL_LINES_BRAIN, PARALLEL_LINES_SPECIAL_TOKENS, ROUTES
from engine.cores.intuone.schema import INTUONE_CONTRACT
from engine.cores.memory_fabric.schema import MEMORY_FABRIC_CONTRACT
from engine.cores.submind.schema import SUBMIND_CONTRACT
from engine.runtime.packets import OrganReading, PerceptionPacket, SignalPacket, packet_from_dict
from engine.training.littlefig import format_tricore_record

def test_brain_lineage_is_fixed_gemma_littlefig():
    assert PARALLEL_LINES_BRAIN.stance=='fixed_base_for_this_lineage'
    assert PARALLEL_LINES_BRAIN.base_model=='google/gemma-4-E4B-it'
    assert '<|core_memory_fabric|>' in PARALLEL_LINES_SPECIAL_TOKENS
    assert ROUTES[CortexMode.TRICORE].expected_packet_type=='intuone'

def test_core_contracts_are_trained_sections():
    assert MEMORY_FABRIC_CONTRACT.trained_into_weights is True
    assert SUBMIND_CONTRACT.trained_into_weights is True
    assert INTUONE_CONTRACT.trained_into_weights is True

def test_packet_round_trip():
    packet=SignalPacket(source='polymarket',topic='btc',payload={'yes_price':0.12})
    assert packet_from_dict(packet.model_dump())==packet

def test_full_engine_cycle_builds_intuone_packet():
    engine=ParallelLinesEngine()
    signals=[SignalPacket(source='polymarket',topic='btc_150k',domain='market',payload={'yes_price':0.0135,'question':'Will Bitcoin hit $150k?'},external_id='poly:1'),SignalPacket(source='forum',topic='btc_150k',domain='social',payload={'text':'Bitcoin will inevitably break 150k, conviction is high.'})]
    output=engine.perceive(signals,topic='btc_150k',trace_id='trace-build')
    assert output.packet_type=='intuone'
    assert output.trace_id=='trace-build'
    assert 'organs' in output.machine_state
    assert 'increase_fracture_monitoring' in output.next_actions

def test_littlefig_tricore_record_format():
    perception=PerceptionPacket(topic='btc',organs={'probability':OrganReading(score=-0.8,confidence=0.9,evidence=['market low'])})
    engine=ParallelLinesEngine(); signal=SignalPacket(source='polymarket',topic='btc',payload={'yes_price':0.1}); output=engine.perceive([signal],topic='btc')
    memory=engine.memory.recall(topic='btc',signals=[signal],perception=perception,trace_id=output.trace_id)
    submind=engine.submind.fire(topic='btc',perception=perception,memory=memory,signals=[signal],trace_id=output.trace_id)
    record=format_tricore_record(topic='btc',perception=perception,memory=memory,submind=submind,target=output)
    assert record.cortex_mode==CortexMode.TRICORE
    assert record.expected_packet_type=='intuone'
    assert json.loads(record.messages[-1].content)['packet_type']=='intuone'
