from pydantic import BaseModel
class BrainLineage(BaseModel):
    name:str='ParallelLines-Gemma4E4B-LittleFig'; base_model:str='google/gemma-4-E4B-it'; stance:str='fixed_base_for_this_lineage'; trainer:str='Harboria-Labs/littlefig'; memory_system:str='Harboria-Labs/embers-diaries'
PARALLEL_LINES_BRAIN=BrainLineage()
