"""Inference classes of the exact-task cross-play evaluator."""
CLASSES={
 'entrant-stateless-600k-v1':dict(parameters=600_000,persistent_memory_bytes_per_board=0,process_limit=64,file_limit=64,action_codec='entrant4609',observation='normalizedHWC14player-visible',first_response_seconds=30,response_seconds=15),
 'trusted-averagejoe-history-16m-v1':dict(parameters=16_000_000,persistent_memory_bytes_per_board=65536,history_spatial=7,history_temporal=512,process_limit=256,file_limit=512,action_codec='direction-major9x24x24-to-entrant4609',observation='rawCHW14player-visible-to38planes',first_response_seconds=90,response_seconds=15),
}
RULES_ID='generals-research-1v1-full-v2'
EVALUATOR_ID='generals-exact-task-h100-crossplay-v2-20261109'
