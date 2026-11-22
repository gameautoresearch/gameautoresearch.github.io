"""Evaluation declarations of the v3/v3.1 contract that fast-v4 practice still uses.

Public seed sets, the evaluation interfaces, the fixed scripted/teacher panel and
the resolution of a public schedule against a release's protected panel. The v3
release constructor and the private final bank are not part of this tree.
"""
import json
from pathlib import Path
from .readiness_contract import INTERFACE

PUBLIC_SEEDS=list(range(2100101,2100109))
VALIDATION_SEEDS=list(range(2100201,2100205))
PUBLIC_DEMO_SEEDS=list(range(2100301,2100305))
PUBLIC_CAPACITY_SEEDS=list(range(2100401,2100433))
TEACHER_SHA='c2576a533b5f936e0499a22d7e84c7e9e94d80492a188a964223f5b9821fd14a'
INTERFACE_V3={**INTERFACE,'version':'clash-evaluation-v3','execution_observer':'clash-visible-execution-observer-v1','official_grading_cpu_cores':8,'official_grading_memory_gib':32,'score':'equal opponent/seat natural game points; paired seed-cluster interval','private_bank_units':32,'reference_normalized_score':None}
INTERFACE_V31={**INTERFACE_V3,'version':'clash-evaluation-v3.1','grading_pair_workers':3,'simulation_threads_per_pair':1}

def selected_interface(grading):
 if grading=='sequential':return INTERFACE_V3
 if grading=='parallel3':return INTERFACE_V31
 raise ValueError('explicit sequential/parallel3 grading mode required')

def fixed_panel():
 return [{'name':'standard','kind':'scripted'},{'name':'pressure','kind':'scripted'},{'name':'intermediate','kind':'intermediate'},{'name':'teacher','kind':'teacher'}]

def resolve_schedule(release,schedule):
 value=json.loads(json.dumps(schedule));private=json.loads((Path(release)/'private/grade.json').read_text());registry={p['name']:p for p in private['opponents']}
 for p in value['opponents']:
  if p['kind']=='checkpoint':
   original=registry[p['name']]
   if p['identity']!=original['identity']:raise ValueError('public opponent identity differs')
   p['bundle']=original['bundle']
 return value
