"""Explicit new-campaign evaluator role guard; never uses the historical default ledger."""
from pathlib import Path
import json
import time
from tools.overnight_budget import CampaignBudget
from integrations.pokemon.autoresearch.contract import ROOT,write

BASE=ROOT/'evidence/reference-readiness-20261109-0251Z/pokemon/evaluator'
LEDGER=BASE.parents[1]/'campaign.sqlite3'
PREFIX='pokemon/evaluator/'


def guard():
    return CampaignBudget('pokemon',path=LEDGER)


def own_usage():
    rows=[r for r in guard().snapshot()['operations'] if r['game']=='pokemon' and r['kind'].startswith(PREFIX)]
    settled=[r for r in rows if r['status']=='settled'];held=[r for r in rows if r['status']!='settled']
    return {'settled_gpu_seconds':sum(r['gpu_charged'] for r in settled),
        'held_gpu_seconds':sum(r['gpu_hold'] for r in held),
        'settled_resource_usd':sum(r['usd_charged'] for r in settled if r['bucket']=='resource'),
        'held_resource_usd':sum(r['usd_hold'] for r in held if r['bucket']=='resource'),
        'settled_model_usd':sum(r['usd_charged'] for r in settled if r['bucket']=='model'),
        'held_model_usd':sum(r['usd_hold'] for r in held if r['bucket']=='model')}


def reserve_cpu(label,seconds):
    counterpart=BASE.parent/'coordination/rl.json'
    if not counterpart.exists():raise ValueError('Read counterpart acknowledgment before paid work')
    json.loads(counterpart.read_text())
    usage=own_usage();auth=guard().auth;rates=auth['pricing']
    quote=seconds*(4*rates['sandbox_cpu_per_core_second']+8*rates['sandbox_memory_per_gib_second'])*rates['reservation_margin']
    if usage['settled_resource_usd']+usage['held_resource_usd']+quote>8:raise ValueError('Evaluator role resource subcap exhausted')
    return guard().reserve_worker(PREFIX+label,seconds,gpu=None,cpu_cores=4,memory_gib=8,sandbox=True)


def coordinate(phase,**kwargs):
    path=BASE.parent/'coordination/evaluator.json';data=json.loads(path.read_text())
    data.update(phase=phase,updated_at_unix=time.time(),budget_use=own_usage(),**kwargs);write(path,data)
