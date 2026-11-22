"""Versioned evaluator-v2 admission; historical bundles retain their exact binding."""
import ast,json
from .common import HERE,digest,engine_binding,sha
from .bundle import verify

INTERFACE={'version':'clash-evaluation-v2','task':'clash-fixed8-v1','observation_ints':7024,'actions':2307,'entity_rows':200,'decisions_per_second':4,'regulation_seconds':180,'overtime_seconds':120,'entrant_parameter_cap':1000000,'entrant_dtype':'float32','entrant_inference':'stateless per-seat pure NN, deterministic masked greedy argmax; no search','entrant_pipe_deadline_seconds':15,'trusted_opponent_inference':{'inputs':'single seat public observation and legal mask only','CPU_only':True,'search':False,'per_decision_deadline_seconds':.2,'model_parameter_cap_applies':False},'bundle_format':'clash-ar-bundle-v1','legacy_engine_binding_unchanged':True}

def review_policy(path):
 """Enforce the explicit greedy entry and reject common state/IO escapes.

 This is a source admission check, not a proof about arbitrary Python. Runtime
 isolation, repeated public probes and owning source review remain required.
 """
 tree=ast.parse(path.read_text());act=next((n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='act'),None)
 if act is None or len(act.args.args)!=3:raise ValueError('act(model,observation,mask) required')
 body=[n for n in act.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
 if len(body)!=1 or not isinstance(body[0],ast.Return):raise ValueError('act must directly return learned greedy logits')
 names=[n.arg for n in act.args.args]
 expected=ast.parse('def act('+','.join(names)+'):\n return '+names[0]+'('+names[1]+','+names[2]+')[0].argmax(-1)').body[0].body[0].value
 if ast.dump(body[0].value)!=ast.dump(expected):raise ValueError('act must return model(observation,mask)[0].argmax(-1)')
 for n in ast.walk(tree):
  if isinstance(n,(ast.Global,ast.Nonlocal,ast.AsyncFunctionDef)):raise ValueError('stateful policy construct')
  if isinstance(n,(ast.Import,ast.ImportFrom)):
   modules=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module or '']
   trusted_wrapper=sha(path)==sha(HERE/'starter/policy.py')
   if any(m.split('.')[0] not in ('torch','numpy','math') and not (trusted_wrapper and m=='simulators.clash.autoresearch.fast_policy') for m in modules):raise ValueError('policy import outside pure NN dependencies')
  if isinstance(n,ast.Call):
   name=n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else ''
   if name in ('open','exec','eval','compile','__import__','load','save','seed','manual_seed','random','rand','randn','randint','Generator'):raise ValueError('policy state/IO/random operation')
 return {'greedy_entry_checked':True,'source_escape_checks_passed':True,'source_sha256':sha(path),'arbitrary_Python_purity_proven':False,'owning_source_review_required':True}

def admit(bundle):
 from pathlib import Path
 bundle=Path(bundle);manifest=verify(bundle);review=review_policy(bundle/'policy.py')
 return {'interface':INTERFACE,'interface_sha256':digest(INTERFACE),'engine_binding_sha256':digest(engine_binding()),'identity':manifest['identity'],'parameters':manifest['parameters'],'source_review':review,'runtime_probes_required':True,'learned_reference_qualified':False,'strong_panel_qualified':False,'superhuman_evidence':False}
