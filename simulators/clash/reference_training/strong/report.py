"""Render this role's new-campaign evidence without reading private seeds."""
import json
from .status import status
from .provider import OWN,CAMPAIGN
from simulators.clash.autoresearch.common import ROOT,dump

def report():
 s=status();attempts=[]
 for path in sorted(OWN.glob('*/provider.json')):
  root=path.parent;p=json.loads(path.read_text());item={'name':root.name,'provider':p}
  for name in ('settlement','artifact-validation','observer-outcome','collection-stream-recovery','transfer-collision-recovery'):
   q=root/(name+'.json')
   if q.exists():item[name.replace('-','_')]=json.loads(q.read_text())
  attempts.append(item)
  result=next((q for q in (root/'collected/output/result.json',root/'collected/output/training/result.json') if q.exists()),None)
  if result:
   r=json.loads(result.read_text());c=r['binding']['config'];parent_updates=r.get('parent_updates',0);new_updates=r['updates']-parent_updates;rows=c['envs']*c['rollout_steps'];assisted=max(0,min(r['updates'],c['teacher_updates'])-min(parent_updates,c['teacher_updates'])) if c['teacher_coef'] else 0
   item['logical_work']={'new_updates':new_updates,'new_learner_rows':new_updates*rows,'new_PPO_updates':0 if p['variant']=='v12' else new_updates,'new_supervised_only_updates':new_updates if p['variant']=='v12' else 0,'new_demonstration_rows':assisted*rows,'extra_discarded_audit_updates':1,'extra_discarded_audit_learner_rows':rows,'extra_discarded_audit_demonstration_rows':rows if c['teacher_coef'] and (r.get('parent_updates',2)<c['teacher_updates']) else 0}
 coord=json.loads((CAMPAIGN/'clash/coordination/rl.json').read_text())
 selection=OWN/'selection-close.json';cleanup=sorted(OWN.glob('cleanup-*.json'))
 expanded=[json.loads(p.read_text()) for p in sorted((OWN/'cpu-development').glob('*/*-paired-public32.json'))]
 value={'at_utc':s['at_utc'],'phase':coord['phase'],'authorization':str(CAMPAIGN/'authorization.json'),'explicit_ledger':str(CAMPAIGN/'campaign.sqlite3'),'accounting':s['accounting'],'sustained_main':coord.get('sustained_main'),'attempts':attempts,'public_development':s['public_development'],'cpu_public_development':s['cpu_public_development'],'expanded_public_comparisons':expanded,'selection':json.loads(selection.read_text()) if selection.exists() else None,'cleanup':json.loads(cleanup[-1].read_text()) if cleanup else None,'strong_panel_qualified':False,'superhuman_evidence':False,'reference_normalization':None,'private_feedback_used_for_training':False,'old_campaign_costs_reassigned':False,'committed_or_pushed':False,'published':False}
 dest=ROOT/'reports/clash/strong-bots-five-hour-2026-11-09-rl';dump(dest.with_suffix('.json'),value)
 a=s['accounting'];rows=[]
 for x in s['public_development']:
  score=f"{x['mean_points']:.4f}" if x['mean_points'] is not None else 'null'
  rows.append(f"| {x['name']} | {x['games']} | {score} |")
 for batch in s['cpu_public_development']:
  for name,x in (batch['summaries'] or {}).items():
   score=f"{x['mean_points']:.4f}" if x['mean_points'] is not None else 'null'
   rows.append(f"| {name} (isolated Linux) | {x['games']} | {score} |")
 main=coord.get('sustained_main',{});best=main.get('best_public32',{});best_line='';latest_line=''
 if best:
  best_line=f"The best validated main snapshot so far is update{best['updates']}: {best['points']:.4f}/100 on the larger published32-seed panel, versus{best['baseline_points']:.4f} for the incumbent. Its paired development change is{best['paired_change']:+.4f}, interval{best['interval95']}; repeated selection is uncorrected and expert strength remains unqualified. Earlier and later snapshots remain preserved."
 latest=main.get('latest_public32',{})
 if latest:
  latest_line=f"The latest fully assessed32-seed stage-end is update{latest['updates']}, scoring{latest['points']:.4f}. The update1456 model regressed after the precommitted assistance cutoff1024; the stronger update778 checkpoint was retained. On matched public grades, placed cards per game fell25.9297→4.0260. Finite loss/source/world checks passed; these observations do not prove a causal mechanism. The sustained unchanged trajectory continues rather than relabeling a weak policy as a simulator fault."
 text=f'''# Clash five-hour campaign — RL evidence

Updated {s['at_utc']}. Phase: {coord['phase']}. Selection closes09:36:36UTC; workers stop10:11:36UTC; report/cleanup deadline10:16:36UTC, November9.

The sustained main is the fixedv11 learner-state trajectory, starting from its validated update266 checkpoint (40.625/100 on the eight-seed public panel). Its source, recipe, optimizer, worlds, RNG and history remain unchanged across strict continuation stages. The current stage runs approximately07:31–08:30UTC, with a second stage targeted to end around09:20UTC for validation/handoff. Further architecture/recipe/probe GPU launches were cancelled under the user's sustained-training steering. Preparedv13 code remains CPU-only and unspent.

{best_line}

{latest_line}

The retained spatial incumbent scores32.2917/100 on that same six-opponent public panel. These are exploratory development points, with repeated model selection on eight public seeds/both seats. Strong play, expert/superhuman strength and reference normalization remain unqualified. The larger published32-seed public bank has a different incumbent score36.9792; compare candidates only with their matching schedule. The older four-opponent28.125 score also uses different seeds/opponents and is not a before/after learning gain. Final primary selection and common private assessment are evaluator-owned and freeze before any fresh private seeds. Starting parent, latest running state, best validated saved state and original incumbent remain distinct in coordination/JSON.

The exact frozen task uses7024 public integers,2307 masked actions, all200 entity rows and the full natural180+120second horizon. Policies are stateless pure neural direct-argmax exports, without test-time teacher/search/LLM. Original spatial models have151,439 trainable parameters. Conditioned models retain151,439 trainable parameters and add224 fixed buffer elements (151,663 persistent tensor elements under the bundle cap).

The separate new ledger uses uncapped Modal dollars/aggregateGPU seconds with fixedH100!/32physicalCPU/128GiB and at most two concurrent/unresolved single-GPU allocations per game. Setup, proof audits, transfers, learning and idle allocation time remain charged. External paid model calls are zero. No old ledger or failed attempt was reset.

Current role accounting: {a['gpu_seconds_settled']:.3f} settled GPU seconds +{a['gpu_seconds_held']:.3f} held; ${a['resource_usd_settled']:.6f} interval-settled resources +${a['resource_usd_held']:.6f} conservative holds. Unknown image-build/observer/storage charges stay held; totals are not invoice verified. Provider intervals and operation IDs are in the JSON companion. Extra discarded next-update audit work is included in full allocations, separately from logical learner counters.

The v5 branch restarts optimizer/world/RNG state explicitly from the qualified v4 weights and changes public-pool sampling. Renewed PublicTeacher assistance regressed. Standard fixed-tie scripts beat PublicTeacher81.25% on the published16-game script comparison, motivating v7 Standard demonstrations. Longer teacher-off and assisted runs also regressed, including a0-point late teacher-off checkpoint; those checkpoints and complete failures remain reported.

The v9 branch regularizes entropy over wait/card marginals; a gradient fixture demonstrates that full joint-action entropy otherwise adds deployment pressure from hundreds of tile choices. Its policy architecture and decoder remain unchanged. This is a one-seed experiment, not an attributed general method gain. The v10 branch numerically conditions tiny public header/card-cost inputs using power-of-two gains and compensated weights, requiring exact parent/new logits, values and greedy actions before learning. v11 additionally reuses public Standard labels already collected on learner-visited states, with explicit model-only initialization and equal independent/learner-state supervision mass. Each new source is separately frozen; an actual GPU proof and public assessment are required before scaling.

Complete downloaded runs independently validate archived source, engine/build/runtime binding, all native-world payload checksums, optimizer/RNG/frozen-history lineage, final tensor bytes and actual GPU next-update replay. Exports compile hierarchical card/wait then tile greediness into an explicit flat-argmax neural head, retaining source provenance and equivalence fixtures. Export softmax is not the original PPO sampling distribution.

| Public candidate | Games | Points/100 |
|---|---:|---:|
{chr(10).join(rows)}

All raw opponent cells and execution diagnostics remain under `{OWN.relative_to(ROOT)}`. The public panel is Standard, Pressure, Intermediate, PublicTeacher, retained DeepSeek and the frozen spatial incumbent. Isolated Linux repeats preserve matching public outcomes where compared. CPU batch mentor-control-02 hit an administrative pair-controller deadline, has no aggregate and was retained; the later controller uses a larger finite pair limit. A prelaunch incorrect checksum was caught before GPU creation; its CPU-build hold and authoritative no-allocation proof remain retained. The earlier v7 long observer timed out after3590seconds because a247second GPU queue consumed its lifetime; the actual GPU completed under its own finite timeout and its full3455.055621allocated seconds settled. Future stages cap queue time at60seconds so the finite observer covers the whole allocation. Unpriced observer charges stay held. Collection directory streams also expired during long transfers; metadata is now fully drained before hash-checked downloads, with original failures retained. None of these is relabeled as a simulator fault or silently omitted.

Observed incumbent execution over96 public games:111,792 decisions,3,461 nonwait attempts,3,439 placed cards, zero request refusals and22 advancing-decision delayed refusals. This points toward tactical/card/placement weaknesses; imitation accuracy alone did not establish improved gameplay.

Read-only status: `.venv-clash-ar/bin/python -m simulators.clash.reference_training.strong.status`. Independent validation: `.venv-clash/bin/python -m simulators.clash.reference_training.strong.validate --name ATTEMPT`. Checks: `.venv-clash/bin/python -m pytest simulators/clash/reference_training/tests -q`. New controllers use `strong.provider`, `strong.fetch`, `strong.dev_remote` and `strong.cleanup`; the old capped campaign controllers remain historical and must not launch this campaign's work.

An overlapping direct fetch/background collection collided on a local temporary filename. The completed milestone's bytes matched its authoritative checksum; per-process temporary names and per-run collection locks now prevent the collision, with two targeted concurrency checks passing. Training workers were not interrupted, and no source, checksum or payload check was waived.

No commit, push, publication, commercial-ladder or production action. Checkpoints/private evidence remain ignored. Cleanup retains artifacts and unpriced holds and stops only this role's recorded new-campaign apps after compute reconciliation.
'''
 dest.with_suffix('.md').write_text(text);return {'json':str(dest.with_suffix('.json')),'markdown':str(dest.with_suffix('.md'))}

if __name__=='__main__':print(json.dumps(report(),indent=2))
