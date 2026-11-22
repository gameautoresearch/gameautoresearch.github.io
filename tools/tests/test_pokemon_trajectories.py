"""Artifact verification against owning primary ledgers, not renderer implementation."""
import json
import math
from pathlib import Path
import re
import sqlite3
import unittest

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/pokemon'
PREFIX='four-researcher-trajectories-2026-11-10'
ORIGINAL='pokemon-four-model-4h-20261109-1354Z'
CONT='pokemon-continuation-20261110-0456Z'
def read(p):return json.loads(p.read_text())

class TrajectoryEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=read(REPORT/(PREFIX+'.json'))
        cls.close=read(REPORT/(PREFIX+'-continuation.json'))
        cls.audit=read(REPORT/(PREFIX+'-audit.json'))

    def test_original_embedded_dataset_unchanged(self):
        html=(REPORT/'sol-kimi-research-behavior-2026-11-10.html').read_text()
        payload=re.search(r'<script id="data"[^>]*>(.*?)</script>',html,re.S).group(1)
        self.assertEqual(json.loads(payload),read(REPORT/'sol-kimi-research-behavior-2026-11-10.json'))

    def test_all_research_allocations_reconcile(self):
        for a in self.data['agents']:
            bases=[ORIGINAL]+([CONT]if a['id']in ['opus55-high','deepseek-v4-flash']else [])
            expected=0
            for base in bases:
                db=sqlite3.connect(f'file:{ROOT}/evidence/{base}/campaign.sqlite3?mode=ro',uri=True)
                rows=db.execute("select actual_seconds,status from pilot_jobs where arm=? and kind='research'",(a['id'],)).fetchall();db.close()
                self.assertTrue(all(status=='settled' for _,status in rows))
                expected+=sum(s for s,_ in rows)
            actual=sum(e['end']-e['start']for e in a['events']if e['include_in_aggregate'])
            self.assertAlmostEqual(actual,expected,places=5)
            self.assertLessEqual(actual,14400)

    def test_continuation_is_one_common_offset(self):
        offset=self.audit['continuation_display_origin_unix']-self.audit['origin_unix']
        for a in self.close['agents']:
            lifetime=next(x for x in self.data['agents']if x['id']==a['id'])
            parent={e['id']:e for e in lifetime['events']}
            for e in a['events']:
                self.assertAlmostEqual(e['start']+offset,parent[e['id']]['wall_start_seconds'],places=6)
                self.assertAlmostEqual(e['end']+offset,parent[e['id']]['wall_end_seconds'],places=6)
                self.assertEqual(e['include_in_aggregate'],parent[e['id']]['include_in_aggregate'])

    def test_failures_preserved_and_closed_gaps_removed(self):
        for aid,count in [('opus55-high',2),('deepseek-v4-flash',8)]:
            a=next(a for a in self.data['agents']if a['id']==aid)
            self.assertEqual(sum(e.get('status')=='failed / incomplete' and 'metrics' in e for e in a['events']),count)
            self.assertFalse(any(e['id'] in {'restart-gap','original-stop'} for e in a['events']))
            self.assertGreater(a['omitted_idle_seconds'],12*3600)
            resume=next(e for e in a['events']if e['id']=='session-resume')
            self.assertEqual(resume['start'],resume['end'])
        opus=next(a for a in self.data['agents']if a['id']=='opus55-high')
        m2=next(e for e in opus['events']if e.get('stage')=='m2')
        self.assertTrue(m2['include_in_aggregate']);self.assertAlmostEqual(m2['end']-m2['start'],3453.970594882965)

    def test_activity_clock_keeps_real_durations_on_one_scale(self):
        self.assertLessEqual(self.data['time_domain_seconds'],5*3600)
        for a in self.data['agents']:
            self.assertLessEqual(a['end_seconds'],self.data['time_domain_seconds'])
            self.assertAlmostEqual(a['end_seconds'],sum(end-start for start,end in a['clock_intervals']))
            for e in a['events']:
                if e['include_in_aggregate'] or e['bucket']=='model':
                    self.assertAlmostEqual(e['end']-e['start'],e['wall_end_seconds']-e['wall_start_seconds'],places=6)

    def test_no_raw_private_payload_fields(self):
        serialized=json.dumps(self.data)
        for forbidden in ['"private_bank":','"reasoning":','"reasoning_details":','Authorization:','api_key','gateway_auth']:
            self.assertNotIn(forbidden,serialized)
        for a in self.data['agents']:
            for e in a['events']:
                self.assertNotIn('seed_start',e)

if __name__=='__main__':unittest.main()
