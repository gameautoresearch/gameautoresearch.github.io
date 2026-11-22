import unittest
from ..scoring import summarize
from ..foul_play import translate,FoulPlay

class CompletionTests(unittest.TestCase):
 def row(self,name='power',index=0,seat=0,points=0,natural=True,**kwargs):
  return dict(opponent=name,seed_index=index,seat=seat,points=points,natural=natural,**kwargs)
 def test_valid_zero_retained(self):
  r=summarize([self.row()],1,[('power',0,0)])
  self.assertEqual(r['raw_score_0_100'],0);self.assertEqual(r['total_point_bounds_0_100'],[0,0])
 def test_caps_faults_and_unstarted_bound_every_cell(self):
  rows=[self.row(points=1),self.row(index=1,points=None,natural=False,truncated=True),self.row(name='other',points=None,natural=False,fault='broken',fault_category='infrastructure_failure')]
  schedule=[('power',0,0),('power',1,0),('other',0,0),('other',1,0)]
  r=summarize(rows,4,schedule)
  self.assertIsNone(r['raw_score_0_100']);self.assertEqual(r['total_point_bounds_0_100'],[25,100]);self.assertEqual(r['unstarted_games'],1)
  self.assertTrue(all(g['raw_score_0_100'] is None for g in r['by_opponent_and_seat'].values()))
 def test_natural_tie_only_not_administrative_draw(self):
  r=summarize([self.row(points=.5)],1,[('power',0,0)])
  self.assertEqual(r['natural_ties'],1);self.assertEqual(r['raw_score_0_100'],50)
  r=summarize([self.row(points=None,natural=False,truncated=True)],1,[('power',0,0)])
  self.assertEqual(r['natural_ties'],0);self.assertIsNone(r['raw_score_0_100'])
 def test_duplicate_rows_rejected(self):
  with self.assertRaises(ValueError):summarize([self.row(),self.row()],2)

class TranslationTests(unittest.TestCase):
 def request(self):return {'side':{'pokemon':[{'details':'Pikachu, L100'},{'details':'Eevee, L100'}]},'active':[{'moves':[{'id':'thunderbolt','move':'Thunderbolt'},{'id':'terablast','move':'Tera Blast'}]}]}
 def test_move_tera_switch_and_struggle(self):
  r=self.request();mask=[True]*14
  self.assertEqual(translate('terablast-tera',r,mask),5)
  self.assertEqual(translate('switch eevee',r,mask),9)
  r['active'][0]['moves']=[{'id':'struggle','move':'Struggle'}]
  self.assertEqual(translate('struggle',r,mask),0)
 def test_forced_switch_and_trapping_never_get_fallback(self):
  r=self.request();mask=[False]*14;mask[9]=True
  self.assertEqual(translate('switch 2',r,mask),9)
  with self.assertRaises(ValueError):translate('thunderbolt',r,mask)
  mask[9]=False
  with self.assertRaises(ValueError):translate('switch 2',r,mask)
 def test_private_extra_or_foreign_request_rejected(self):
  bot=FoulPlay.__new__(FoulPlay)
  with self.assertRaises(ValueError):bot.choose({'actual_hidden_team':['secret']})
  visible={'seat':0,'request':{'side':{'id':'p2'}},'protocol':[],'protocol_offset':0,'mask':[True]*14,'turn':1,'termination_policy':'test'}
  with self.assertRaises(ValueError):bot.choose(visible)

if __name__=='__main__':unittest.main()
