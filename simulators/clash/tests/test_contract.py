"""Real-table contract checks; all run locally, no paid compute."""
import copy,hashlib,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from game import HERE,Session,make_vector,status,canonical
from runner import advantages,network,setup,Timer,ppo_update,WorkerClock

class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=json.loads((HERE/'configs/smoke.json').read_text());cls.c.update(envs=4,threads=2)
        setup(cls.c);cls.v=make_vector(cls.c);cls.s=cls.v.schema
    @classmethod
    def tearDownClass(cls):cls.v.close()
    def session(self,seed=7):return Session(self.v,self.c,np.random.default_rng(seed))
    def test_engine_determinism_across_threads_and_both_seats(self):
        v=self.v;s=self.session();rng=np.random.default_rng(98);history=[];digests=[]
        with make_vector(dict(self.c,threads=1)) as other:
            other.reset(s.seeds,self.c['rules']['deck'],self.c['rules']['deck'])
            for _ in range(96):
                a=np.array([[rng.choice(np.flatnonzero(m)) for m in row] for row in v.masks],np.int32)
                v.step(a);other.step(a)
                for name in ('observations','masks','infos','rewards','dones'):
                    np.testing.assert_array_equal(getattr(v,name),getattr(other,name))
            self.assertFalse(status(v)['fault'].any())
            self.assertFalse(status(v)['unsupported'].any())
    def test_resume_reconstructs_uneven_histories(self):
        session=self.session();rng=np.random.default_rng(41)
        for t in range(15):
            a=np.array([[rng.choice(np.flatnonzero(m)) for m in row] for row in self.v.masks],np.int32)
            session.step(a)
            if t==6:
                session.seeds[0]=784;session.histories[0]=[];self.v.reset_env(0,784)
            if t==14:
                session.seeds[3]=785;session.histories[3]=[];self.v.reset_env(3,785)
        state=copy.deepcopy(session.state());obs=self.v.observations.copy();mask=self.v.masks.copy()
        session.restore(state)
        np.testing.assert_array_equal(self.v.observations,obs);np.testing.assert_array_equal(self.v.masks,mask)
        self.assertEqual(canonical(session.state()['rng']),canonical(state['rng']))
    def test_entity_trim_exact_and_masked_sampling(self):
        self.session();m=network(self.v,self.c);rng=np.random.default_rng(12)
        # Create nontrivial observations, both seats, and padded buffers.
        for _ in range(30):
            a=np.array([[rng.choice(np.flatnonzero(x)) for x in row] for row in self.v.masks],np.int32);self.v.step(a)
        dense=m.pack(self.v.observations,self.v.masks,trim=False)
        compact=m.pack(self.v.observations,self.v.masks,trim=True)
        self.assertLess(compact['entities'].shape[1],dense['entities'].shape[1])
        l1,v1=m(dense);l2,v2=m(compact)
        torch.testing.assert_close(l1,l2,rtol=1e-5,atol=1e-6);torch.testing.assert_close(v1,v2,rtol=1e-5,atol=1e-6)
        for _ in range(30):
            a=m.distribution(compact)[0].sample().numpy()
            self.assertTrue(self.v.masks.reshape(-1,self.s.actions)[np.arange(8),a].all())
        # Masked padding cannot alter a pooled state.
        n=int(compact['header'][:,self.s.header['entity_count'][0]].max())
        dense['entities'][:,n:]=12345
        torch.testing.assert_close(m(dense)[1],v1)
    def test_gae_terminal_truncation_and_fault(self):
        r=np.array([[1.,2.,50.]],np.float32);v=np.array([[.2,.3,.4]],np.float32)
        done=np.ones((1,3),bool);tv=np.array([[0.,.8,0.]],np.float32);valid=np.array([[True,True,False]])
        adv,ret=advantages(r,v,done,tv,np.array([9.,9.,9.]),valid,.9,.95)
        np.testing.assert_allclose(ret,[[1.,2.72,.4]],atol=1e-6)
        adv,ret=advantages(np.zeros((1,1)),np.array([[.1]]),np.zeros((1,1),bool),np.zeros((1,1)),np.array([.8]),np.ones((1,1),bool),.9,.95)
        np.testing.assert_allclose(ret,[[.72]])
    def test_real_time_limit_not_a_loss(self):
        c=copy.deepcopy(self.c);c['rules']['max_ticks']=10
        # Explicit lower-level test of upstream truncation, outside the pinned benchmark task.
        from crforge_env import VectorBattle,LadderConfig
        from game import resolve
        with VectorBattle(2,1,install=resolve(c['install']),tables=resolve(c['tables']),config=LadderConfig(max_ticks=10)) as v:
            v.set_autoreset(False);v.reset([13,14],c['rules']['deck'],c['rules']['deck'])
            v.actions[:]=0;v.step();v.step();st=status(v)
            self.assertTrue(v.dones.all());self.assertTrue(st['truncated'].all());self.assertFalse(st['fault'].any())
            self.assertTrue((st['winner']==v.schema.winner_none).all());self.assertEqual(float(v.rewards.sum()),0.)
    def test_fault_channel_not_game_outcome(self):
        from crforge_env import VectorBattle
        from game import resolve
        with VectorBattle(2,1,install=resolve(self.c['install']),engine='synthetic',synthetic_fault_at=1) as v:
            v.set_autoreset(False);v.reset([11,12],self.c['rules']['deck'],self.c['rules']['deck'])
            v.actions[:]=0;v.step();st=status(v)
            self.assertTrue(v.dones.all());self.assertTrue(st['fault'].all());self.assertTrue(st['truncated'].all())
            self.assertTrue((st['winner']==v.schema.winner_none).all())
    def test_shared_direct_buffers(self):
        self.session()
        self.assertFalse(self.v.observations.flags['OWNDATA']);ptr=self.v.observations.__array_interface__['data'][0]
        self.v.actions[:]=0;self.v.step()
        self.assertEqual(ptr,self.v.observations.__array_interface__['data'][0])
    def test_actual_learning_on_real_teacher_decisions(self):
        """Supervised diagnostic: learn real legal ScriptedBot decisions; not game-strength evidence."""
        from crforge_env import ScriptedBot
        self.session();bot=ScriptedBot().bind(self.v);rng=np.random.default_rng(20);m=network(self.v,self.c)
        obs=[];masks=[];targets=[]
        for _ in range(240):
            a=np.array([[bot(o,mask,rng) for o,mask in zip(row,ms)] for row,ms in zip(self.v.observations,self.v.masks)],np.int32)
            for i in range(self.v.count):
                for seat in (0,1):
                    if self.v.masks[i,seat].sum()>1:
                        obs.append(self.v.observations[i,seat].copy());masks.append(self.v.masks[i,seat].copy());targets.append(a[i,seat])
            self.v.step(a)
            for i in np.flatnonzero(self.v.dones):self.v.reset_env(int(i),int(rng.integers(100000)))
        b=m.pack(np.array(obs),np.array(masks));a=torch.tensor(targets,dtype=torch.int64)
        opt=torch.optim.Adam(m.parameters(),lr=.003)
        with torch.no_grad():before=float(-m.distribution(b)[0].log_prob(a).mean())
        for _ in range(45):
            d,_=m.distribution(b);loss=-d.log_prob(a).mean();opt.zero_grad();loss.backward();opt.step()
        after=float(-m.distribution(b)[0].log_prob(a).mean().detach())
        self.assertGreater(len(targets),20);self.assertLess(after,before*.7)
        self.learning={'examples':len(targets),'teacher_nll_before':before,'teacher_nll_after':after,'scope':'real-state optimization diagnostic; not held-out policy strength'}
        print('LEARNING_DIAGNOSTIC',json.dumps(self.learning),flush=True)

if __name__=='__main__':unittest.main(verbosity=2)
