"""Real-state loss/gradient and RNG tests for structural execution changes."""
import copy,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from runner import setup,network,Timer,sample,eval_policy,collect,Opponents,WorkerClock
from actor import sample_actions,predict_values
from game import HERE,make_vector,Session

class Efficiency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=json.loads((HERE/'configs/profile-trimmed.json').read_text())
        cls.c.update(envs=4,threads=2,device='cpu')
        setup(cls.c)
        with make_vector(cls.c) as v:
            session=Session(v,cls.c,np.random.default_rng(17))
            snapshots=[];masks=[];rng=np.random.default_rng(19)
            for t in range(720):
                actions=np.array([[rng.choice(np.flatnonzero(m)) for m in row] for row in v.masks],np.int32)
                session.step(actions);session.reset_done()
                if t%60==0:
                    snapshots.append(v.observations.copy().reshape(-1,v.schema.seat_ints));masks.append(v.masks.copy().reshape(-1,v.schema.actions))
            cls.obs=np.concatenate(snapshots);cls.masks=np.concatenate(masks)
            cls.model=network(v,cls.c);cls.schema=v.schema
    def compare_gradients(self,a,b,batch,terms=False):
        actions=torch.tensor(np.argmax(self.masks[:len(batch['header'])],axis=1))
        results=[]
        for model in (a,b):
            if terms:
                logp,entropy,value=model.action_terms(batch,actions)
            else:
                dist,value=model.distribution(batch);logp=dist.log_prob(actions);entropy=dist.entropy()
            loss=(logp*.17+entropy*.013+value.square()).mean();loss.backward()
            results.append((logp,entropy,value))
        for x,y in zip(*results):torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-6)
        for (name,x),(_,y) in zip(a.named_parameters(),b.named_parameters()):
            self.assertIsNotNone(x.grad,name);self.assertIsNotNone(y.grad,name)
            torch.testing.assert_close(x.grad,y.grad,rtol=3e-4,atol=3e-6,msg=name)
    def test_live_encoding_values_gradients_and_max_ties(self):
        a=copy.deepcopy(self.model);b=copy.deepcopy(a);b.ragged_entities=True
        batch=a.pack(self.obs,self.masks)
        # Both zero entities and identical entities exercise empty/tied-max gradients.
        batch['header'][0,self.schema.header['entity_count'][0]]=0
        batch['entities'][1,1:]=batch['entities'][1,0].clone()
        self.compare_gradients(a,b,batch)
    def test_sparse_terms_and_empty_head_keep_adam_gradients(self):
        for forced in (False,True):
            a=copy.deepcopy(self.model);b=copy.deepcopy(a)
            batch=a.pack(self.obs,self.masks)
            if forced:batch['mask'][:]=False;batch['mask'][:,0]=True
            # Compare full categorical vs sparse loss, including forced-row constants.
            actions=batch['mask'].to(torch.int32).argmax(1)
            da,va=a.distribution(batch);la=da.log_prob(actions);ea=da.entropy()
            lb,eb,vb=b.action_terms(batch,actions)
            for x,y in ((la,lb),(ea,eb),(va,vb)):torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-6)
            (la.mean()+ea.mean()*.01+va.square().mean()).backward()
            (lb.mean()+eb.mean()*.01+vb.square().mean()).backward()
            for name,x in a.named_parameters():
                y=dict(b.named_parameters())[name]
                self.assertIsNotNone(y.grad,name)
                torch.testing.assert_close(x.grad,y.grad,rtol=3e-4,atol=3e-6,msg=name)
    def test_deferred_values_and_actor_rng(self):
        m=copy.deepcopy(self.model);c=self.c;timer=Timer('cpu')
        torch.manual_seed(123)
        a,lp,v=sample(m,self.obs,self.masks,c,timer);rng=torch.get_rng_state()
        torch.manual_seed(123)
        aa,ll=sample_actions(m,self.obs,self.masks,c,timer)
        np.testing.assert_array_equal(a,aa);torch.testing.assert_close(rng,torch.get_rng_state())
        np.testing.assert_allclose(lp,ll,rtol=2e-5,atol=2e-6)
        vv=predict_values(m,self.obs,dict(c,value_batch_size=17),timer)
        np.testing.assert_allclose(v,vv,rtol=2e-5,atol=2e-6)
    def test_rollout_preallocation_snapshots_and_boundaries(self):
        # A direct-buffer step must not overwrite any previously collected row.
        c=dict(self.c,rollout_steps=12,opponent_mode='selfplay')
        outputs=[];states=[]
        with make_vector(c) as v:
            for preallocated in (False,True):
                rng=setup(c);m=network(v,c);session=Session(v,c,rng)
                opponents=Opponents(v,c);opponents.select(session,range(v.count))
                data=collect(m,session,opponents,dict(c,preallocated_rollout=preallocated),Timer('cpu'),WorkerClock(100))
                outputs.append(data);states.append(session.state())
            for key in outputs[0]:np.testing.assert_array_equal(outputs[0][key],outputs[1][key])
            self.assertEqual(states[0]['observation_sha256'],states[1]['observation_sha256'])
            # Keep a retained observation while overwriting the JVM's direct buffers.
            before=outputs[1]['obs'].copy();v.actions[:]=0;v.step()
            np.testing.assert_array_equal(outputs[1]['obs'],before)

    def test_vector_bookkeeping_keeps_replay_and_counters(self):
        c=dict(self.c);rng=np.random.default_rng(203)
        with make_vector(c) as reference,make_vector(c) as candidate:
            a=Session(reference,c,np.random.default_rng(5))
            b=Session(candidate,dict(c,vector_bookkeeping=True),np.random.default_rng(5))
            for _ in range(100):
                actions=np.array([[rng.choice(np.flatnonzero(mask)) for mask in row] for row in reference.masks],np.int32)
                a.step(actions);b.step(actions)
            self.assertEqual(a.histories,b.histories);self.assertEqual(a.counters,b.counters)
            self.assertEqual(a.state()['observation_sha256'],b.state()['observation_sha256'])

    def test_eval_forced_skip_preserves_per_job_streams(self):
        m=copy.deepcopy(self.model);timer=Timer('cpu')
        for greedy in (False,True):
            a_rng=[np.random.default_rng(i) for i in range(len(self.obs))];b_rng=copy.deepcopy(a_rng)
            c=dict(self.c,eval_greedy=greedy)
            a=eval_policy('neural',self.obs,self.masks,m,None,a_rng,c,timer)
            b=eval_policy('neural',self.obs,self.masks,m,None,b_rng,dict(c,eval_skip_forced=True),timer)
            np.testing.assert_array_equal(a,b)
            self.assertEqual([x.bit_generator.state for x in a_rng],[x.bit_generator.state for x in b_rng])

if __name__=='__main__':unittest.main(verbosity=2)
