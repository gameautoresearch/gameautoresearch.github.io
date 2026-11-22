import copy
import unittest
from tools.analyze_paired_recipes import analyze


def rows():
    return [dict(family=family, train_seed=s, eval_unit=u, opponent=op, seat_or_role=seat,
                 score=30+s+u+(10 if family=='agent' else 0), valid=True)
            for family in ('starter','agent') for s in range(3) for u in range(4)
            for op in ('a','b') for seat in (0,1)]


class PairedTests(unittest.TestCase):
    def test_known_constant_paired_gain(self):
        r=analyze(rows(),replicates=200)
        self.assertEqual(r['paired_delta_points'],10)
        self.assertEqual(r['paired_delta_ci95'],[10,10])
        self.assertEqual(r['games_per_family'],48)
        self.assertEqual(r['per_opponent_delta_points'],{'a':10,'b':10})

    def test_training_seed_variance_is_resampled(self):
        data=rows()
        for r in data:
            if r['family']=='agent':r['score']+=(-20,0,20)[r['train_seed']]
        out=analyze(data,replicates=1000)
        self.assertLess(out['paired_delta_ci95'][0],0)
        self.assertGreater(out['paired_delta_ci95'][1],10)

    def test_partial_arm_is_not_silently_dropped(self):
        with self.assertRaises(ValueError):analyze(rows()[:-1],replicates=100)

    def test_missing_role_from_both_arms_still_fails(self):
        data=[r for r in rows() if not(r['eval_unit']==0 and r['seat_or_role']==1)]
        with self.assertRaises(ValueError):analyze(data,replicates=100)

    def test_missing_units_in_one_training_seed_fails(self):
        data=[r for r in rows() if not(r['train_seed']==2 and r['eval_unit']==0)]
        with self.assertRaises(ValueError):analyze(data,replicates=100)

    def test_fault_duplicate_and_nonfinite_rows_fail(self):
        for mutate in ('fault','duplicate','nan'):
            data=copy.deepcopy(rows())
            if mutate=='fault':data[0]['valid']=False
            elif mutate=='duplicate':data.append(data[0])
            else:data[0]['score']=float('nan')
            with self.assertRaises(ValueError):analyze(data,replicates=100)


if __name__=='__main__':unittest.main()
