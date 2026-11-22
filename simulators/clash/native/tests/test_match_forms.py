"""Match forms (match_forms.py): every binary runs the cadence and timeline its form names.

The 10-tick cadence is checked against the pinned Java oracle by
`test_world.py --form cadence10`; Java takes no timeline override, so the short-v1
timeline is checked here: its phase and elixir-step boundaries, and that until its first
elixir step it plays exactly like the pinned timeline at the same cadence (only the
regulation countdown field differs). Native CPU only; no Java or GPU.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

TIME_MS, SECTION_LEFT_MS, FULL_BAR_MS, OVERTIME, DECISION_TICKS = 55, 58, 59, 60, 50
TICK, WINNER, FAULT, ILLEGAL0, ILLEGAL1, TRUNCATED = 0, 2, 5, 7, 8, 9


def wait_match(form, seed=4242):
    from runtime import NativeCPU
    env = NativeCPU([seed], form=form)
    rows = []
    try:
        while True:
            o, _, info, _ = env.read()
            rows.append((int(info[0, TICK]), int(o[0, 0, TIME_MS]), int(o[0, 0, SECTION_LEFT_MS]),
                         int(o[0, 0, FULL_BAR_MS]), int(o[0, 0, OVERTIME]), int(o[0, 0, DECISION_TICKS])))
            if info[0, WINNER] >= 0:
                return rows, info[0].copy()
            if len(rows) > 3000:
                raise AssertionError('wait match did not end naturally')
            env.step(np.zeros((1, 2), np.int32))
    finally:
        env.close()


class MatchForms(unittest.TestCase):
    def test_binaries_report_their_form(self):
        import match_forms
        from build import build
        libraries = {}
        for name in ('full', 'short-v1', 'cadence10'):
            library = build(name)
            libraries[name] = library.name
            self.assertEqual(match_forms.check_library(library, name)['decision_ticks'],
                             match_forms.VARIANTS[name]['decision_ticks'])
        self.assertEqual(len(set(libraries.values())), 3)
        self.assertEqual(match_forms.flags('full'), [])
        self.assertEqual(match_forms.describe('short-v1')['decisions_per_second'], 2)
        self.assertEqual((match_forms.describe('short-v1')['regulation_seconds'],
                          match_forms.describe('short-v1')['overtime_seconds']), (60, 30))

    def test_pinned_form_wait_match(self):
        rows, info = wait_match('full')
        self.assertEqual(len(rows) - 1, 1230)
        self.assertEqual(int(info[TICK]), 6147)
        self.assertTrue(all(r[5] == 5 for r in rows))

    def test_short_timeline_boundaries(self):
        rows, info = wait_match('short-v1')
        self.assertEqual(int(info[WINNER]), 2)  # draw after the tiebreak
        self.assertFalse(info[FAULT] or info[ILLEGAL0] or info[ILLEGAL1] or info[TRUNCATED])
        # Ticks advance ten per decision until the natural end.
        ticks = [r[0] for r in rows]
        self.assertTrue(all(b - a == 10 for a, b in zip(ticks, ticks[1:-1])))
        self.assertTrue(all(r[5] == 10 for r in rows))
        played = [r for r in rows if r[0] > 0 and r[1] < 90000]
        for tick, time_ms, left, bar, overtime, _ in played:
            self.assertEqual(time_ms, (tick - 1) * 50)  # the timeline's time at the last stepped tick
            self.assertEqual(bar, 28000 if time_ms < 40000 else 14000 if time_ms < 75000 else 9300, time_ms)
            self.assertEqual(overtime, int(time_ms >= 60000), time_ms)
            self.assertEqual(left, 60000 - time_ms if time_ms < 60000 else 90000 - time_ms, time_ms)
        self.assertTrue(90 * 20 <= int(info[TICK]) <= 100 * 20)
        self.assertEqual(len(rows) - 1, 195)

    def test_short_matches_pinned_timeline_before_first_elixir_step(self):
        """Same cadence and seed, same random legal actions: identical until 40 s but the countdown."""
        from runtime import NativeCPU
        rng = np.random.default_rng(11)
        envs = [NativeCPU([771], form=name) for name in ('cadence10', 'short-v1')]
        try:
            compared = 0
            while True:
                (oa, ma, ia, ra), (ob, mb, ib, rb) = [e.read() for e in envs]
                if oa[0, 0, TIME_MS] >= 40000:
                    break
                keep = np.ones(oa.shape[-1], bool)
                keep[SECTION_LEFT_MS] = False
                np.testing.assert_array_equal(oa[..., keep], ob[..., keep])
                np.testing.assert_array_equal(ma, mb)
                np.testing.assert_array_equal(ia, ib)
                np.testing.assert_array_equal(ra, rb)
                self.assertEqual(int(ob[0, 0, SECTION_LEFT_MS]), 60000 - int(ob[0, 0, TIME_MS]))
                actions = np.array([[int(rng.choice(np.flatnonzero(ma[0, s]))) for s in (0, 1)]], np.int32)
                for e in envs:
                    e.step(actions)
                compared += 1
            self.assertEqual(compared, 81)  # decisions 0..80; the timeline reaches 40 s at tick 801
        finally:
            for e in envs:
                e.close()


if __name__ == '__main__':
    unittest.main()
