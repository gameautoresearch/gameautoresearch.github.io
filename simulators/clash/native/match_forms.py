"""Benchmark match forms over the frozen fixed8 rules: decision cadence and match timeline.

The pinned crforge data (generated-v2/, rules.lock.json) is never edited. A form other than
`full` is compiled into its own library with explicit -D flags that become part of the build
identity, and every loaded binary reports the cadence and timeline it actually runs
(`cn_match_form`), which `check_library` compares with this registry.

`full` is the pinned Ladder form: 180 s regulation + 120 s overtime, elixir x2 at 120 s and
x3 at 240 s, one decision per 5 ticks (4 per simulated second). It compiles with no extra
flags, so its behaviour is the historical one.

`short-v1` is the shortened benchmark form: 60 s regulation + 30 s overtime, both phases a
third and a quarter of the pinned lengths, with the elixir/card-cycle steps moved in the same
proportion (x2 from 40 s, the last third of regulation; x3 from 75 s, half-way through
overtime), and one decision per 10 ticks (2 per simulated second). Everything else (elixir
amounts per step, start cooldown, deploy delay, tiebreak, end delay, arena, deck, levels,
observation and action layout) stays pinned.

`cadence10` changes only the cadence and keeps the pinned timeline. It exists so the Java
oracle, which takes `decision_ticks` but not a timeline override, can check the 10-tick
decision framing on complete matches (native/test_world.py --form cadence10).
"""
import ctypes as C
import json

TICK_MS = 50
# The pinned Ladder timeline, as generated-v2/config.hpp holds it (milliseconds).
PINNED_TIMELINE = {'section_ms': [180000, 120000], 'section_types': [0, 1],
                   'rate_ms': [120000, 120000, 60000], 'full_bar_ms': [28000, 14000, 9300],
                   'cooldown_ms': [120000, 120000, 60000], 'card_cooldown_ms': [1000, 500, 350]}
SHORT_V1_TIMELINE = {**PINNED_TIMELINE, 'section_ms': [60000, 30000],
                     'rate_ms': [40000, 35000, 15000], 'cooldown_ms': [40000, 35000, 15000]}

VARIANTS = {
    'full': {'decision_ticks': 5, 'timeline': PINNED_TIMELINE, 'timeline_override': False},
    'short-v1': {'decision_ticks': 10, 'timeline': SHORT_V1_TIMELINE, 'timeline_override': True},
    'cadence10': {'decision_ticks': 10, 'timeline': PINNED_TIMELINE, 'timeline_override': False,
                  'purpose': 'differential check of the 10-tick cadence against the Java oracle only'},
}


def variant(name):
    if name not in VARIANTS:
        raise ValueError(f'unknown Clash match form {name!r}; known: {", ".join(VARIANTS)}')
    return VARIANTS[name]


def describe(name):
    """Plain-data description of a form, for receipts, releases and bundles."""
    v = variant(name)
    t = v['timeline']
    return {'name': name, 'decision_ticks': v['decision_ticks'], 'tick_ms': TICK_MS,
            'decisions_per_second': 1000 // (TICK_MS * v['decision_ticks']),
            'regulation_seconds': t['section_ms'][0] // 1000, 'overtime_seconds': t['section_ms'][1] // 1000,
            'timeline_ms': dict(t)}


def flags(name):
    """Compiler flags for a form; `full` adds none, so it builds exactly as before."""
    v = variant(name)
    out = []
    if v['decision_ticks'] != 5:
        out.append(f'-DCN_DECISION_TICKS={v["decision_ticks"]}')
    if v['timeline_override']:
        t = v['timeline']
        if [len(t['section_ms']), len(t['rate_ms']), len(t['cooldown_ms'])] != [2, 3, 3]:
            raise ValueError('a timeline override keeps the pinned section/rate/cooldown counts')
        # One define per value: nvcc splits a -D value at commas.
        out.append('-DCN_TIMELINE_OVERRIDE=1')
        for letter, key in (('S', 'section_ms'), ('R', 'rate_ms'), ('C', 'cooldown_ms')):
            out += [f'-DCN_TIMELINE_{letter}{i}={value}' for i, value in enumerate(t[key])]
    return out


def library_suffix(name):
    variant(name)
    return '' if name == 'full' else '-' + name


def reported(library_or_cdll, prefix='cn'):
    """What a loaded native library says it runs: decision ticks and its rules timeline."""
    lib = library_or_cdll if isinstance(library_or_cdll, C.CDLL) else C.CDLL(str(library_or_cdll))
    fn = getattr(lib, prefix + '_match_form')
    fn.argtypes = [C.POINTER(C.c_int), C.c_int]
    fn.restype = C.c_int
    buf = (C.c_int * 64)()
    n = fn(buf, 64)
    if n < 2 or n > 64:
        raise ValueError('native library match form report is malformed')
    v = list(buf[:n])
    ticks, sections, rates, cooldowns = v[0], v[1], v[2], v[3]
    at = 4
    def take(k):
        nonlocal at
        out = v[at:at + k]
        at += k
        return out
    report = {'decision_ticks': ticks, 'section_ms': take(sections), 'section_types': take(sections),
              'rate_ms': take(rates), 'full_bar_ms': take(rates), 'cooldown_ms': take(cooldowns),
              'card_cooldown_ms': take(cooldowns)}
    return report


def check_library(library_or_cdll, name, prefix='cn'):
    """Refuse a binary whose compiled cadence or timeline differs from the named form."""
    v = variant(name)
    want = {'decision_ticks': v['decision_ticks'], **v['timeline']}
    got = reported(library_or_cdll, prefix)
    if got != want:
        raise ValueError(f'native library runs a different match form than {name!r}: '
                         + json.dumps({'expected': want, 'reported': got}, sort_keys=True))
    return got
