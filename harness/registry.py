"""The seven games. Each `harness/games/<game>.py` supplies SPEC and Backend.

SPEC is plain data and importing a game module must stay cheap: catalogs and
task builds run without any game or cloud dependency installed.
"""
import importlib

from harness import agents, budget
from harness.workspace import GAMES
from harness.broker import VARIANTS

SPEC_KEYS = ('game', 'venv', 'constructor', 'mode', 'hardware', 'final_train_seconds', 'smoke_final_train_seconds',
             'retrain_seed', 'parameter_cap')
# Every game offers both research budget modes (harness/budget.py) and declares the defaults of each, plus
# `budget_mode`, the mode a session gets when it names none. `wall_seconds` is the dollar mode's wall clock; a
# seconds-mode session's wall clock follows from its allocation (session.default_wall).
SECONDS_KEYS = ('research_seconds', 'smoke_research_seconds')
USD_KEYS = ('research_usd', 'smoke_research_usd', 'wall_seconds', 'smoke_wall_seconds')


def module(game):
    if game not in GAMES:
        raise ValueError('Unknown game: ' + str(game))
    return importlib.import_module('harness.games.' + game)


def spec(game):
    value = dict(module(game).SPEC)
    missing = [key for key in SPEC_KEYS + SECONDS_KEYS + USD_KEYS + ('budget_mode',) if key not in value]
    if missing or value['game'] != game:
        raise ValueError(f'{game}: incomplete SPEC: {missing}')
    if value['budget_mode'] not in budget.MODES:
        raise ValueError(f'{game}: budget_mode is one of {budget.MODES}')
    return value


def backend(game):
    return module(game).Backend


def constructor(game):
    name, function = spec(game)['constructor'].split(':')
    return getattr(importlib.import_module(name), function)


def research(game, smoke=False, budget_mode=None, research_seconds=None, research_usd=None):
    """The session's research budget: {'mode', 'seconds', 'usd'}, exactly one of the amounts set."""
    value = spec(game)
    mode = budget.choose(value, budget_mode, research_seconds, research_usd)
    prefix = 'smoke_' if smoke else ''
    if mode == budget.SECONDS:
        return dict(mode=mode, seconds=float(research_seconds or value[prefix + 'research_seconds']), usd=None)
    return dict(mode=mode, seconds=None, usd=float(research_usd or value[prefix + 'research_usd']))


def budgets(game, smoke=False, research_seconds=None, final_train_seconds=None, research_usd=None, budget_mode=None):
    """(research seconds or None in dollar mode, fresh final seconds); explicit values override the defaults."""
    value = spec(game)
    chosen = research(game, smoke, budget_mode, research_seconds, research_usd)
    return chosen['seconds'], float(final_train_seconds or value[('smoke_' if smoke else '') + 'final_train_seconds'])


def research_usd(game, smoke=False, research_usd=None, budget_mode=None, research_seconds=None):
    """The research budget in US dollars in dollar mode, else None."""
    return research(game, smoke, budget_mode, research_seconds, research_usd)['usd']



def catalog():
    games = {}
    for game in GAMES:
        value = spec(game)
        games[game] = dict(value, agents=list(agents.allowed(value)))
    return dict(harness='terminus-harness-1', default_agent=agents.TERMINUS, entrypoint='python -m harness',
                variants=list(VARIANTS), games=games)
