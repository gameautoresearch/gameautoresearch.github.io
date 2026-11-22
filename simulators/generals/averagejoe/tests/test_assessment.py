"""The new assessor must actually honor the self-play general-trade rule."""
from dataclasses import replace
import jax
import jax.numpy as jnp
from simulators.generals.averagejoe import assessment
from simulators.generals.averagejoe.framework import recipe, make_env
from generals.core.game import create_initial_state


def test_assessment_rule_changes_simultaneous_general_capture(monkeypatch):
    cfg = replace(recipe(), pad_to=6, truncation=1)
    grid = jnp.zeros((6,6), jnp.int32).at[2,1].set(1).at[2,2].set(2)
    s = create_initial_state(grid)
    s = s._replace(armies=s.armies.at[1,2].set(10).at[1,1].set(10),
                   ownership=s.ownership.at[0,1,2].set(True).at[1,1,1].set(True))
    starts = jax.tree.map(lambda x:x[None], s)
    def move(obs):
        where = jnp.argwhere(obs.owned_cells, size=1)[0]
        return jnp.array([0, where[0], where[1], 1, 0], jnp.int32)
    monkeypatch.setattr(assessment, 'random_action', lambda key, obs: move(obs))
    # The first movable tile is each player's stack above the opposing general.
    def greedy(net, x, mask, temporal):
        where = jnp.argwhere(mask.any(-1), size=1)[0]
        return jnp.array([0, where[0], where[1], 1, 0], jnp.int32)
    monkeypatch.setattr(assessment, 'greedy_action_transformer', greedy)
    training_rule = make_env(recipe()).general_trade
    assert training_rule is False
    no_trade, _ = assessment.make_assessor(cfg, general_trade=training_rule)(None, starts, jax.random.PRNGKey(0))
    trade, _ = assessment.make_assessor(cfg, general_trade=True)(None, starts, jax.random.PRNGKey(0))
    assert int(no_trade['wins'])==1 and int(no_trade['losses'])==1 and int(no_trade['caps'])==0
    assert int(trade['wins'])==0 and int(trade['losses'])==0 and int(trade['caps'])==2
