"""Optional standards-compliant single-seat Gymnasium adapter (trainer uses BatchEnv)."""
import json
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import taverngym as tg

STOP_REASONS = {0: "ongoing", 1: "natural", 2: "no_progress", 3: "driver_guard",
                4: "turn_limit", 5: "combat_cutoff", 6: "recruit_cutoff"}


def state_fault(state):
    if state.get("stopped_reason"):
        return state["stopped_reason"]
    if state.get("turn_limit_end"):
        return "turn_limit"
    if state.get("truncated_combats", 0):
        return "combat_cutoff"
    if state.get("truncated_recruit_contexts", 0):
        return "recruit_cutoff"
    return None


class TavernEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None, seat=0):
        self.env = tg.BattlegroundsEnv(config=config, agent=seat)
        self.action_space = spaces.Discrete(tg.N_ACTIONS)
        self.observation_space = spaces.Dict({
            "features": spaces.Box(-np.inf, np.inf, (tg.OBS_FLOATS,), np.float32),
            "cards": spaces.Box(0, tg.N_CARDS - 1, (tg.OBS_CARDS,), np.int32),
            "action_mask": spaces.Box(0, 1, (tg.N_ACTIONS,), np.bool_),
        })
        self.ended = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ended = False
        if seed is None:
            seed = int(self.np_random.integers(0, 2**63))
        return self.env.reset(seed=int(seed))

    def step(self, action):
        if self.ended:
            raise RuntimeError("reset required after termination/truncation")
        obs, reward, done, _, info = self.env.step(action)
        state = json.loads(self.env.state_json())
        fault = state_fault(state)
        truncated = fault is not None
        terminated = bool(done and not truncated)
        self.ended = terminated or truncated
        if truncated:
            reward, info["placement"] = 0.0, 0
        info["stop_reason"] = fault or ("natural" if terminated else "ongoing")
        return obs, reward, terminated, truncated, info
