"""Add true terminal-state retention around the frozen scoring driver."""
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import taverngym as tg
from .terminal_replay import shadow_step


class TerminalRecordingEnv(tg.BatchEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.n_games!=1 or self.n_slots!=int((kwargs.get('config') or {}).get('num_players',8)):raise ValueError('Exact single full-lobby evaluator required')
        self._record_finished=np.zeros(self.n_slots,bool);self._record_obs=None;self.terminal_game=None
    def observe(self,*args,**kwargs):
        obs=super().observe(*args,**kwargs);self._record_obs=obs;return obs
    def step(self,actions,*args,**kwargs):
        before=self.game(0);reward,done,info=super().step(actions,*args,**kwargs)
        if (self._record_finished|done).all() and not (info['stop_reason']>1).any():
            applied,deferred=shadow_step(before,self._record_obs,np.asarray(actions),self._record_finished)
            if not np.array_equal(applied,info['applied']) or not np.array_equal(deferred,info['deferred']) or not before.is_over():raise ValueError('Terminal-frame native admission differs')
            placements=before.placements()
            if sorted(placements)!=list(range(1,self.n_slots+1)) or any(placements[int(p)]!=int(info['placement'][p]) for p in np.flatnonzero(done)):raise ValueError('True terminal placement differs')
            self.terminal_game=before.state_json()
        self._record_finished|=done
        return reward,done,info


def install():
    """Only evaluator construction/recording changes; source/action/score stays fixed."""
    from .. import lobby,screen
    original_tg=lobby.tg;original_run=screen.run_lobby
    def run(*args,**kwargs):
        instances=[]
        def factory(*a,**k):
            env=TerminalRecordingEnv(*a,**k);instances.append(env);return env
        lobby.tg=SimpleNamespace(BatchEnv=factory,action_name=tg.action_name,build_info=tg.build_info)
        try:result=original_run(*args,**kwargs)
        finally:lobby.tg=original_tg
        if result['complete_valid_lobby']:
            if len(instances)!=1 or instances[0].terminal_game is None:raise ValueError('Missing exact terminal game recording')
            game=instances[0].terminal_game;state=json.loads(game)
            if any(int(p['placement'])!=result['terminals'][int(p['id'])]['placement'] for p in state['players']):raise ValueError('Recorded all8 terminals differ')
            digest=hashlib.sha256(game.encode()).hexdigest();result['actual_terminal_game_sha256']=digest;result['post_step_state_is_next_episode']=True
            trace=kwargs.get('trace')
            if trace:
                target=Path(trace)/'actual-terminal-game.json'
                if target.exists():raise FileExistsError(target)
                target.write_text(game);target.chmod(0o600)
                (Path(trace)/'terminal-state.json').write_text(json.dumps(dict(actual_terminal_game_sha256=digest,post_step_sha256=result['final_state_sha256'],all8_terminals_exact=True,source='Pinned native Game clone and exact final admission frame; full batch driver unchanged'),indent=2)+'\n')
        return result
    screen.run_lobby=run
    def undo():screen.run_lobby=original_run;lobby.tg=original_tg
    return undo
