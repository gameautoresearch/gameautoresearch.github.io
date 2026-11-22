"""Conservative native-stream exclusions; never create or reveal a final bank."""
import json
from pathlib import Path
from .release import ROOT,sha
from .seed_keys import lobby_key,direct_game_key


def collect(configs=(),roots=None):
    configs=list(configs);keys={lobby_key(seed,g) for seed in (1,2026,30101,30102,30103,12345) for g in range(256)};receipts=[]
    roots=list(roots) if roots is not None else [*sorted((ROOT/'evidence').glob('*/tavern')),ROOT/'simulators/tavern/recipes',ROOT/'simulators/tavern/reference_training']

    def declarations(value):
        used=False
        if isinstance(value,dict):
            # Legacy films used direct Game(seed); newer films use BatchEnv's
            # derived seed. Exclude both interpretations conservatively.
            for name in ('seed','demoSeed','public_seed'):
                if type(value.get(name)) is int:
                    keys.add(lobby_key(value[name]));keys.add(direct_game_key(value[name]));used=True
            if all(type(value.get(k)) is int for k in ('seed','n_games','updates')):
                if value['n_games']<=0 or value['updates']<0:raise ValueError('Invalid declared training-stream bound')
                configs.append({k:value[k] for k in ('seed','n_games','updates')});used=True
            if isinstance(value.get('seeds'),list):
                games=max(1,int(value.get('games_per_seat',value.get('n_games',1))))
                for seed in value['seeds']:
                    if type(seed) is int:
                        keys.update(lobby_key(seed,g) for g in range(games));used=True
            for child in value.values():used=declarations(child) or used
        elif isinstance(value,list):
            for child in value:used=declarations(child) or used
        return used

    for config in list(configs):declarations(config.get('evaluation',{}))
    for root in roots:
        for path in sorted(Path(root).rglob('*.json')):
            if 'traces' in path.parts or path.name in ('episodes.json','result.json','last-status.json') or path.stat().st_size>2000000:continue
            try:value=json.loads(path.read_text())
            except (ValueError,OSError):continue
            if declarations(value):receipts.append(dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),sha256=sha(path)))
    configs=list({(c['seed'],c['n_games'],c['updates']):{k:c[k] for k in ('seed','n_games','updates')} for c in configs}.values())
    return keys,configs,dict(schedule_declarations=receipts,excluded_native_schedule_keys=len(keys),training_cohorts_excluded=configs,entire_declared_update_ceilings=True,direct_Game_and_BatchEnv_public_seed_interpretations_excluded=True)
