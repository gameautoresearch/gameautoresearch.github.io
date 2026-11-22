#!/usr/bin/env python3
"""Keep the six agent rulebooks Markdown-only; preserve build/audit inputs separately."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import re
import shutil
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
GAMES = ('pokemon', 'clash', 'tavern', 'generals', 'doudizhu', 'polytopia')
FENCE = re.compile(r'^```([^\n]*)\n(.*?)^```[ \t]*$', re.M | re.S)
STATS = {'atk':'Attack','def':'Defense','spa':'Special Attack','spd':'Special Defense','spe':'Speed','accuracy':'Accuracy','evasion':'Evasion'}
NAMES = {'basePower':'Base power','baseMoveType':'Base move type','critRatio':'Critical-hit stage','pp':'Base PP',
 'volatileStatus':'Temporary condition','sideCondition':'Side condition','slotCondition':'Position effect','pseudoWeather':'Field effect',
 'boosts':'Stat-stage changes','self':'User','selfBoost':'User stat changes','secondary':'Additional effect','secondaries':'Additional effects',
 'chance':'Chance (%)','heal':'Healing fraction','drain':'Drain fraction','recoil':'Recoil fraction','multihit':'Number of hits',
 'ignoreImmunity':'Ignores type immunity','ignoreAbility':'Ignores the target’s ability','ignoreDefensive':'Ignores defensive stat stages',
 'ignoreEvasion':'Ignores evasion','willCrit':'Guaranteed critical hit','forceSwitch':'Forces the target to switch','selfSwitch':'User switches',
 'noPPBoosts':'PP cannot be increased','isChoice':'Locks the holder into a move','isBerry':'Berry','fling':'Fling',
 'naturalGift':'Natural Gift','itemUser':'Eligible holders','forcedForme':'Required form','breaksProtect':'Breaks protection',
 'hasCrashDamage':'Inflicts crash damage on failure','struggleRecoil':'Uses Struggle recoil','sleepUsable':'Usable while asleep',
 'callsMove':'Calls another move','stallingMove':'Shares the repeated-protection penalty','noCopy':'Cannot be copied',
 'suppressWeather':'Suppresses weather effects','thawsTarget':'Thaws the target','damage':'Damage','counterMax':'Counter maximum',
 'overrideOffensivePokemon':'Offensive Pokémon used','overrideOffensiveStat':'Offensive stat used','overrideDefensiveStat':'Defensive stat used'}
FLAGS = {'contact':'Makes contact','protect':'Blocked by Protect','breakable':'Can be bypassed by ability-ignoring effects',
 'punch':'Punching move','sound':'Sound move','bullet':'Ball or bomb move','bite':'Biting move','slicing':'Slicing move',
 'powder':'Powder move','wind':'Wind move','dance':'Dance move','reflectable':'Can be reflected','bypasssub':'Bypasses Substitute',
 'defrost':'Thaws the user','charge':'Has a charging turn','recharge':'Requires a recharge turn','cantusetwice':'Cannot be used twice consecutively',
 'gravity':'Prevented by Gravity','minimize':'Has a special interaction with Minimize'}
STATUS = {'brn':'Burn','psn':'Poison','tox':'Bad poison','par':'Paralysis','slp':'Sleep','frz':'Freeze','normal':'Selected opponent',
 'self':'User','allAdjacentFoes':'All adjacent opponents','allySide':'User’s side','foeSide':'Opponent’s side',
 'all':'Entire field','randomNormal':'Random opponent','allAdjacent':'All adjacent Pokémon','any':'Any eligible Pokémon'}


def words(key):
    if key in NAMES:return NAMES[key]
    if key in STATS:return STATS[key]
    if key.startswith('on') and len(key)>2 and key[2].isupper():
        key=key[2:]
        key=key.replace('SubOrder',' sequence within stage').replace('Priority',' resolution priority').replace('Order',' resolution order')
    return re.sub(r'([a-z])([A-Z])',r'\1 \2',key).replace('_',' ').strip().capitalize()


def value_text(value, key=''):
    if isinstance(value,bool):return 'Yes' if value else 'No'
    if isinstance(value,list):
        if key in {'heal','drain','recoil'} and len(value)==2:return f'{value[0]}/{value[1]}'
        return ', '.join(value_text(v) for v in value)
    if value is None:return 'None'
    return STATUS.get(str(value), STATS.get(str(value), str(value)))


def rule_table(data):
    rows=[]
    def walk(v, prefix=()):
        if isinstance(v,dict):
            for key,val in v.items():
                if key=='rating':continue  # builder heuristic, not a game rule
                if key=='flags':
                    for flag,enabled in val.items():
                        if flag in FLAGS:rows.append((FLAGS[flag], 'Yes' if enabled else 'No'))
                else:walk(val,prefix+(key,))
        elif isinstance(v,list) and any(isinstance(x,dict) for x in v):
            for i,item in enumerate(v,1):walk(item,prefix+(str(i),))
        else:
            labels=[words(k) if not k.isdigit() else 'Option '+k for k in prefix]
            rows.append((' — '.join(labels),value_text(v,prefix[-1] if prefix else '')))
    walk(data)
    if not rows:return ''
    return '| Rule detail | Value |\n|---|---|\n'+'\n'.join('| '+label.replace('|','\\|')+' | '+value.replace('|','\\|').replace('\n',' ')+' |' for label,value in rows)+'\n'


def clean_markdown(text, name):
    # Callback bodies and generic code-event plumbing are maintainer material.
    text=re.sub(r'## Event timing and exact exceptions\n.*?</details>', '',text,flags=re.S)
    text=re.sub(r'<details>.*?</details>', '',text,flags=re.S)
    def fence(match):
        language,body=match.group(1).strip(),match.group(2)
        if language=='json':return rule_table(json.loads(body))
        if language=='text':
            rows=[]
            for line in body.strip().splitlines():
                if ':' in line:
                    row,cells=line.split(':',1);rows.append(f'| {row.strip()} | {cells.strip()} |')
            return '| Row | Cell flags, from left to right |\n|---|---|\n'+'\n'.join(rows)+'\n'
        if not language and 'hit_tenths' in body:
            return '''| Step | Exact arithmetic rule |
|---|---|
| Scale attack | Let A be the attack stat multiplied by 100 and rounded to an integer. |
| Scale defense bonus | Let B be the floor of ten times the defense bonus plus 0.001. |
| Scale defense | Let D be the defense stat multiplied by ten, rounded, then multiplied by B. |
| Scale health | Multiply each current and maximum HP by ten and round; call these hₐ, mₐ, hᵈ and mᵈ. |
| Attacking force | Fₐ is the floor of hₐ × A × 100 ÷ mₐ; use at least 1 for the divisor. |
| Defending force | Fᵈ is the floor of D × hᵈ × 100 ÷ mᵈ; use at least 1 for the divisor. |
| Total force | T = Fₐ + Fᵈ. If T is zero or negative, all damage is zero. |
| Main hit in HP tenths | First floor A × Fₐ × 450 ÷ (T × 1,000), then round to the nearest multiple of ten, with ties upward. |
| Retaliation in HP tenths | Zero if the main hit kills the defender or B is zero or negative. Otherwise first floor Fᵈ × D × 4,500 ÷ (T × 1,000 × B), then round to the nearest multiple of ten, ties upward. |
| Ordinary HP damage | Divide the rounded main hit in tenths by ten. |
| Area HP damage | Floor the main hit in tenths divided by twenty; this is the area damage in HP. |
'''
        if language in {'sh','shell','bash','python','javascript','js','typescript','cpp','rust'}:return ''
        raise ValueError(f'Unclassified fenced block in {name}: {language!r}')
    text=FENCE.sub(fence,text)
    text=re.sub(r'<a id="([^"]+)"></a>\s*\n(## [^\n]+)', lambda m:m[2]+'\n\n### '+m[1], text)
    text=re.sub(r'## Sources\n.*?(?=\n\[Wiki home\]|\Z)','',text,flags=re.S)
    text=re.sub(r'^Source: pinned .*\n?', '',text,flags=re.M)
    text=re.sub(r'`boosts` gives stages.*?ordinary formula\.\n','',text,flags=re.S)
    text=re.sub(r'Duration is in engine turns where supplied\..*?may end it\.\n',
        'Durations are measured in battle turns. Switching, fainting or the stated effect-specific conditions may end an effect before its maximum duration.\n',text,flags=re.S)
    text=re.sub(r'(?m)^\| `on[A-Z][^`]*` \| (?:none|no|0|\[\]) \|\n?', '',text)
    text=text.replace('## Exact structured effects','## Additional rule details').replace('## Exact rule fields','## Additional rule details').replace('## Exact condition fields','## Duration and timing details')
    text=text.replace('`round_half_up_to_multiple_of_10` rounds integer\ntenths to whole HP, with ties upward.', 'Rounding to the nearest multiple of ten converts tenths to whole HP, with ties upward.')
    text=text.replace('the exact shared counter and cap are in the callbacks.', 'the shared counter and chance floor apply across repeated protection attempts.')
    text=text.replace('callback exceptions','move-specific exceptions').replace('its callback','its effect').replace('callbacks','effect rules').replace('callback','effect rule')
    # Technical source inventories and commands stay outside the readable wiki.
    text=re.sub(r'(?m)^.*(?:python3? |\.venv-[^ /]+/bin/python|build_wiki\.py|validate_wiki\.py|generate_wiki\.cjs).*\n?', '',text)
    text=re.sub(r'(?m)^.*(?:\*\*Implementation provenance:\*\*|Source owner:).*\n?', '',text)
    # Keep source location metadata off individual card pages.
    text=re.sub(r'(?m)^\*\*Implementation provenance:\*\*.*\n?', '',text)
    text=re.sub(r'(?m)^- `on[A-Z][^`]*`:.*\n?', '',text)
    text=re.sub(r'\n{3,}','\n\n',text)
    return '\n'.join(line.rstrip() for line in text.strip().splitlines())+'\n'


def relocate_tools(maintenance):
    tools=maintenance/'tools';raw=maintenance/'raw_build';snapshot=maintenance/'source_markdown'
    for p in tools.glob('*'):
        if p.suffix not in {'.py','.cjs','.js'}:continue
        text=p.read_text()
        if p.suffix=='.py':
            root="next(p for p in Path(__file__).resolve().parents if (p/'simulators/catalog.json').is_file())"
            text=text.replace("ROOT=Path(__file__).resolve().parents[5]",'ROOT='+root)
            text=text.replace('ROOT = HERE.parents[4]','ROOT = '+root).replace('ROOT=HERE.parents[4]','ROOT='+root)
            if p.name!='validate_wiki.py':
                text=text.replace('HERE = Path(__file__).resolve().parent','HERE = Path(__file__).resolve().parents[1] / \'raw_build\'')
                text=text.replace('HERE=Path(__file__).resolve().parent','HERE=Path(__file__).resolve().parents[1]/\'raw_build\'')
                text=re.sub(r"BASE=ROOT/'simulators/[^']+/autoresearch/task/game_rules'", "BASE=Path(__file__).resolve().parents[1]/'raw_build'",text)
                text=re.sub(r"base=root/'simulators/[^']+/autoresearch/task/game_rules'", "base=Path(__file__).resolve().parents[1]/'raw_build'",text)
                text=text.replace("BASE/'validate_wiki.py'","Path(__file__).with_name('validate_wiki.py')")
            else:
                text=text.replace('HERE = Path(__file__).resolve().parent', 'HERE = Path(__file__).resolve().parents[1] / \'raw_build\'')
            p.write_text(text)
        else:
            text=text.replace("const HERE=__dirname,ROOT=path.resolve(HERE,'../../../../..'),ENGINE=path.join(ROOT,'vendor/engines/pokemon-showdown');", "const HERE=path.resolve(__dirname,'../raw_build');\nlet ROOT=__dirname;while(!fs.existsSync(path.join(ROOT,'simulators/catalog.json'))){const parent=path.dirname(ROOT);if(parent===ROOT)throw new Error('Repository root not found');ROOT=parent;}\nconst ENGINE=path.join(ROOT,'vendor/engines/pokemon-showdown');")
            p.write_text(text)
    raw.mkdir(exist_ok=True)
    shutil.copytree(snapshot,raw,dirs_exist_ok=True)


def clean_game(game):
    ar=ROOT/f'simulators/{game}/autoresearch';base=ar/'task/game_rules';maintenance=ar/'wiki_maintenance'
    snapshot=maintenance/'source_markdown';metadata=maintenance/'metadata';tools=maintenance/'tools'
    for folder in (snapshot,metadata,tools):folder.mkdir(parents=True,exist_ok=True)
    for path in list(base.rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(base)
        if path.suffix=='.md':
            dest=snapshot/relative
            if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        else:
            destination=(metadata if path.suffix=='.json' else tools)/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():raise FileExistsError(destination)
            shutil.move(path,destination)
    for name in ('reference.md','provenance.md'):
        path=base/name
        if path.exists():shutil.move(path,maintenance/name)
    coverage=json.loads((metadata/'coverage.json').read_text())
    scope=['# Scope and limitations','', '[Wiki home](README.md)','',str(coverage['mode']),'', '## Documented content','', '| Catalog | Entries |','|---|---:|']
    for item in coverage.get('catalogs',[]):scope.append(f"| {item['name'].replace('_',' ')} | {item.get('documented',item.get('entries',item.get('expected','—')))} |")
    scope+=['','## Limits','']
    for item in coverage.get('gaps',coverage.get('known_gaps',coverage.get('known_limits',coverage.get('limitations',[])))):
        scope.append('- '+str(item))
    scope+=['','The written rules apply to the pinned mode identified above. Catalog membership does not by itself establish that every combination of effects has been independently tested.']
    (base/'scope-and-limitations.md').write_text('\n'.join(scope)+'\n')
    for original in snapshot.rglob('*.md'):
        relative=original.relative_to(snapshot)
        if relative.as_posix() in {'reference.md','provenance.md'}:continue
        dest=base/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        text=clean_markdown(original.read_text(),str(relative))
        def link(match):
            label,target=match.group(1),match.group(2);plain=unquote(target.split('#',1)[0]);resolved=(dest.parent/plain).resolve()
            if not plain:return match.group(0)
            if '://' in plain:return label
            if resolved.is_relative_to(base.resolve()):
                rel=resolved.relative_to(base.resolve()).as_posix()
                if rel=='catalog-index.json':return f'[{label}]('+Path(__import__('os').path.relpath(base/'catalog-index.md',dest.parent)).as_posix()+')'
                if Path(rel).suffix!='.md' or rel in {'reference.md','provenance.md'}:
                    return f'[{label}]('+Path(__import__('os').path.relpath(base/'scope-and-limitations.md',dest.parent)).as_posix()+')'
                return match.group(0)
            return label if Path(plain).suffix=='.md' else ''
        text=re.sub(r'(?<!!)\[([^\]]*)\]\(([^)]+)\)',link,text)
        text=text.replace('coverage.json','scope and limitations').replace('reference.json','maintainer source inventory').replace('catalog-index.json','catalog index')
        text=text.replace('Exact event callbacks','Effect conditions').replace('exact event guards','effect conditions').replace('event guards','effect conditions')
        text=text.replace('and, when present, exact pinned effect rules.\nCallbacks preserve rare guards/rounding rather than being replaced by guesses.', 'and readable tables for additional rule details.')
        text=re.sub(r'(?m)^.*SHA-256.*\n?', '',text)
        text=re.sub(r'\n{3,}','\n\n',text)
        dest.write_text(text.strip()+'\n')
    if game=='pokemon':
        index=json.loads((metadata/'catalog-index.json').read_text())
        lines=['# Catalog lookup','', '[Wiki home](README.md)','', 'Lookup the readable entry for a species, move, ability, item or condition by its stable game ID.','']
        for cat in index['catalogs']:
            lines+=['## '+cat['name'].capitalize(),'','| Game ID | Entry |','|---|---|']
            for ident in cat['ids']:
                target=base/cat['name']/f'{ident}.md'
                if target.is_file():
                    title=target.read_text().splitlines()[0].removeprefix('# ')
                    lines.append(f"| {ident} | [{title}]({cat['name']}/{ident}.md) |")
            lines.append('')
        (base/'catalog-index.md').write_text('\n'.join(lines)+'\n')
    # Agent entry point makes the delivery boundary explicit in readable prose.
    readme=base/'README.md';text=readme.read_text()
    text=re.sub(r'## Provenance and completeness\n.*?(?=\n## |\Z)', '## Scope\n\nSee [scope and limitations](scope-and-limitations.md) for the exact mode, documented catalogs and known limits.\n',text,flags=re.S)
    text=re.sub(r'## Coverage and sources\n.*?(?=\n## |\Z)', '## Scope\n\nSee [scope and limitations](scope-and-limitations.md) for catalog counts and known limits. This folder contains the readable game reference and lookup indexes.\n',text,flags=re.S)
    text=re.sub(r'## What “coverage” means here\n.*?(?=\n## |\Z)', '## Scope\n\nSee [scope and limitations](scope-and-limitations.md) for catalog counts and known limits. Ordinary offers, conditional effects and related definitions remain clearly distinguished.\n',text,flags=re.S)
    text=re.sub(r'This is a \*\*catalog-complete documentation snapshot\*\*.*?The old \[mechanics overview\]', 'The readable [scope and limitations](scope-and-limitations.md) describes catalog boundaries and uncertainty. Additional values appear in Markdown tables, with no implementation appendices.\n\nThe old [mechanics overview]',text,flags=re.S)
    # Drop maintainer-only paragraphs left in the compact Generals README.
    text=re.sub(r'\nRebuild and run.*?(?=\n\n|\Z)','',text,flags=re.S)
    if 'scope-and-limitations.md' not in text:text+='\n[Scope and limitations](scope-and-limitations.md).\n'
    readme.write_text(text.strip()+'\n')
    for directory in sorted((p for p in base.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
        if not any(directory.iterdir()):directory.rmdir()
    relocate_tools(maintenance)
    (maintenance/'README.md').write_text(f'''# {game} wiki maintenance\n\nThis directory is maintainer-only. It is excluded from the agent rulebook, loader, exporter and HTML.\n\n- tools/: relocated generators and validators. Generators write raw_build/, never task/game_rules/.\n- metadata/: source inventories, catalog coverage and hashes moved out of the agent folder.\n- source_markdown/: preserved original written content and implementation appendices.\n- raw_build/: a maintainer staging area seeded with the original written chapters.\n\nThe agent receives only task/game_rules/*.md and its Markdown subfolders. Run the shared Markdown-only validator/exporter after deliberate documentation changes.\n''')
    pages=list(base.rglob('*.md'))
    return {'game':game,'agent_markdown_files':len(pages),'agent_bytes':sum(p.stat().st_size for p in pages),'maintenance':str(maintenance.relative_to(ROOT))}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--game',choices=GAMES,action='append');args=p.parse_args()
    print(json.dumps([clean_game(game) for game in args.game or GAMES],indent=2))
