"""Public species trapping possibilities from frozen Showdown 567c1cedddeaa2f35576005d3bfd526cf4f8a13e.

Offline Dex metadata only; no source engine is imported by native environments.
"""
TRAPPING_SPECIES={'diglett': 2, 'dugtrio': 2, 'geodudealola': 1, 'graveleralola': 1, 'golemalola': 1, 'magnemite': 1, 'magneton': 1, 'gengarmega': 4, 'wobbuffet': 4, 'nosepass': 1, 'trapinch': 2, 'wynaut': 4, 'magnezone': 1, 'probopass': 1, 'gothita': 4, 'gothorita': 4, 'gothitelle': 4, 'meltan': 1}

def trapping_flags(species):return TRAPPING_SPECIES.get(species,0)<<16

def intrinsic_flags(species):
    flags=1 if species.startswith('pikachu') else 2 if species.startswith(('latias','latios')) else 4 if species.startswith('arceus') else 0
    for name,bit in (('dialga',8),('palkia',16),('giratina',32),('zacian',64),('zamazenta',128)):
        if species.startswith(name):flags|=bit
    flags|={'taurospaldeacombat':256,'taurospaldeablaze':512,'taurospaldeaaqua':768,'hoopaunbound':2048,'pecharunt':4096,'greninjabond':8192,'greninjaash':16384}.get(species,0)
    if species.startswith('ogerpon'):flags|=32768
    return flags|trapping_flags(species)

BATTLE_FORMES=[
 {'name':'Zacian-Crowned','id':'zaciancrowned','types':['Fairy','Steel'],'stats':{'hp':92,'atk':150,'def':115,'spa':80,'spd':115,'spe':148},'fixed_hp':0,'nfe':False,'weight_hg':3550,'paradox_species':False},
 {'name':'Zamazenta-Crowned','id':'zamazentacrowned','types':['Fighting','Steel'],'stats':{'hp':92,'atk':120,'def':140,'spa':80,'spd':140,'spe':128},'fixed_hp':0,'nfe':False,'weight_hg':7850,'paradox_species':False},
]

if __name__=='__main__':
    # Verification only. Native environment construction never executes Node.
    import argparse
    import hashlib
    import json
    from pathlib import Path
    import subprocess
    from .bridge import HERE
    from .validate import oracle_identity
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    upstream=HERE.parents[2]/'candidates/pokemon-showdown'
    if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!='567c1cedddeaa2f35576005d3bfd526cf4f8a13e':
        raise ValueError('Public metadata source pin changed')
    script="""
const {Dex}=require('./dist/sim');const dex=Dex.mod('gen9'),traps={};
for(const p of dex.species.all()){
 let f=0;for(const [slot,name]of Object.entries(p.abilities)){
  if(slot==='H'&&p.unreleasedHidden)continue;
  f|={'Magnet Pull':1,'Arena Trap':2,'Shadow Tag':4}[name]||0;
 }if(f)traps[p.id]=f;
}
const forms=['Zacian-Crowned','Zamazenta-Crowned'].map(name=>{
 const p=dex.species.get(name);return {name,id:p.id,types:p.types,stats:p.baseStats,fixed_hp:p.maxHP||0,nfe:!!p.nfe,weight_hg:p.weighthg,paradox_species:p.tags.includes('Paradox')};
});
console.log(JSON.stringify({traps,forms,intrinsic:dex.species.all().map(p=>[p.id,(p.baseSpecies==='Pikachu'?1:0)|([380,381].includes(p.num)?2:0)|(p.num===493?4:0)|({483:8,484:16,487:32,888:64,889:128}[p.num]||0)|(p.baseSpecies==='Ogerpon'?32768:0)|({taurospaldeacombat:256,taurospaldeablaze:512,taurospaldeaaqua:768,hoopaunbound:2048,pecharunt:4096,greninjabond:8192,greninjaash:16384}[p.id]||0)])}));
"""
    reference=json.loads(subprocess.check_output(['node','-e',script],cwd=upstream,text=True))
    if reference['traps']!=TRAPPING_SPECIES or reference['forms']!=BATTLE_FORMES:raise AssertionError('Public metadata differs from frozen Dex')
    for name,flags in reference['intrinsic']:
        if intrinsic_flags(name)&65535!=flags:raise AssertionError(('Intrinsic species flags',name))
    result=dict(full_gen9_admission=False,trapping_species=len(TRAPPING_SPECIES),
                exact_intrinsic_species=len(reference['intrinsic']),exact_battle_formes=len(BATTLE_FORMES),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),oracle_identity=oracle_identity())
    with Path(args.output).open('x')as target:target.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='oracle_identity'},indent=2))
