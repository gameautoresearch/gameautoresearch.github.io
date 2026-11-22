"""Conservative rule lowering for constructed native development fixtures.

This is deliberately not a full-task data compiler. Unknown IDs, callbacks and
unimplemented effect payloads are rejected instead of silently becoming Tackle.
"""
from .bridge import Move

TYPES = ('Normal', 'Fire', 'Water', 'Grass', 'Electric', 'Ice', 'Fighting',
         'Poison', 'Ground', 'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost',
         'Dragon', 'Steel', 'Dark', 'Fairy', 'Stellar')
STATS = ('atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion')
STATUSES = ('', 'brn', 'par', 'slp', 'frz', 'psn', 'tox')
ABILITY_IDS = {name:index for index,name in enumerate(('noability','naturalcure','regenerator','pressure','hugepower','purepower','guts','levitate','intimidate','quickfeet','marvelscale','skilllink','owntempo','innerfocus','oblivious','insomnia','vitalspirit','immunity','limber','waterveil','magmaarmor','shielddust','serenegrace','soundproof','bulletproof','infiltrator','drizzle','drought','sandstream','snowwarning','swiftswim','chlorophyll','sandrush','slushrush','sandveil','snowcloak','raindish','icebody','solarpower','hydration','airlock','cloudnine','technician','thickfat','adaptability','tintedlens','overgrow','blaze','torrent','swarm','ironfist','strongjaw','toughclaws','furcoat','icescales','filter','solidrock','neuroforce','sniper','multiscale','shadowshield','protosynthesis','quarkdrive','clearbody','whitesmoke','fullmetalbody','defiant','competitive','contrary','simple','sheerforce','sturdy','battlearmor','shellarmor','waterabsorb','voltabsorb','sapsipper','stormdrain','lightningrod','motordrive','eartheater','wellbakedbody','unaware','prankster','overcoat','stickyhold','hustle','compoundeyes','sharpness','megalauncher','reckless','dragonsmaw','transistor','rockypayload','steelworker','flareboost','toxicboost','protean','libero','suctioncups','heavymetal','lightmetal','triage','galewings','flamebody','static','poisonpoint','justified','stamina','weakarmor','cursedbody','speedboost','stakeout','rockhead','heatproof','fluffy','waterbubble','punkrock','purifyingsalt','earlybird','leafguard','shedskin','frisk','flashfire','synchronize','thermalexchange','watercompaction','rattled','toxicchain','poisontouch','aftermath','roughskin','ironbarbs','windrider','psychicsurge','grassysurge','electricsurge','mistysurge','download','dauntlessshield','intrepidsword','noguard','powerspot','plus','minus','moldbreaker','teravolt','turboblaze','scrappy','mindseye','keeneye','bigpecks','prismarmor','surgesurfer','liquidvoice','pixilate','galvanize','corrosion','unseenfist','goodasgold','queenlymajesty','analytic','gooey','tanglinghair','toxicdebris','seedsower','aromaveil','flowerveil','poisonheal','dryskin','superluck','magicguard','moxie','chillingneigh','grimneigh','soulheart','unburden','cheekpouch','unnerve','harvest','comatose','slowstart','baddreams','supremeoverlord','liquidooze','beadsofruin','tabletsofruin','vesselofruin','swordofruin','multitype','arenatrap','magnetpull','shadowtag','sandforce','damp','mirrorarmor','poisonpuppeteer','hadronengine','orichalcumpulse','asoneglastrier','asonespectrier','truant','myceliummight','effectspore','electromorphosis','berserk','angershell','battlebond','magicbounce','magician','pickpocket','trace','cutecharm','cudchew','dancer','disguise','iceface','hungerswitch','zerotohero','gulpmissile','shieldsdown','imposter','terashift','terashell','teraformzero','embodyaspectteal','embodyaspectwellspring','embodyaspecthearthflame','embodyaspectcornerstone','illusion'))}
ITEM_IDS = {name:index for index,name in enumerate(('', 'leftovers','choiceband','choicespecs','choicescarf','focussash','sitrusberry','lumberry','loadeddice','lifeorb','assaultvest','damprock','heatrock','smoothrock','icyrock','lightclay','heavydutyboots','terrainextender','chestoberry','eviolite','widelens','boosterenergy','safetygoggles','floatstone','whiteherb','flameorb','toxicorb','weaknesspolicy','rockyhelmet','lightball','souldew','silkscarf','charcoal','mysticwater','miracleseed','magnet','nevermeltice','blackbelt','poisonbarb','softsand','sharpbeak','twistedspoon','silverpowder','hardstone','spelltag','dragonfang','metalcoat','blackglasses','fairyfeather','airballoon','scopelens','powerherb','leppaberry','throatspray','flameplate','splashplate','meadowplate','zapplate','icicleplate','fistplate','toxicplate','earthplate','skyplate','mindplate','insectplate','stoneplate','spookyplate','dracoplate','ironplate','dreadplate','pixieplate','griseouscore','lustrousglobe','adamantcrystal','lustrousorb','adamantorb','griseousorb','rustedsword','rustedshield','wellspringmask','hearthflamemask','cornerstonemask','expertbelt','custapberry'))}
# These IDs exercise the implemented scalar effect families in an inert arena.
# Inclusion here is not qualification under arbitrary Gen 9 items/abilities.
DEVELOPMENT_MOVES = frozenset('''tackle pound scratch watergun quickattack
wingattack vinewhip swift ember thundershock icebeam acid bubble poweruppunch
chillingwater recover milkdrink softboiled swordsdance growl tailwhip agility
doubleteam drainpunch absorb megadrain gigadrain drainingkiss doubleedge
takedown bravebird willowisp sing uturn voltswitch flipturn earthquake mudshot protect'''.split())

SCALAR_FIELDS = frozenset('''name id fullname effectType exists num gen isNonstandard noCopy
affectsFainted sourceEffect type target basePower accuracy critRatio baseMoveType
hasSheerForceBoost priority category ignoreAccuracy ignoreEvasion ignoreNegativeOffensive ignorePositiveDefensive
ignoreOffensive ignoreDefensive ignoreImmunity pp noPPBoosts isZ isMax flags ignoreAbility
spreadHit forceSTAB maxMove zMove contestType desc shortDesc heal drain recoil boosts status
secondary secondaries self selfSwitch forceSwitch damage willCrit multihit multiaccuracy selfBoost smartTarget thawsTarget volatileStatus weather sideCondition slotCondition terrain sleepUsable callsMove overrideOffensiveStat overrideDefensiveStat overrideOffensivePokemon overrideDefensivePokemon native_callbacks native_callback_paths beforeTurnCallback priorityChargeCallback breaksProtect selfdestruct hasCrashDamage pseudoWeather'''.split())
UNIMPLEMENTED_TRUE_FIELDS = ('noCopy','affectsFainted','hasSheerForceBoost',
    'isZ','isMax','spreadHit','forceSTAB')
UNIMPLEMENTED_FLAGS = frozenset()
PORTED_CALLBACKS = {'hurricane':{'onModifyMove'},'thunder':{'onModifyMove'},'blizzard':{'onModifyMove'},
    'weatherball':{'onModifyType','onModifyMove'},'hydrosteam':{'onBasePower'},'acrobatics':{'basePowerCallback'},'facade':{'onBasePower'},
    'terablast':{'basePowerCallback','onPrepareHit','onModifyType','onModifyMove'},
    'taunt':{'condition.onStart','condition.onEnd','condition.onDisableMove','condition.onBeforeMove'},
    'encore':{'condition.onStart','condition.onOverrideAction','condition.onResidual','condition.onEnd','condition.onDisableMove'},
    'leechseed':{'onTryImmunity','condition.onStart','condition.onResidual'},
    'yawn':{'onTryHit','condition.onStart','condition.onEnd'},
    'substitute':{'onTryHit','onHit','condition.onStart','condition.onTryPrimaryHit','condition.onEnd'},
    'disable':{'onTryHit','condition.onStart','condition.onEnd','condition.onBeforeMove','condition.onDisableMove'}}

PORTED_CALLBACKS.update({'reflect': {'condition.onSideEnd', 'condition.onAnyModifyDamage', 'condition.durationCallback', 'condition.onSideStart'}, 'lightscreen': {'condition.onSideEnd', 'condition.onAnyModifyDamage', 'condition.durationCallback', 'condition.onSideStart'}, 'auroraveil': {'condition.onAnyModifyDamage', 'onTry', 'condition.onSideEnd', 'condition.onSideStart', 'condition.durationCallback'}, 'tailwind': {'condition.onSideEnd', 'condition.onSideStart', 'condition.onModifySpe', 'condition.durationCallback'}, 'safeguard': {'condition.onTryAddVolatile', 'condition.onSideEnd', 'condition.onSideStart', 'condition.onSetStatus', 'condition.durationCallback'}, 'mist': {'condition.onTryBoost', 'condition.onSideEnd', 'condition.onSideStart'}})

PORTED_CALLBACKS.update({'stealthrock': {'condition.onSideStart', 'condition.onSwitchIn'}, 'spikes': {'condition.onSideStart', 'condition.onSideRestart', 'condition.onSwitchIn'}, 'toxicspikes': {'condition.onSideStart', 'condition.onSideRestart', 'condition.onSwitchIn'}, 'stickyweb': {'condition.onSideStart', 'condition.onSwitchIn'}, 'rapidspin': {'onAfterHit', 'onAfterSubDamage'}, 'mortalspin': {'onAfterHit', 'onAfterSubDamage'}, 'defog': {'onHit'}, 'tidyup': {'onHit'}})

PORTED_CALLBACKS.update({'electricterrain': {'condition.onSetStatus', 'condition.onFieldEnd', 'condition.onBasePower', 'condition.durationCallback', 'condition.onFieldStart', 'condition.onTryAddVolatile'}, 'grassyterrain': {'condition.onResidual', 'condition.onFieldEnd', 'condition.onBasePower', 'condition.durationCallback', 'condition.onFieldStart'}, 'mistyterrain': {'condition.onSetStatus', 'condition.onFieldEnd', 'condition.onBasePower', 'condition.durationCallback', 'condition.onFieldStart', 'condition.onTryAddVolatile'}, 'psychicterrain': {'condition.onFieldEnd', 'condition.onBasePower', 'condition.durationCallback', 'condition.onFieldStart', 'condition.onTryHit'}})

PORTED_CALLBACKS.update({'storedpower': {'basePowerCallback'}, 'powertrip': {'basePowerCallback'}, 'eruption': {'basePowerCallback'}, 'waterspout': {'basePowerCallback'}, 'hex': {'basePowerCallback'}, 'venoshock': {'onBasePower'}, 'brine': {'onBasePower'}, 'knockoff': {'onAfterHit', 'onBasePower'}, 'poltergeist': {'onTry', 'onTryHit'}, 'ragefist': {'basePowerCallback'}, 'lastrespects': {'basePowerCallback'}})

PORTED_CALLBACKS.update({'roost':{'condition.onStart','condition.onType'},'rest':{'onTry','onHit'},'synthesis':{'onHit'},'morningsun':{'onHit'},'moonlight':{'onHit'},'suckerpunch':{'onTry'},'sleeptalk':{'onTry','onHit'},'wish':{'condition.onStart','condition.onRestart','condition.onResidual','condition.onEnd'}})

PORTED_CALLBACKS.update({'heavyslam': {'basePowerCallback', 'onTryHit'}, 'heatcrash': {'basePowerCallback', 'onTryHit'}, 'grassknot': {'basePowerCallback', 'onTryHit'}, 'lowkick': {'basePowerCallback', 'onTryHit'}, 'freezedry': {'onEffectiveness'}, 'psychicfangs': {'onTryHit'}, 'brickbreak': {'onTryHit'}, 'icespinner': {'onAfterHit', 'onAfterSubDamage'}, 'haze': {'onHitField'}, 'painsplit': {'onHit'}, 'superfang': {'damageCallback'}, 'ruination': {'damageCallback'}, 'endeavor': {'onTryImmunity', 'damageCallback'}, 'shoreup': {'onHit'}, 'alluringvoice': {'secondaries.0.onHit', 'secondary.onHit'}, 'strengthsap': {'onHit'}})

PORTED_CALLBACKS['throatchop']={'secondary.onHit','secondaries.0.onHit','condition.onStart','condition.onDisableMove','condition.onBeforeMove','condition.onModifyMove','condition.onEnd'}

PORTED_CALLBACKS.update({'tripleaxel':{'basePowerCallback'},'fakeout':{'onTry'},'firstimpression':{'onTry'}})

PORTED_CALLBACKS.update({'trick':{'onTryImmunity','onHit'},'switcheroo':{'onTryImmunity','onHit'},'triattack':{'secondary.onHit','secondaries.0.onHit'},'direclaw':{'secondary.onHit','secondaries.0.onHit'}})

PORTED_CALLBACKS['partingshot']={'onHit'}
PORTED_CALLBACKS.update({'dragonenergy':{'basePowerCallback'},'payback':{'basePowerCallback'},'avalanche':{'basePowerCallback'},'revenge':{'basePowerCallback'},'lashout':{'onBasePower'},'psyblade':{'onBasePower'},'expandingforce':{'onBasePower','onModifyMove'},'collisioncourse':{'onBasePower'},'electrodrift':{'onBasePower'},'grassyglide':{'onModifyPriority'}})

PORTED_CALLBACKS.update({'stoneaxe': {'onAfterHit', 'onAfterSubDamage'}, 'ceaselessedge': {'onAfterHit', 'onAfterSubDamage'}, 'thunderclap': {'onTry'}, 'takeheart': {'onHit'}, 'revelationdance': {'onModifyType'}, 'clearsmog': {'onHit'}, 'pollenpuff': {'onTryHit', 'onTryMove', 'onHit'}, 'healbell': {'onHit'}, 'bellydrum': {'onHit'}, 'filletaway': {'onTryHit', 'onTry', 'onHit'}, 'clangoroussoul': {'onTryHit', 'onTry', 'onHit'}, 'teleport': {'onTry'}, 'ficklebeam': {'onBasePower'}, 'gravapple': {'onBasePower'}, 'burningjealousy': {'secondary.onHit', 'secondaries.0.onHit'}})

PORTED_CALLBACKS.update({'highjumpkick':{'onMoveFail'},'supercellslam':{'onMoveFail'},'stompingtantrum':{'basePowerCallback'},'temperflare':{'basePowerCallback'}})

PORTED_CALLBACKS.update({'trickroom':{'condition.durationCallback','condition.onFieldStart','condition.onFieldRestart','condition.onFieldEnd'},'revivalblessing':{'onTryHit'}})

PORTED_CALLBACKS.update({'curse':{'onModifyMove','onTryHit','onHit','condition.onStart','condition.onResidual'},'saltcure':{'condition.onStart','condition.onResidual','condition.onEnd'}})

PORTED_CALLBACKS['doubleshock']={'onTryMove','self.onHit'}
PORTED_CALLBACKS.update({'sparklingaria':{'onAfterMove'},'bleakwindstorm':{'onModifyMove'},'shellsidearm':{'onPrepareHit','onModifyMove','onHit','onAfterSubDamage'}})

PORTED_CALLBACKS.update({'solarbeam':{'onTryMove','onBasePower'},'meteorbeam':{'onTryMove'}})

PORTED_CALLBACKS.update({'noretreat':{'onTry','condition.onStart'},'magnetrise':{'onTry','condition.onStart','condition.onImmunity','condition.onEnd'},'glaiverush':{'condition.onStart','condition.onAccuracy','condition.onSourceModifyDamage','condition.onBeforeMove'}})

PORTED_CALLBACKS['judgment']={'onModifyType'}
PORTED_CALLBACKS.update({'watershuriken':{'basePowerCallback'},'chillyreception':{'priorityChargeCallback','condition.onBeforeMove'},'focuspunch':{'priorityChargeCallback','beforeMoveCallback','condition.onStart','condition.onHit','condition.onTryAddVolatile'},'beakblast':{'priorityChargeCallback','condition.onStart','condition.onHit','onAfterMove'}})
PORTED_CALLBACKS.update({'bugbite':{'onHit'},'destinybond':{'onPrepareHit','condition.onStart','condition.onFaint','condition.onBeforeMove','condition.onMoveAborted'},'shedtail':{'onTryHit','onHit','self.onHit'},'healingwish':{'onTryHit','condition.onSwitchIn','condition.onSwap'},'mirrorcoat':{'damageCallback','beforeTurnCallback','onTry','condition.onStart','condition.onRedirectTarget','condition.onDamagingHit'},'beatup':{'basePowerCallback','onModifyMove'}})
PORTED_CALLBACKS.update({'aurawheel':{'onTry','onModifyType'},'relicsong':{'onAfterMoveSecondarySelf'}})
PORTED_CALLBACKS['futuresight']={'onTry'}
PORTED_CALLBACKS['transform']={'onHit'}
PORTED_CALLBACKS.update({'ivycudgel':{'onPrepareHit','onModifyType'},'terastarstorm':{'onModifyType','onModifyMove'}})
PORTED_CALLBACKS['spiritshackle']={'secondary.onHit','secondaries.0.onHit'}
PORTED_CALLBACKS['photongeyser']={'onModifyMove'}
PORTED_CALLBACKS.update({'ragingbull':{'onTryHit','onModifyType'},'fusionflare':{'onBasePower'},'fusionbolt':{'onBasePower'},'hyperspacefury':{'onTry'},'courtchange':{'onHitField'}})

def compile_moves(records):
    moves = (Move * len(records))()
    for index, data in enumerate(records):
        callbacks = set(data.get('native_callback_paths',data.get('native_callbacks',[])))
        protect = data['id']=='protect' and set(data.get('native_callbacks',[]))=={'onPrepareHit','onHit'}
        ported = data['id'] in PORTED_CALLBACKS and callbacks==PORTED_CALLBACKS[data['id']]
        if ported:callbacks=set()
        if protect and callbacks <= {'onPrepareHit','onHit','condition.onStart','condition.onTryHit'}:
            callbacks=set()
        if callbacks or (data['id'] not in DEVELOPMENT_MOVES and 'native_callback_paths' not in data):
            raise ValueError(f"Unimplemented native move: {data['id']}")
        extra=set(data)-SCALAR_FIELDS-({'condition'} if data['id'] in ('destinybond','healingwish','mirrorcoat','chillyreception','focuspunch','beakblast','noretreat','magnetrise','glaiverush','curse','saltcure','trickroom','revivalblessing','throatchop','roost','wish','taunt','encore','disable','leechseed','yawn','substitute','reflect','lightscreen','auroraveil','tailwind','safeguard','mist','stealthrock','spikes','toxicspikes','stickyweb','electricterrain','grassyterrain','mistyterrain','psychicterrain') else set())-({'condition','volatileStatus','stallingMove','onPrepareHit','onHit'} if protect else set())-(PORTED_CALLBACKS[data['id']] if ported else set())
        if extra:raise ValueError(f"Unimplemented native move fields: {data['id']} {sorted(extra)}")
        if data.get('ignoreAbility') and (data['id'] not in ('moongeistbeam','sunsteelstrike','photongeyser') or data['ignoreAbility'] is not True):raise ValueError('Unimplemented native ability-ignore move')
        if data.get('breaksProtect') and (data['id']!='hyperspacefury' or data['breaksProtect'] is not True):raise ValueError('Unsupported native protect-breaking move')
        if data.get('pseudoWeather') and (data['id']!='trickroom' or data['pseudoWeather']!='trickroom'):raise ValueError('Unimplemented native pseudo-weather')
        if data.get('selfdestruct') and not ((data['id']=='explosion' and data['selfdestruct']=='always') or (data['id']=='healingwish' and data['selfdestruct']=='ifHit')):raise ValueError('Unimplemented native self-destruct payload')
        if data.get('hasCrashDamage') and data['id'] not in ('highjumpkick','supercellslam'):raise ValueError('Unimplemented native crash damage')
        if any(data.get(field) for field in UNIMPLEMENTED_TRUE_FIELDS):raise ValueError('Unimplemented native modifier flag')
        if data.get('flags',{}).get('recharge') and data['id']!='gigaimpact':raise ValueError('Unsupported native recharge move')
        if data.get('flags',{}).get('charge') and data['id'] not in ('solarbeam','meteorbeam'):raise ValueError('Unsupported native charge move')
        if any(data.get('flags',{}).get(flag) for flag in UNIMPLEMENTED_FLAGS):raise ValueError('Unimplemented native move flag')
        if not isinstance(data.get('ignoreImmunity',False),bool):raise ValueError('Conditional native immunity is unimplemented')
        if data.get('target')=='all' and not data.get('weather') and not data.get('terrain') and data['id'] not in ('haze','trickroom','courtchange'):raise ValueError('Unsupported native field move')
        if data.get('target') not in ('normal','self','allAdjacentFoes','allAdjacent','randomNormal','any','adjacentFoe','scripted','all','allySide','foeSide','allyTeam'):
            raise ValueError('Unsupported native singles target')
        if not 1<=data.get('critRatio',1)<=4:raise ValueError('Unsupported native critical ratio')
        move = moves[index]
        move.id = index
        move.attack_stat=STATS.index(data['overrideOffensiveStat']) if data.get('overrideOffensiveStat') else -1
        move.defense_stat=STATS.index(data['overrideDefensiveStat']) if data.get('overrideDefensiveStat') else -1
        if move.attack_stat>4 or move.defense_stat>4:raise ValueError('Invalid damage stat')
        if data.get('overrideOffensivePokemon') not in (None,'target') or data.get('overrideDefensivePokemon') not in (None,'source'):raise ValueError('Unimplemented stat owner')
        move.attack_target=data.get('overrideOffensivePokemon')=='target';move.defense_source=data.get('overrideDefensivePokemon')=='source'
        move.ignore_boosts=sum(flag*bool(data.get(key)) for flag,key in ((1,'ignoreOffensive'),(2,'ignoreDefensive'),(4,'ignoreNegativeOffensive'),(8,'ignorePositiveDefensive')))
        move.kind={'toxic':1,'acrobatics':2,'facade':3,'terablast':4,'hurricane':5,'thunder':5,'blizzard':6,'weatherball':7,'hydrosteam':8,'rapidspin':9,'mortalspin':9,'defog':10,'tidyup':11,'storedpower':12,'powertrip':12,'eruption':13,'waterspout':13,'dragonenergy':13,'hex':14,'venoshock':15,'brine':16,'knockoff':17,'poltergeist':18,'ragefist':19,'lastrespects':20,'roost':21,'rest':22,'synthesis':23,'morningsun':23,'moonlight':23,'suckerpunch':24,'thunderclap':24,'wish':25,'sleeptalk':26,'heavyslam':27,'heatcrash':27,'grassknot':28,'lowkick':28,'freezedry':29,'psychicfangs':30,'brickbreak':30,'icespinner':31,'haze':32,'painsplit':33,'superfang':34,'ruination':34,'endeavor':35,'shoreup':36,'alluringvoice':37,'strengthsap':38,'throatchop':39,'outrage':40,'petaldance':40,'thrash':40,'tripleaxel':41,'fakeout':42,'firstimpression':42,'trick':43,'switcheroo':43,'triattack':44,'direclaw':45,'partingshot':46,'sparklingaria':73,'bleakwindstorm':74,'shellsidearm':75,'doubleshock':76,'solarbeam':77,'meteorbeam':78,'glaiverush':79,'noretreat':80,'magnetrise':81,'judgment':82,'photongeyser':83,'gigaimpact':89,'spiritshackle':90,'watershuriken':91,'chillyreception':92,'focuspunch':93,'beakblast':94,'bugbite':95,'destinybond':96,'shedtail':97,'healingwish':98,'mirrorcoat':99,'beatup':100,'aurawheel':101,'relicsong':102,'surf':103,'transform':104,'ivycudgel':105,'terastarstorm':106,'futuresight':107,'ragingbull':84,'fusionflare':85,'fusionbolt':86,'hyperspacefury':87,'courtchange':88,'curse':71,'saltcure':72,'trickroom':69,'revivalblessing':70,'stompingtantrum':66,'temperflare':66,'highjumpkick':67,'supercellslam':67,'explosion':68,'stoneaxe':54,'ceaselessedge':55,'takeheart':56,'revelationdance':57,'clearsmog':58,'healbell':59,'bellydrum':60,'filletaway':61,'clangoroussoul':62,'teleport':63,'ficklebeam':64,'burningjealousy':65,'payback':47,'avalanche':48,'revenge':48,'lashout':49,'psyblade':50,'expandingforce':51,'collisioncourse':52,'electrodrift':52,'grassyglide':53}.get(data['id'],0)
        move.traits=int(bool(data.get('flags',{}).get('defrost')))|2*int(bool(data.get('thawsTarget')))|4*int(bool(data.get('flags',{}).get('failencore')))|8*int(bool(data.get('flags',{}).get('bypasssub')))|16*int(bool(data.get('flags',{}).get('sound')))|32*int(bool(data.get('flags',{}).get('bullet')))|64*int(data['id'] in ('earthquake','bulldoze','magnitude'))|128*int('basePowerCallback' in data.get('native_callbacks',[]) and data['basePower'] in (0,150))|256*int(bool(data.get('ignoreAccuracy')))|512*int(bool(data.get('ignoreEvasion')))
        move.traits|=2048*int(bool(data.get('flags',{}).get('contact')))|4096*int(bool(data.get('flags',{}).get('punch')))|8192*int(bool(data.get('flags',{}).get('bite')))
        move.traits|=1024*int(bool(data.get('flags',{}).get('nosleeptalk') or data.get('flags',{}).get('charge') or data.get('isZ') or data.get('isMax')))
        move.traits|=1073741824*int(bool(data.get('flags',{}).get('dance')))
        move.traits|=536870912*int(bool(data.get('flags',{}).get('reflectable')))
        move.traits|=268435456*int(bool(data.get('ignoreAbility')))
        move.traits|=134217728*int(bool(data.get('flags',{}).get('cantusetwice')))|4194304*int(bool(data.get('flags',{}).get('wind')))|33554432*int(data['target']=='foeSide')|67108864*int(data['target']=='all')
        move.traits|=1048576*int(bool(data.get('flags',{}).get('slicing')))|2097152*int(bool(data.get('flags',{}).get('pulse')))|524288*int(bool(data.get('flags',{}).get('powder')))|16384*int(data.get('secondaries') is not None)|262144*int(bool(data.get('flags',{}).get('heal')))
        move.pressure_target=int(bool(data.get('flags',{}).get('mustpressure')) or data['target'] not in ('self','allySide','allyTeam','foeSide'))
        move.power = data['basePower']
        move.accuracy = 101 if data['accuracy'] is True else data['accuracy']
        move.priority = data['priority']
        move.type = TYPES.index(data['type'])
        move.category = ('Physical', 'Special', 'Status').index(data['category'])
        move.maxpp = data['pp'] if data.get('noPPBoosts') else data['pp'] * 8 // 5
        move.crit_ratio = data.get('critRatio',1)
        move.always_crit = bool(data.get('willCrit',False))
        move.ignore_immunity = bool(data.get('ignoreImmunity',move.category==2))
        if data.get('multihit'):
            hits=data['multihit']
            lo,hi=hits if isinstance(hits,list) else (hits,hits)
            if not 1<=lo<=hi<=10:raise ValueError('Unsupported native multi-hit count')
            move.hit_min,move.hit_max=lo,hi
            move.multi_accuracy=bool(data.get('multiaccuracy'))
        if data.get('smartTarget') and data['id']!='dragondarts':raise ValueError('Unsupported native smart target')
        if data.get('selfBoost'):
            if set(data['selfBoost'])!={'boosts'}:raise ValueError('Unsupported native after-move self effect')
            for name,delta in data['selfBoost']['boosts'].items():move.after_stat_delta[STATS.index(name)]=delta
        if data.get('damage'):
            if data['damage']=='level':move.fixed_damage=-1
            elif isinstance(data['damage'],int):move.fixed_damage=data['damage']
            else:raise ValueError('Unsupported native fixed damage')
        move.target_self = data['target'] in ('self','all','allySide','allyTeam')
        if data.get('ignoreAbility') and (data['id'] not in ('moongeistbeam','sunsteelstrike','photongeyser') or data['ignoreAbility'] is not True):raise ValueError('Unimplemented native ability-ignore move')
        if data.get('pseudoWeather'):move.effects|=16384
        if data.get('terrain'):
            move.effects|=4096;move.terrain=('', 'electricterrain','grassyterrain','mistyterrain','psychicterrain').index(data['terrain'])
        if data.get('weather'):
            move.effects|=1024;move.weather={'raindance':1,'sunnyday':2,'sandstorm':3,'snowscape':4,'hail':5}[data['weather'].lower()]
        move.protectable = bool(data.get('flags',{}).get('protect'))
        if data.get('sideCondition'):
            move.effects|=2048;move.side_condition=('stealthrock','spikes','toxicspikes','stickyweb','reflect','lightscreen','auroraveil','tailwind','safeguard','mist').index(data['sideCondition'])
        if protect:move.effects |= 1
        if data.get('forceSwitch'):move.effects|=8192
        if data.get('selfSwitch') is True:
            move.effects |= 128
        for field, flag in [('heal', 2), ('drain', 4), ('recoil', 8)]:
            if data.get(field):
                move.effects |= flag
                prefix = field if field != 'heal' else 'heal'
                setattr(move, prefix + '_num', data[field][0])
                setattr(move, prefix + '_den', data[field][1])
        if data.get('boosts'):
            move.effects |= 16
            for name, delta in data['boosts'].items():
                move.stat_delta[STATS.index(name)] = delta
        if data.get('volatileStatus') and not protect:
            if data['volatileStatus'] not in ('confusion','flinch','taunt','encore','disable','leechseed','yawn','substitute','healblock','curse','saltcure','noretreat','magnetrise','partiallytrapped','destinybond'):raise ValueError('Unsupported native volatile')
            move.effects|=512;move.volatile_status={'confusion':1,'flinch':2,'taunt':3,'encore':4,'disable':5,'leechseed':6,'yawn':7,'substitute':8,'healblock':9,'curse':11,'saltcure':12,'noretreat':13,'magnetrise':14,'partiallytrapped':15,'destinybond':16}[data['volatileStatus']]
        if data.get('status'):
            move.effects |= 32
            move.secondary_status = STATUSES.index(data['status'])
        if data.get('self') and not ((data['id']=='shedtail' and data['self']=={}) or (data['id']=='gigaimpact' and data['self']=={'volatileStatus':'mustrecharge'}) or (data['id']=='roost' and data['self']=={'volatileStatus':'roost'}) or (data['id']=='doubleshock' and data['self']=={}) or (data['id']=='glaiverush' and data['self']=={'volatileStatus':'glaiverush'}) or (data['id'] in ('outrage','petaldance','thrash') and data['self']=={'volatileStatus':'lockedmove'})):
            payload=data['self']
            if set(payload)-{'boosts','chance'} or not payload.get('boosts'):
                raise ValueError('Unsupported native primary self effect')
            move.effects |= 256
            move.self_chance=payload.get('chance',100)
            for name,delta in payload['boosts'].items():move.self_stat_delta[STATS.index(name)]=delta
        secondaries = data.get('secondaries') or []
        if len(secondaries)>1:
            if data['id'] not in ('icefang','firefang','triplearrows') or len(secondaries)!=2 or secondaries[1]!={'chance':30 if data['id']=='triplearrows' else 10,'volatileStatus':'flinch'}:raise ValueError('Unimplemented multiple secondary payload')
            move.additional_flinch_chance=secondaries[1]['chance'];secondaries=secondaries[:1]
        for secondary in secondaries:
            if data['id']=='sparklingaria':
                if secondary!={'chance':100,'volatileStatus':'sparklingaria'}:raise ValueError('Unimplemented Sparkling Aria payload')
                secondary={'chance':100}
            if set(secondary) - {'chance', 'status', 'boosts', 'self', 'volatileStatus'}:
                raise ValueError('Unsupported native secondary payload')
            move.effects |= 64
            move.secondary_chance = secondary.get('chance', 100)
            payload = secondary.get('self', secondary)
            move.secondary_self = 'self' in secondary
            if set(payload) - {'chance', 'status', 'boosts', 'volatileStatus'}:
                raise ValueError('Unsupported native secondary self payload')
            if payload.get('volatileStatus'):
                if payload['volatileStatus'] not in ('confusion','flinch','taunt','encore','disable','leechseed','yawn','substitute','healblock','curse','saltcure','noretreat','magnetrise','partiallytrapped','destinybond'):raise ValueError('Unsupported native secondary volatile')
                move.secondary_volatile={'confusion':1,'flinch':2,'taunt':3,'encore':4,'disable':5,'leechseed':6,'yawn':7,'substitute':8,'healblock':9,'curse':11,'saltcure':12,'noretreat':13,'magnetrise':14,'partiallytrapped':15,'destinybond':16}[payload['volatileStatus']]
            if payload.get('status'):
                move.secondary_status = STATUSES.index(payload['status'])
            for name, delta in payload.get('boosts', {}).items():
                move.secondary_stat_delta[STATS.index(name)] = delta
        if move.category==2 and not move.effects and move.kind not in (10,11,22,23,25,26,32,33,36,38,43,46,56,59,60,88,98,104):raise ValueError('Effectless native status moves are not admitted')
    return moves
