"""Independent oracle regression matrix for CPU completion additions."""
import argparse
import json
from pathlib import Path
from .teams_check import run as check


def scenarios():
    plans = []
    def mon(moves, **kw):
        return dict(moves=moves.split(), **kw)
    def pair(a, b, **kw):
        plans.append(dict(teams=[[a, mon('tackle')], [b, mon('tackle')]], **kw))
    for form in ('Tauros', 'Tauros-Paldea-Combat', 'Tauros-Paldea-Blaze', 'Tauros-Paldea-Aqua'):
        pair(mon('ragingbull', species=form), mon('reflect lightscreen auroraveil tackle'))
        pair(mon('ragingbull', species=form), mon('substitute protect icebeam', species='Gengar'), tera=True)
    for form in ('Hoopa', 'Hoopa-Unbound', 'Mew'):
        pair(mon('hyperspacefury tackle', species=form), mon('protect substitute tackle'))
    for ability in ('No Ability', 'Sand Force', 'Damp', 'Mold Breaker'):
        pair(mon('sandstorm earthquake stoneedge ironhead', ability=ability), mon('tackle'))
        pair(mon('explosion tackle', ability=ability), mon('explosion tackle', ability='Damp'))
        pair(mon('tackle', ability=ability), mon('explosion', ability='Aftermath', hp_fraction=.1))
    for a, b in [('fusionflare', 'fusionbolt'), ('fusionbolt', 'fusionflare')]:
        pair(mon(a), mon(b))
        pair(mon(a, speed_ev=0), mon(b, speed_ev=4, item='Choice Scarf'))
        pair(mon(a), mon(b + ' protect substitute', item='Leftovers'))
    pair(mon('courtchange stealthrock spikes stickyweb'), mon('reflect lightscreen toxicspikes tailwind'))
    pair(mon('courtchange tackle'), mon('courtchange tackle'))
    for target in ('No Ability', 'Mirror Armor', 'Contrary', 'Defiant', 'Clear Body'):
        pair(mon('growl metalSound tickle chillingwater', ability='Mirror Armor'), mon('tickle acid charm', ability=target))
    for target in ('No Ability', 'Own Tempo', 'Synchronize'):
        for item in ('', 'Lum Berry'):
            pair(mon('toxic mortalspin malignantchain', species='Pecharunt', ability='Poison Puppeteer'), mon('tackle', ability=target, item=item))
    for ability in ('Hadron Engine', 'Orichalcum Pulse'):
        for opponent in ('No Ability', 'Cloud Nine', 'Unaware', 'Thick Fat'):
            pair(mon('thunderbolt tackle terablast', ability=ability),
                 mon('raindance psychicterrain earthquake', ability=opponent), switch_every=9, tera=True)
    for ability in ('As One (Glastrier)', 'As One (Spectrier)'):
        for item in ('Sitrus Berry', 'Lum Berry', 'Leppa Berry'):
            pair(mon('tackle toxic shadowball', ability=ability), mon('tackle', item=item), switch_every=11)
    for status in ('', 'slp', 'par'):
        pair(mon('tackle protect sleeptalk', ability='Truant', status=status), mon('tackle'), switch_every=13)
    for target in ('No Ability', 'Good as Gold', 'Overcoat', 'Magic Guard'):
        pair(mon('spore toxic earthquake', ability='Mycelium Might'), mon('taunt icebeam', ability=target))
    for ability in ('No Ability', 'Truant', 'Mold Breaker'):
        for target in ('No Ability', 'Shield Dust', 'Magic Guard'):
            pair(mon('gigaimpact tackle', ability=ability), mon('protect substitute thunderwave tackle', ability=target), tera=True)
    for move in ('spiritshackle', 'whirlpool', 'magmastorm'):
        for target in ('Mew', 'Gengar'):
            pair(mon(move + ' protect tackle'), mon('recover substitute tackle', species=target), switch_every=5)
        pair(mon(move + ' uturn tackle'), mon('protect tackle', ability='Magic Guard'), switch_every=7)
    for species, item in (('Mew', ''), ('Venusaur', ''), ('Mew', 'Safety Goggles')):
        pair(mon('tackle bulletseed', species=species, item=item), mon('tackle', ability='Effect Spore'))
    for ability in ('Berserk', 'Anger Shell', 'Electromorphosis'):
        for item in ('', 'Sitrus Berry', 'Leftovers'):
            for attacker in ('No Ability', 'Sheer Force'):
                pair(mon('tackle doublehit icebeam thunderwave', ability=attacker),
                     mon('thunderbolt gigaimpact protect', ability=ability, item=item))
    for species in ('Greninja-Bond', 'Mew'):
        pair(mon('waterpulse darkpulse watershuriken', ability='Battle Bond', species=species),
             mon('tackle', hp_fraction=.1), switch_every=11)
    for move in ('chillyreception', 'focuspunch', 'beakblast'):
        for attack in ('tackle', 'icebeam', 'fakeout', 'thunderwave'):
            pair(mon(move + ' sleeptalk protect', item='Leftovers'), mon(attack + ' substitute protect'))
        pair(mon(move, status='slp'), mon('tackle'))
        pair(mon(move, ability='Truant'), mon(move))
    for item in ('Sitrus Berry', 'Lum Berry', 'Chesto Berry', 'Leppa Berry'):
        pair(mon('bugbite toxic', item='Leftovers', ability='Cheek Pouch'), mon('protect tackle', item=item))
    for move in ('destinybond', 'shedtail', 'healingwish', 'mirrorcoat', 'beatup'):
        for attack in ('tackle', 'icebeam', 'doublehit', 'toxic'):
            pair(mon(move + ' protect tackle'), mon(attack + ' substitute uturn'))
        pair(mon(move + ' tackle', ability='Prankster'), mon('darkpulse tackle', species='Umbreon'), tera=True)
    for ability in ('No Ability', 'Magic Bounce', 'Mold Breaker', 'Prankster', 'Mycelium Might'):
        pair(mon('toxic taunt willowisp tackle', ability=ability), mon('toxic growl substitute', ability='Magic Bounce'))
        pair(mon('stealthrock spikes stickyweb tackle', ability=ability), mon('toxicspikes tackle', ability='Magic Bounce'))
    for ability in ('Magician', 'Pickpocket'):
        for item in ('Leftovers', 'Choice Band', 'Sitrus Berry', 'Rocky Helmet', 'Air Balloon'):
            pair(mon('tackle uturn icebeam', ability=ability), mon('tackle doublehit icebeam', item=item))
            pair(mon('tackle doublehit icebeam', item=item), mon('tackle uturn icebeam', ability=ability))
    for ability in ('Intimidate', 'Drizzle', 'Hadron Engine', 'Unaware', 'Unnerve', 'Truant', 'Trace', 'Magic Bounce'):
        pair(mon('tackle toxic calmmind', ability='Trace'), mon('tackle toxic', ability=ability), switch_every=7)
    for species in ('Gallade', 'Jynx', 'Mew'):
        for ability in ('No Ability', 'Oblivious', 'Aroma Veil'):
            pair(mon('tackle doublehit', species=species, ability=ability), mon('tackle thunderwave', species='Jynx', ability='Cute Charm'), switch_every=9)
    for item in ('Sitrus Berry', 'Lum Berry', 'Chesto Berry', 'Leppa Berry'):
        pair(mon('rest tackle', item=item, ability='Cud Chew'), mon('tackle toxic'), switch_every=17)
    for move in ('swordsdance', 'quiverdance', 'petaldance', 'dragondance', 'fierydance'):
        for item in ('', 'Choice Scarf', 'Life Orb'):
            pair(mon(move + ' tackle'), mon('tackle protect', ability='Dancer', item=item))
        pair(mon(move + ' tackle', ability='Dancer'), mon(move + ' tackle', ability='Dancer'))
    for tera in (False, True):
        pair(mon('relicsong psychic closecombat', species='Meloetta', ability='Serene Grace'), mon('tackle toxic'), switch_every=9, tera=tera)
        pair(mon('aurawheel crunch protect', species='Morpeko', ability='Hunger Switch'), mon('tackle toxic'), switch_every=9, tera=tera)
        pair(mon('flipturn jetpunch drainpunch', species='Palafin', ability='Zero to Hero'), mon('tackle toxic'), switch_every=9, tera=tera)
        pair(mon('surf hurricane roost', species='Cramorant', ability='Gulp Missile'), mon('tackle icebeam'), switch_every=9, tera=tera)
    for attack in ('tackle', 'doublehit', 'icebeam', 'sunsteelstrike'):
        pair(mon('shadowclaw curse playrough', species='Mimikyu', ability='Disguise'), mon(attack + ' toxic'))
        pair(mon('snowscape avalanche substitute', species='Eiscue', ability='Ice Face'), mon(attack + ' toxic'))
    for fraction in (1, .4):
        pair(mon('shellsmash powergem acrobatics', species='Minior', ability='Shields Down', hp_fraction=fraction), mon('tackle toxic yawn'), switch_every=11)
    for ability in ('No Ability', 'Imposter'):
        for opponent in ('Intimidate', 'Levitate', 'Unaware', 'Protosynthesis', 'Good as Gold', 'Mold Breaker'):
            pair(mon('transform', species='Ditto', ability=ability, item='Choice Scarf'),
                 mon('swordsdance earthquake recover', species='Garchomp', ability=opponent), switch_every=11, tera=True)
        pair(mon('transform', species='Ditto', ability=ability), mon('shadowclaw substitute', species='Mimikyu', ability='Disguise'))
        pair(mon('transform', species='Ditto', ability=ability), mon('aurawheel protect', species='Morpeko', ability='Hunger Switch'))
        pair(mon('transform', species='Ditto', ability=ability), mon('transform', species='Ditto', ability=ability))
    for tera in (False, True):
        for attack in ('tackle', 'doublehit', 'closecombat', 'icebeam'):
            pair(mon('terastarstorm calmmind earthpower', species='Terapagos', ability='Tera Shift', tera='Stellar'), mon(attack + ' protect'), switch_every=11, tera=tera)
        pair(mon('terastarstorm ancientpower', species='Terapagos', ability='Tera Shift', tera='Stellar'), mon('sandstorm psychicterrain tackle'), tera=tera)
        for species, ability, item, typ in (('Ogerpon','Defiant','Leftovers','Grass'), ('Ogerpon-Wellspring','Water Absorb','Wellspring Mask','Water'), ('Ogerpon-Hearthflame','Mold Breaker','Hearthflame Mask','Fire'), ('Ogerpon-Cornerstone','Sturdy','Cornerstone Mask','Rock')):
            pair(mon('ivycudgel hornleech swordsdance', species=species, ability=ability, item=item, tera=typ), mon('knockoff icebeam trick'), switch_every=13, tera=tera)
    for ability in ('No Ability', 'Infiltrator', 'Magic Guard', 'Analytic'):
        for target in ('No Ability', 'Magic Guard', 'Stamina', 'Cursed Body', 'Berserk'):
            pair(mon('futuresight psychic protect', ability=ability, item='Life Orb'), mon('substitute protect tackle', ability=target, item='Sitrus Berry'), switch_every=7)
    for species in ('Umbreon', 'Mimikyu', 'Blissey'):
        pair(mon('futuresight teleport psychic', ability='Regenerator'), mon('tackle toxic protect', species=species, ability='Disguise' if species=='Mimikyu' else 'No Ability'), tera=True)
    for species in ('Zoroark', 'Zoroark-Hisui'):
        for attack in ('tackle', 'icebeam', 'toxic', 'knockoff', 'futuresight'):
            for tera in (False, True):
                pair(mon('nastyplot darkpulse uturn', species=species, ability='Illusion', item='Life Orb'), mon(attack + ' substitute protect'), tera=tera, switch_every=9)
    for species, ability, item, typ, move in (('Terapagos','Tera Shift','Leftovers','Stellar','terastarstorm'), ('Ogerpon','Defiant','Leftovers','Grass','ivycudgel'), ('Ogerpon-Wellspring','Water Absorb','Wellspring Mask','Water','ivycudgel'), ('Ogerpon-Hearthflame','Mold Breaker','Hearthflame Mask','Fire','ivycudgel'), ('Ogerpon-Cornerstone','Sturdy','Cornerstone Mask','Rock','ivycudgel')):
        for attack in ('tackle', 'icebeam', 'knockoff', 'sandstorm'):
            pair(mon(move + ' swordsdance substitute', species=species, ability=ability, item=item, tera=typ), mon(attack + ' protect psychicterrain'), tera=True)
    for ability in ('No Ability', 'Unburden', 'Cud Chew', 'Cheek Pouch'):
        for opponent in ('No Ability', 'Unnerve', 'As One (Glastrier)'):
            pair(mon('tackle protect earthquake', item='Custap Berry', hp_fraction=.2, ability=ability), mon('icebeam tackle', ability=opponent))
        pair(mon('icebeam thunderbolt tackle', item='Expert Belt', ability=ability), mon('drainingkiss leechlife', species='Mimikyu', ability='Disguise'))
    pair(mon('protect transform', species='Ditto', ability='No Ability'), mon('aurawheel', species='Morpeko', ability='Hunger Switch'), schedule=[[0,0],[1,0]])
    plans.append(dict(teams=[[mon('protect', pp=[2000])],[mon('protect',pp=[2000])]], phases=1100))
    for ability in ('No Ability','Regenerator','Mold Breaker'):
        for species, target_ability, moves in (('Cramorant','Gulp Missile','surf protect roost'),('Terapagos','Tera Shift','terastarstorm protect calmmind'),('Mimikyu','Disguise','shadowclaw protect curse')):
            pair(mon('futuresight teleport psychic', ability=ability, item='Life Orb', tera='Stellar'),mon(moves,species=species,ability=target_ability,hp_fraction=.4),tera=True)
    return plans


def run(output):
    output = Path(output)
    specs = [dict(seed=f'{seed},{seed+1},{seed+2},{seed+3}', policy_seed=seed*31,
                  observe=True, **({'phases':256}|options))
             for options in scenarios() for seed in range(1, 5)]
    output.parent.mkdir(parents=True, exist_ok=True)
    path = output.with_suffix('.specs.json')
    with path.open('x') as stream:
        json.dump(specs, stream)
    check(output, specs_file=path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
