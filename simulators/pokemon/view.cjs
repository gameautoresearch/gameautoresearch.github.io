'use strict';
// The encoder accepts ONLY one channel's protocol plus that player's request.
// Dex is public rules data; no Battle/Pokemon engine objects enter this module.
const path = require('node:path');
const {Dex} = require(path.resolve(__dirname, '../../candidates/pokemon-showdown/dist/sim'));
const dex = Dex.mod('gen9');
const id = s => (s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
const SCHEMA = {version: 'pokemon-visible-v1', buckets: 16384, mons: 12, mon_tokens: 12,
  mon_features: 24, global_features: 40, global_tokens: 4, actions: 14, action_tokens: 3, action_features: 12};
function token(prefix, s) {
  if (!s) return 0;
  let h = 2166136261;
  for (const c of `${prefix}:${id(s)}`) h = Math.imul(h ^ c.charCodeAt(0), 16777619) >>> 0;
  return 1 + h % (SCHEMA.buckets - 1);
}
// Public string -> categorical token memoization. Bounded per process; cache
// eviction changes only work performed, never token values or game randomness.
const tokenCache = new Map();
function cachedToken(prefix, s) {
  if (!s) return 0;
  const k = `${prefix}:${s}`;
  if (tokenCache.has(k)) return tokenCache.get(k);
  const value = token(prefix, s);
  if (tokenCache.size >= 32768) tokenCache.clear();
  tokenCache.set(k, value);
  return value;
}
function health(s = '') {
  if (s.includes('fnt')) return [0, 1, 1];
  const m = s.match(/^(\d+)(?:\/(\d+))?/);
  return m ? [Number(m[1]) / Number(m[2] || 100), 1, 0] : [0, 0, 0];
}
const boosts = ['atk','def','spa','spd','spe','accuracy','evasion'];
const hazards = ['stealthrock','spikes','toxicspikes','stickyweb','reflect','lightscreen','auroraveil','tailwind','safeguard','mist'];
function key(ident) { return ident.replace(/^(p[12])[a-z]: /, '$1: '); }
class View {
  constructor(side, tokenMemo=false, monMemo=false) {
    this.tokenMemo = tokenMemo; this.monMemo=monMemo; this.monCache=new Map();
    this.side = side; this.request = null; this.turn = 0; this.mons = {};
    this.order = {p1: [], p2: []}; this.active = {p1: null, p2: null};
    this.conditions = {p1: {}, p2: {}}; this.weather = ''; this.terrain = ''; this.fields = {};
    this.tera = {p1: false, p2: false}; this.errors = []; this.winner = null;
  }
  mon(ident, details) {
    const k = key(ident), side = k.slice(0, 2);
    if (!this.mons[k]) {
      this.mons[k] = {ident: k, moves: [], boosts: {}, volatile: {}, condition: ''};
      this.order[side].push(k);
    }
    const p = this.mons[k];
    if (details) { p.details = details; p.species = details.split(',')[0]; p.level = Number((details.match(/L(\d+)/) || [0,100])[1]); }
    return p;
  }
  receive(text) {
    for (const line of text.split('\n')) {
      const x = line.split('|'), t = x[1];
      if (t === 'request') { this.request = JSON.parse(x.slice(2).join('|')); this.syncOwn(); continue; }
      if (t === 'error') { this.errors.push(x.slice(2).join('|')); continue; }
      if (t === 'win') this.winner = x[2];
      if (t === 'tie') this.winner = '';
      if (t === 'turn') this.turn = Number(x[2]);
      if (['switch','drag','replace'].includes(t)) {
        const p = this.mon(x[2], x[3]), side = x[2].slice(0,2);
        if (t === 'replace') {
          // Illusion's reveal omits HP; retain the publicly observed active HP.
          const previous = this.mons[this.active[side]];
          if (previous && previous !== p) { p.condition = previous.condition; p.boosts = {...previous.boosts}; p.volatile = {...previous.volatile}; }
        } else { p.boosts = {}; p.volatile = {}; }
        this.active[side] = p.ident; if (x[4]) p.condition = x[4];
      } else if (['detailschange','-formechange'].includes(t)) this.mon(x[2], x[3]);
      else if (['-damage','-heal','-sethp'].includes(t)) this.mon(x[2]).condition = x[3];
      else if (t === 'faint') this.mon(x[2]).condition = '0 fnt';
      else if (t === 'move') { const p = this.mon(x[2]); if (!p.moves.includes(id(x[3]))) p.moves.push(id(x[3])); p.lastMove = x[3]; }
      else if (t === '-status') { const p = this.mon(x[2]); p.condition = p.condition.replace(/ (brn|par|slp|frz|psn|tox)$/, '') + ' ' + x[3]; }
      else if (t === '-curestatus') { const p = this.mon(x[2]); p.condition = p.condition.replace(/ (brn|par|slp|frz|psn|tox)$/, ''); }
      else if (t === '-ability') this.mon(x[2]).ability = x[3];
      else if (t === '-item') this.mon(x[2]).item = x[3];
      else if (t === '-enditem') this.mon(x[2]).item = '';
      else if (t === '-terastallize') { this.mon(x[2]).terastallized = x[3]; this.tera[x[2].slice(0,2)] = true; }
      else if (['-boost','-unboost','-setboost'].includes(t)) {
        const p = this.mon(x[2]); p.boosts[x[3]] = t === '-setboost' ? Number(x[4]) : (p.boosts[x[3]] || 0) + Number(x[4]) * (t === '-boost' ? 1 : -1);
      } else if (t === '-clearallboost') { for (const p of Object.values(this.mons)) p.boosts = {}; }
      else if (t === '-clearboost') this.mon(x[2]).boosts = {};
      else if (t === '-start') this.mon(x[2]).volatile[id(x[3])] = true;
      else if (t === '-end') delete this.mon(x[2]).volatile[id(x[3])];
      else if (t === '-weather') this.weather = x[2] === 'none' ? '' : x[2];
      else if (t === '-fieldstart') { this.fields[id(x[2])] = true; if (x[2].includes('Terrain')) this.terrain = x[2]; }
      else if (t === '-fieldend') { delete this.fields[id(x[2])]; if (x[2] === this.terrain) this.terrain = ''; }
      else if (t === '-sidestart') { const c = id(x[3].replace(/^move: /,'')), side = x[2].slice(0,2); this.conditions[side][c] = (this.conditions[side][c] || 0) + 1; }
      else if (t === '-sideend') delete this.conditions[x[2].slice(0,2)][id(x[3].replace(/^move: /,''))];
    }
  }
  syncOwn() {
    const r = this.request;
    if (!r?.side) return;
    for (const q of r.side.pokemon) { const p = this.mon(q.ident, q.details); Object.assign(p, q); p.species = q.details.split(',')[0]; }
  }
  potential() {
    const foe = this.side === 'p1' ? 'p2' : 'p1';
    const dead = s => this.order[s].filter(k => health(this.mons[k].condition)[2]).length;
    return (dead(foe) - dead(this.side)) / 6;
  }
  encode() {
    const token = this.tokenMemo ? cachedToken : module.exports.token;
    const r = this.request, own = this.side, foe = own === 'p1' ? 'p2' : 'p1';
    const ownMons = (r?.side?.pokemon || []).map(q => this.mons[key(q.ident)]);
    const foeMons = this.order[foe].map(k => this.mons[k]);
    const encodeMon = (p, side) => {
      if (!p) return [Array(12).fill(0), Array(24).fill(0)];
      // Only fields read by this encoder enter the memo key. The cache is
      // per player's visible state, never shared with the other private view.
      const memoKey=this.monMemo ? JSON.stringify([p.species,p.ability,p.baseAbility,p.item,p.terastallized,p.teraType,p.condition,p.moves,p.lastMove,p.level,p.stats,p.boosts,p.volatile,this.active[side]===key(p.ident),side===own]) : null;
      if(this.monMemo && this.monCache.get(p.ident)?.key===memoKey)return this.monCache.get(p.ident).value;
      const species = dex.species.get(p.species), types = p.terastallized ? [p.terastallized] : species.types;
      const moves = p.moves || [], hp = health(p.condition), status = ((p.condition || '').match(/ (brn|par|slp|frz|psn|tox)$/) || [0,''])[1];
      const ts = [token('species',p.species),token('ability',p.ability || p.baseAbility),token('item',p.item),
        token('type',types[0]),token('type',types[1]),token('type',p.teraType || p.terastallized),token('status',status),
        ...[0,1,2,3].map(i => token('move', moves[i])),token('move',p.lastMove)];
      const ns = [1,Number(this.active[side] === key(p.ident)),...hp,(p.level || 100)/100,
        ...['atk','def','spa','spd','spe'].map(s => (p.stats?.[s] || species.baseStats[s] || 0)/300),
        ...boosts.map(s => (p.boosts[s] || 0)/6),Number(!!p.terastallized),Number(!!p.volatile.substitute),
        Number(!!p.volatile.confusion),Number(!!p.volatile.leechseed),Number(!!p.volatile.taunt),Number(side === own)];
      const value=[ts,ns];
      if(this.monMemo)this.monCache.set(p.ident,{key:memoKey,value});
      return value;
    };
    const all = [...Array.from({length:6},(_,i)=>encodeMon(ownMons[i],own)), ...Array.from({length:6},(_,i)=>encodeMon(foeMons[i],foe))];
    const mask = Array(14).fill(false), at = Array.from({length:14},()=>[0,0,0]), an = Array.from({length:14},()=>Array(12).fill(0));
    const active = r?.active?.[0], target = this.mons[this.active[foe]], targetSpecies = target ? dex.species.get(target.species) : null;
    const targetTypes = target?.terastallized ? [target.terastallized] : targetSpecies?.types;
    const actor = ownMons.find(p => p.active), actorTypes = actor ? dex.species.get(actor.species).types : [];
    for (let i = 0; i < 4; i++) {
      const m = active?.moves[i]; if (!m) continue;
      const d = dex.moves.get(m.id || m.move);
      const effectiveness = targetTypes ? (dex.getImmunity(d.type,targetTypes) ? Math.pow(2,dex.getEffectiveness(d.type,targetTypes)) : 0) : 1;
      for (const tera of [0,1]) {
        const j = i + tera*4;
        mask[j] = !r.wait && !m.disabled && (!tera || !!active.canTerastallize);
        at[j] = [token('move',d.id),token('type',d.type),token('category',d.category)];
        an[j] = [(d.basePower || 0)/150,(d.accuracy === true ? 100 : (d.accuracy || 0))/100,(d.priority || 0)/5,(m.pp || 0)/64,(m.maxpp || 0)/64,
          Number(d.category === 'Physical'),Number(d.category === 'Special'),Number(d.category === 'Status'),effectiveness/4,
          Number(actorTypes.includes(d.type)),tera,0];
      }
    }
    for (let i = 0; i < ownMons.length; i++) {
      const p = ownMons[i], hp = health(p.condition), reviving = !!r?.side?.pokemon[0]?.reviving;
      mask[8+i] = !r.wait && !p.active && (reviving ? !!hp[2] : !hp[2]) && (!!r.forceSwitch?.[0] || !active?.trapped);
      at[8+i] = [token('species',p.species),token('type',dex.species.get(p.species).types[0]),token('type',dex.species.get(p.species).types[1])];
      an[8+i] = [hp[0],Number(!!p.terastallized),0,0,0,0,0,0,0,0,0,1];
    }
    if (!r || r.wait) mask.fill(false);
    if (r?.teamPreview) throw Error('Team preview unsupported in pinned random-battle task');
    const g = [Math.min(this.turn,1000)/1000,Number(!!r?.forceSwitch),Number(!!active?.trapped),Number(!!active?.maybeTrapped),
      Number(!!active?.canTerastallize),Number(this.tera[foe]),this.order[foe].length/6,this.potential(),
      ...[own,foe].flatMap(s=>hazards.map(h=>(this.conditions[s][h] || 0)/3)),
      ...['trickroom','gravity','magicroom','wonderroom'].map(f=>Number(!!this.fields[f])),
      Number(!!r?.wait),Number(!!actor?.terastallized),0,0,0,0,0,0];
    if (g.length !== 40 || all.some(x=>x[1].length !== 24)) throw Error('Schema length mismatch');
    return {tokens: all.map(x=>x[0]), mons: all.map(x=>x[1]), global: g,
      global_tokens: [token('weather',this.weather),token('terrain',this.terrain),0,0], action_tokens:at, actions:an, mask};
  }
}
function choice(a) {
  if (!Number.isInteger(a) || a < 0 || a >= 14) throw Error(`Bad action ${a}`);
  return a < 8 ? `move ${a%4+1}${a>=4 ? ' terastallize' : ''}` : `switch ${a-7}`;
}
module.exports = {View, choice, SCHEMA, token, health};
