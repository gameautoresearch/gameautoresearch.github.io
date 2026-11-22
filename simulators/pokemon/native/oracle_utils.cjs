'use strict';
// Preserve callback obligations that JSON would otherwise silently drop.
function callbackPaths(value,prefix='') {
  const paths=[];
  for(const [key,child] of Object.entries(value)) {
    const name=prefix?`${prefix}.${key}`:key;
    if(typeof child==='function')paths.push(name);
    else if(child&&typeof child==='object')paths.push(...callbackPaths(child,name));
  }
  return paths;
}
function serializeMove(move) {
  return {...move,native_callbacks:Object.keys(move).filter(k=>typeof move[k]==='function'),
    native_callback_paths:callbackPaths(move)};
}
// The format a fixture is generated in: unmodified gen9randombattle, or the same format with a
// smaller Max Team Size custom rule (pinned Showdown source unchanged; its generator honours it).
function formatId(teamSize) {
  if(teamSize===undefined||teamSize===null||teamSize===6)return 'gen9randombattle';
  if(!Number.isInteger(teamSize)||teamSize<1||teamSize>6)throw Error('Team size must be 1..6');
  return `gen9randombattle@@@Max Team Size = ${teamSize}`;
}
module.exports={serializeMove,formatId};
