'use strict';
// Semantics-preserving specialization of the pinned getCallback implementation.
// The upstream method only transforms onSwitchIn callbacks; all other event
// names return effect[callbackName]. Avoid its instanceof chain on that path.
// Retain upstream for SwitchIn, including special cases and future flags.
function installFastCallbacks(battle) {
  const original=battle.getCallback;
  battle.getCallback=function(target,effect,callbackName) {
    if(callbackName!=='onSwitchIn')return effect[callbackName];
    return original.call(this,target,effect,callbackName);
  };
}
module.exports={installFastCallbacks};
