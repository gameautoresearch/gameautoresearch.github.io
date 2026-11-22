"""Validate generated rule constants against independently observed oracle fields."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
from data_codegen import emit_rules,rule_values,DECK
HERE=Path(__file__).resolve().parent

def main():
    source=Path(sys.argv[1]);oracle=Path(sys.argv[2]);output=Path(sys.argv[3]);output.mkdir(parents=True,exist_ok=False)
    data=json.loads(source.read_text());observed=json.loads(oracle.read_text());config,cards=rule_values(data)
    order=['cost','productionStopMs','unitType','count','summonRadius','summonWidth','summonDelayMs','deployTimeMs','collisionRadius','spawnRadius','spawnAngleShift','flyingHeight','spawnPathfindSpeed','tileSizeOverride','margin','startY','endY','enemyOk','onBuildings','onWater','fullLane','building','spellAsDeploy']
    for name,card in zip(DECK,cards):
        got=[int(card[k]) for k in order];expected=observed['card_fields'][name].copy();expected[2]=-1 if data['cards'][name]['unit'] is None else data['unit_ids'][data['cards'][name]['unit']['name']]
        assert got==expected,(name,got,expected)
    flat=[config[k] for k in ['startingElixir','maxMana','startCooldownTicks','endDelayMs','sectionCount','rateCount','cooldownCount','eventCount']]
    for k,n in [('sectionLengths',8),('sectionTypes',8),('rateLengths',8),('fullBarMs',8),('cooldownLengths',8),('cooldownMs',8),('eventTimes',16)]:flat+=config[k]+[0]*(n-len(config[k]))
    assert flat==observed['config_fields']
    header=output/'rules-data.hpp';header.write_text(emit_rules(data))
    cpp=output/'check.cpp';cpp.write_text('#include "'+str(HERE/'rules.hpp')+'"\n#include "'+str(header.resolve())+'"\n#include <cassert>\nint main(){using namespace clash_native;auto c=fixed8_rules_config();assert(c.maxMana==10&&c.startCooldownTicks==90&&c.endDelayMs==4000);for(int i=0;i<8;++i){auto card=fixed8_card_rule(i);assert(card.cost>0);assert(card.unitType==(i<5?i:-1));}auto arena=fixed8_arena_view();assert(arena.width==36&&arena.height==64&&arena.bits(0,0)==16);}\n')
    binary=output/'check';command=[shutil.which('clang++') or shutil.which('g++'),'-std=c++20','-Wall','-Wextra','-Werror',str(cpp),'-o',str(binary)];subprocess.run(command,check=True);subprocess.run([str(binary.resolve())],check=True)
    result=dict(status='passed',cards=8,arena_cells=2304,config_fields=len(flat),input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),generator_sha256=hashlib.sha256((HERE/'data_codegen.py').read_bytes()).hexdigest(),oracle=str(oracle),compiler_command=command,complete_task_parity=False)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
