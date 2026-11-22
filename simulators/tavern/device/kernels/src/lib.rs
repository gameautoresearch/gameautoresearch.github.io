//! Complete device transitions from the independently compiled shared rule source.
//! No oracle library is linked into this crate or called from these kernels.
#![cfg_attr(target_arch="nvptx64",no_std)]
extern crate alloc;
use alloc::boxed::Box;
use cuda_std::{kernel,thread};
use tavern_device_core::{Game,RulesConfig};
use tavern_device_core::engine::StoppedReason;

use tavern_device_wire::Episode;
fn splitmix64(mut x:u64)->u64{x=x.wrapping_add(0x9e3779b97f4a7c15);x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
fn game_seed(seed:u64,g:usize,index:u64)->u64{splitmix64(splitmix64(seed^splitmix64(g as u64))^index)}
fn fault(g:&Game)->u8{
    if let Some(r)=g.stopped_reason{return match r{StoppedReason::NoProgress=>2,StoppedReason::DriverGuard=>3}}
    if g.turn_limit_end{4}else if g.truncated_combats>0{5}else if g.truncated_recruit_contexts>0{6}else{0}
}
fn over(g:&Game)->bool{g.is_over()||fault(g)!=0}
fn advance(g:&mut Game,controlled:&[usize]){
    for _ in 0..10000{
        if over(g){return}
        let mut progressed=false;
        for p in 0..g.players.len(){if !controlled.contains(&p)&&g.needs_action(p){progressed|=tavern_device_core::bot::play_turn(g,p)}}
        if controlled.iter().any(|&p|g.needs_action(p)){return}
        if !progressed {if !over(g){g.stopped_reason=Some(StoppedReason::NoProgress)}return}
    }
    if !over(g)&&!controlled.iter().any(|&p|g.needs_action(p)){g.stopped_reason=Some(StoppedReason::DriverGuard)}
}
fn fresh(seed:u64,gi:usize,index:u64,cfg:RulesConfig,seats:&[usize])->Episode{
    let mut game=Game::new(game_seed(seed,gi,index),cfg);advance(&mut game,seats);
    let mut controlled=[0u8;8];for(j,&seat)in seats.iter().enumerate(){controlled[j]=seat as u8}
    Episode{game,index,seed,finished:[false;8],controlled,k:seats.len() as u8}
}

#[kernel]
pub unsafe fn tavern_prewarm(){
    if thread::index_1d()!=0{return}
    #[cfg(target_arch="nvptx64")]{
        tavern_device_core::cards::prewarm_device_db();
        tavern_device_core::script::prewarm_device_registry();
    }
    let _=tavern_device_core::cards::db();
    let _=tavern_device_core::script::registry();
    // This full immutable text-derived cache otherwise races its expensive
    // constructors across every lobby of the first initialization kernel.
    let _=tavern_device_core::script::implemented(tavern_device_core::cards::CardId(0));
    #[cfg(target_arch="nvptx64")]
    unsafe{
        tavern_device_core::cards::publish_read_only_db();
        tavern_device_core::script::publish_read_only_registry();
    }
}
#[kernel]
pub unsafe fn tavern_init(games:*mut u64,seed:u64,n:u32,seats:*const u32,k:u32,cfg:*const u8,cfg_len:u32){
    let gi=thread::index_1d() as usize;if gi>=n as usize{return}
    unsafe{
        #[cfg(not(feature="frozen_rules"))]
        let config:RulesConfig=serde_json::from_slice(core::slice::from_raw_parts(cfg,cfg_len as usize)).expect("validated rules JSON");
        #[cfg(feature="frozen_rules")]
        let config=RulesConfig{num_players:8,content:tavern_device_core::config::ContentMode::Implemented,
            turn_action_caps:alloc::vec![30],max_actions_per_turn:30,max_turns:100,record_log:false,
            ..RulesConfig::default()};
        config.validate().expect("supported rule configuration");
        let mut controlled=[0usize;8];for j in 0..k as usize{controlled[j]=*seats.add(j) as usize}
        let e=fresh(seed,gi,0,config,&controlled[..k as usize]);
        games.add(gi).write(Box::into_raw(Box::new(e)) as u64);
    }
}
#[kernel]
pub unsafe fn tavern_observe(games:*const u64,n:u32,seats:*const u32,k:u32,active:*const u8,
    features:*mut f32,cards:*mut i32,masks:*mut u8,ready:*mut u8){
    let gi=thread::index_1d() as usize;if gi>=n as usize{return}
    unsafe{
        let e=&*(*games.add(gi) as *const Episode);
        for j in 0..k as usize{
            let slot=gi*k as usize+j;let p=*seats.add(j) as usize;
            let act=*active.add(gi)!=0&&!e.finished[j]&&!over(&e.game)&&e.game.needs_action(p);
            ready.add(slot).write(act as u8);
            let f=core::slice::from_raw_parts_mut(features.add(slot*tavern_device_core::obs::OBS_FLOATS),tavern_device_core::obs::OBS_FLOATS);
            let c=core::slice::from_raw_parts_mut(cards.add(slot*tavern_device_core::obs::OBS_CARDS),tavern_device_core::obs::OBS_CARDS);
            let m=core::slice::from_raw_parts_mut(masks.add(slot*tavern_device_core::N_ACTIONS),tavern_device_core::N_ACTIONS);
            if act{
                let o=tavern_device_core::obs::observe(&e.game,p);f.copy_from_slice(&o.floats);c.copy_from_slice(&o.cards);
                for (dst,legal) in m.iter_mut().zip(e.game.legal_mask(p)){*dst=legal as u8}
            }else{f.fill(0.);c.fill(0);m.fill(0)}
        }
    }
}
#[kernel]
pub unsafe fn tavern_step(games:*mut u64,actions:*const i64,n:u32,seats:*const u32,k:u32,active:*const u8,
    rewards:*mut f32,dones:*mut u8,places:*mut u8,rejected:*mut u8,stop:*mut u8){
    let gi=thread::index_1d() as usize;if gi>=n as usize{return}
    unsafe{
        let e=&mut *(*games.add(gi) as *mut Episode);
        let mut controlled=[0usize;8];let mut ready=[false;8];
        for j in 0..k as usize{
            let slot=gi*k as usize+j;controlled[j]=*seats.add(j) as usize;
            rewards.add(slot).write(0.);dones.add(slot).write(0);places.add(slot).write(0);rejected.add(slot).write(0);stop.add(slot).write(0);
            ready[j]=!e.finished[j]&&!over(&e.game)&&e.game.needs_action(controlled[j]);
        }
        if *active.add(gi)==0{return}
        for j in 0..k as usize{
            let p=controlled[j];if !ready[j]||over(&e.game)||!e.game.needs_action(p){continue}
            let slot=gi*k as usize+j;let a=*actions.add(slot);let legal=e.game.legal_mask(p);
            let ok=usize::try_from(a).ok().filter(|&x|x<legal.len()&&legal[x]);
            if !matches!(ok.map(|x|e.game.apply_index(p,x)),Some(Ok(()))){rejected.add(slot).write(1);rewards.add(slot).write(-0.01)}
        }
        advance(&mut e.game,&controlled[..k as usize]);
        for j in 0..k as usize{
            let p=controlled[j];let slot=gi*k as usize+j;
            if !e.finished[j]&&(over(&e.game)||!e.game.players[p].alive){
                e.finished[j]=true;let reason=fault(&e.game);let place=if reason==0{e.game.players[p].placement}else{0};
                dones.add(slot).write(1);places.add(slot).write(place);stop.add(slot).write(if reason==0{1}else{reason});
                let reward=if place==0{0.}else{1.-2.*((place-1) as f32)/(e.game.players.len()-1) as f32};
                rewards.add(slot).write(*rewards.add(slot)+reward);
            }
        }
        if e.finished[..k as usize].iter().all(|x|*x){
            let next=fresh(e.seed,gi,e.index+1,e.game.cfg.clone(),&controlled[..k as usize]);*e=next;
        }
    }
}
#[cfg(feature="state_io")]
#[kernel]
pub unsafe fn tavern_export(games:*const u64,n:u32,out:*mut u8,capacity:u32,lengths:*mut u32){
    let gi=thread::index_1d() as usize;if gi>=n as usize{return}
    unsafe{
        let e=&*(*games.add(gi) as *const Episode);
        let text=serde_json::to_vec(e).expect("state export");lengths.add(gi).write(text.len() as u32);
        if text.len()<=capacity as usize{core::slice::from_raw_parts_mut(out.add(gi*capacity as usize),text.len()).copy_from_slice(&text)}
    }
}
#[kernel]
pub unsafe fn tavern_destroy(games:*mut u64,n:u32){
    let gi=thread::index_1d() as usize;if gi>=n as usize{return}
    unsafe{let p=*games.add(gi) as *mut Episode;if !p.is_null(){drop(Box::from_raw(p));games.add(gi).write(0)}}
}

#[cfg(feature="state_io")]
#[kernel]
pub unsafe fn tavern_import(games:*mut u64,n:u32,input:*const u8,capacity:u32,lengths:*const u32,status:*mut i32,seats:*const u32,k:u32){
    let gi=thread::index_1d() as usize;if gi>=n as usize{return}
    unsafe{
        let len=*lengths.add(gi) as usize;
        if len>capacity as usize{status.add(gi).write(-3);return}
        let bytes=core::slice::from_raw_parts(input.add(gi*capacity as usize),len);
        match serde_json::from_slice::<Episode>(bytes){
            Ok(e) if e.k as u32==k&&e.game.players.len()==8&&e.game.cfg.num_players==8&&
                e.game.cfg==(&*(*games.add(gi) as *const Episode)).game.cfg&&
                (0..k as usize).all(|j|e.controlled[j] as u32==*seats.add(j))&&e.game.check_loaded().is_ok()=>{
                *(*games.add(gi) as *mut Episode)=e;status.add(gi).write(0)},
            _=>status.add(gi).write(-3),
        }
    }
}

/// Complete natural trajectory reverse recurrence; all input/output buffers are CUDA.
#[kernel]
pub unsafe fn tavern_gae(values:*const f32,rewards:*const f32,offsets:*const u64,n:u32,
    gamma:f64,lambda:f64,advantages:*mut f32,returns:*mut f32){
    let i=thread::index_1d() as usize;if i>=n as usize{return}
    unsafe{
        let start=*offsets.add(i) as usize;let end=*offsets.add(i+1) as usize;
        let mut next_value=0f64;let mut carry=0f64;
        for j in (start..end).rev(){
            let value=*values.add(j) as f64;
            let delta=*rewards.add(j) as f64+gamma*next_value-value;
            carry=delta+gamma*lambda*carry;
            advantages.add(j).write(carry as f32);returns.add(j).write(carry as f32+value as f32);next_value=value;
        }
    }
}

/// ABI regression: opaque data pointers and wide-slice metadata survive calls.
#[inline(never)]
fn abi_slice(input:&[u8])->Result<&[u8],u32>{if input.is_empty(){Err(7)}else{Ok(&input[1..])}}
#[kernel]
pub unsafe fn tavern_abi_test(input:*const u8,n:u32,out:*mut u64){
    if thread::index_1d()!=0{return}
    unsafe{
        let values=core::slice::from_raw_parts(input,n as usize);
        match abi_slice(values){Ok(tail)=>{out.write(tail.len() as u64);out.add(1).write(tail.first().copied().unwrap_or(0) as u64)},Err(code)=>{out.write(u64::MAX);out.add(1).write(code as u64)}}
    }
}

#[cfg(feature="arena")]
#[kernel]
pub unsafe fn tavern_bind_arenas(storage:*mut u8,n:u32,stride:u64,initialize:u32){
    let i=thread::index_1d() as usize;if i>=n as usize{return}
    unsafe{tavern_device_core::arena::bind(storage,stride as usize,initialize!=0)}
}
#[cfg(feature="arena")]
#[kernel]
pub unsafe fn tavern_arena_stats(n:u32,out:*mut u64){
    let i=thread::index_1d() as usize;if i>=n as usize{return}
    unsafe{
        let arena=tavern_device_core::arena::lane();
        let stats=(*arena).stats();core::slice::from_raw_parts_mut(out.add(i*4),4).copy_from_slice(&stats);
    }
}

#[kernel]
pub unsafe fn tavern_layout(out:*mut u64){
    if thread::index_1d()!=0{return}
    unsafe{core::slice::from_raw_parts_mut(out,8).copy_from_slice(&tavern_device_wire::layout())}
}
#[kernel]
pub unsafe fn tavern_validate_import(games:*const u64,parsed:*const u64,n:u32,status:*mut i32,seats:*const u32,k:u32){
    let gi=thread::index_1d() as usize;if gi>=n as usize{return}
    unsafe{
        let ptr=*parsed.add(gi) as *const Episode;
        if ptr.is_null(){status.add(gi).write(-3);return}
        let e=&*ptr;
        let valid=e.k as u32==k&&e.game.players.len()==8&&e.game.cfg.num_players==8&&
            e.game.cfg==(&*(*games.add(gi) as *const Episode)).game.cfg&&
            (0..k as usize).all(|j|e.controlled[j] as u32==*seats.add(j))&&e.game.check_loaded().is_ok();
        status.add(gi).write(if valid{0}else{-3});
    }
}
#[kernel]
pub unsafe fn tavern_commit_import(games:*mut u64,parsed:*mut u64,n:u32){
    let gi=thread::index_1d() as usize;if gi>=n as usize{return}
    unsafe{
        let prior=*games.add(gi) as *mut Episode;let next=*parsed.add(gi);
        assert_ne!(next,0);games.add(gi).write(next);parsed.add(gi).write(0);drop(Box::from_raw(prior));
    }
}

/// One warp per game packs only valid serialized bytes for checkpoint transfer.
#[kernel]
pub unsafe fn tavern_pack_states(input:*const u8,output:*mut u8,lengths:*const u32,offsets:*const u64,n:u32,capacity:u32){
    let i=thread::index_1d() as usize;let gi=i/32;if gi>=n as usize{return}
    unsafe{
        let len=*lengths.add(gi) as usize;let offset=*offsets.add(gi) as usize;
        assert!(len<=capacity as usize);
        let mut j=i%32;while j<len{output.add(offset+j).write(*input.add(gi*capacity as usize+j));j+=32}
    }
}

/// Identifies the explicit compiled frozen-task contract; it never changes game state.
#[kernel]
pub unsafe fn tavern_contract(out:*mut u64){
    if thread::index_1d()!=0{return}
    unsafe{out.write(cfg!(feature="frozen_rules") as u64|((cfg!(feature="ordered_tables") as u64)<<1))}
}

#[cfg(feature="arena")]
#[kernel]
pub unsafe fn tavern_bind_persistent(storage:*mut u8,capacity:u64,initialize:u32){
    unsafe{tavern_device_core::arena::bind_persistent(storage,capacity as usize,initialize!=0)}
}
#[cfg(feature="arena")]
#[kernel]
pub unsafe fn tavern_persistent_stats(out:*mut u64){
    if thread::index_1d()!=0{return}
    unsafe{core::slice::from_raw_parts_mut(out,4).copy_from_slice(&tavern_device_core::arena::persistent_stats())}
}
