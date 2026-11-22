//! GPU checkpoint serialization/parsing, separated from hot complete-rule simulation.
//! Parsing never commits: the simulation module validates source/card/rule/state/seat
//! invariants before replacing any running episode. No CPU simulation is involved.
#![cfg_attr(target_arch="nvptx64",no_std)]
extern crate alloc;
use alloc::boxed::Box;
use cuda_std::{kernel,thread};
use tavern_device_wire::Episode;
#[kernel]
pub unsafe fn tavern_state_prewarm(){
    if thread::index_1d()!=0{return}
    #[cfg(target_arch="nvptx64")]{
        tavern_device_core::cards::prewarm_device_db();
    }
}
#[kernel]
pub unsafe fn tavern_state_layout(out:*mut u64){
    if thread::index_1d()!=0{return}
    unsafe{core::slice::from_raw_parts_mut(out,8).copy_from_slice(&tavern_device_wire::layout())}
}
#[kernel]
pub unsafe fn tavern_state_export(games:*const u64,n:u32,out:*mut u8,capacity:u32,lengths:*mut u32){
    let gi=thread::index_1d() as usize;if gi>=n as usize{return}
    unsafe{
        let e=&*(*games.add(gi) as *const Episode);
        let text=serde_json::to_vec(e).expect("state export");lengths.add(gi).write(text.len() as u32);
        if text.len()<=capacity as usize{core::slice::from_raw_parts_mut(out.add(gi*capacity as usize),text.len()).copy_from_slice(&text)}
    }
}
#[kernel]
pub unsafe fn tavern_state_parse(parsed:*mut u64,n:u32,input:*const u8,capacity:u32,lengths:*const u32,status:*mut i32){
    let gi=thread::index_1d() as usize;if gi>=n as usize{return}
    unsafe{
        parsed.add(gi).write(0);let len=*lengths.add(gi) as usize;
        if len>capacity as usize{status.add(gi).write(-3);return}
        let bytes=core::slice::from_raw_parts(input.add(gi*capacity as usize),len);
        match serde_json::from_slice::<Episode>(bytes){
            Ok(e)=>{parsed.add(gi).write(Box::into_raw(Box::new(e)) as u64);status.add(gi).write(0)},
            Err(_)=>status.add(gi).write(-3),
        }
    }
}
#[kernel]
pub unsafe fn tavern_state_destroy(parsed:*mut u64,n:u32){
    let gi=thread::index_1d() as usize;if gi>=n as usize{return}
    unsafe{let p=*parsed.add(gi) as *mut Episode;if !p.is_null(){drop(Box::from_raw(p));parsed.add(gi).write(0)}}
}
#[cfg(feature="arena")]
#[kernel]
pub unsafe fn tavern_state_bind_arenas(storage:*mut u8,n:u32,stride:u64,initialize:u32){
    let gi=thread::index_1d() as usize;if gi>=n as usize{return}
    unsafe{tavern_device_core::arena::bind(storage,stride as usize,initialize!=0)}
}

/// One warp per game packs only valid serialized bytes for checkpoint transfer.
#[kernel]
pub unsafe fn tavern_state_pack(input:*const u8,output:*mut u8,lengths:*const u32,offsets:*const u64,n:u32,capacity:u32){
    let i=thread::index_1d() as usize;let gi=i/32;if gi>=n as usize{return}
    unsafe{
        let len=*lengths.add(gi) as usize;let offset=*offsets.add(gi) as usize;
        assert!(len<=capacity as usize);
        let mut j=i%32;while j<len{output.add(offset+j).write(*input.add(gi*capacity as usize+j));j+=32}
    }
}

#[cfg(feature="arena")]
#[kernel]
pub unsafe fn tavern_state_bind_persistent(storage:*mut u8,capacity:u64,initialize:u32){
    unsafe{tavern_device_core::arena::bind_persistent(storage,capacity as usize,initialize!=0)}
}
#[cfg(feature="arena")]
#[kernel]
pub unsafe fn tavern_state_persistent_stats(out:*mut u64){
    if thread::index_1d()!=0{return}
    unsafe{core::slice::from_raw_parts_mut(out,4).copy_from_slice(&tavern_device_core::arena::persistent_stats())}
}
