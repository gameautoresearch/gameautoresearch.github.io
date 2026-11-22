//! Runtime diagnostic only: never a substitute for complete-game qualification.
#![no_std]
extern crate alloc;
use cuda_std::{kernel,thread};
use alloc::vec::Vec;
static PER_LANE:[tavern_device_core::sync::OnceLock<u64>;64]=[const{tavern_device_core::sync::OnceLock::new()};64];
static ONCE:tavern_device_core::sync::OnceLock<u64>=tavern_device_core::sync::OnceLock::new();
#[inline(never)]
fn tail(input:&[u8])->Result<&[u8],u32>{if input.is_empty(){Err(7)}else{Ok(&input[1..])}}
#[kernel]
pub unsafe fn tavern_contract(out:*mut u64){if thread::index_1d()==0{unsafe{out.write(0)}}}
#[kernel]
pub unsafe fn native_primitives(out:*mut u64,n:u32){
    let i=thread::index_1d() as usize;if i>=n as usize{return}
    let bytes:Vec<u8>=(0..64).map(|j|((i+j)%251) as u8).collect();
    let slice=tail(&bytes).unwrap();
    let high=unsafe{cuda_std::intrinsics::umul64hi((0xfedcba9876543210u64^i as u64) as i64,0xa1b2c3d4e5f60718u64 as i64)} as u64;
    let once=*ONCE.get_or_init(||0x123456789abcdef0);
    let unique=*PER_LANE[i].get_or_init(||{
        let temp:Vec<u64>=(0..32).map(|j|j+i as u64).collect();
        temp.iter().sum()
    });
    let row=[unique-496-32*i as u64+i as u64,slice.len() as u64,slice[0] as u64,bytes.iter().map(|&x|x as u64).sum(),high,once];
    unsafe{core::slice::from_raw_parts_mut(out.add(i*6),6).copy_from_slice(&row)}
}

#[kernel]
pub unsafe fn tavern_bind_persistent(storage:*mut u8,capacity:u64,initialize:u32){
    unsafe{tavern_device_core::arena::bind_persistent(storage,capacity as usize,initialize!=0)}
}

#[kernel]
pub unsafe fn native_panic_test(){if thread::index_1d()==0{panic!("explicit runtime diagnostic")}}
