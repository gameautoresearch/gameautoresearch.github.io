//! One ABI/type definition shared by independently compiled simulation and state modules.
#![cfg_attr(target_arch="nvptx64",no_std)]
#[repr(C)]
#[derive(serde::Serialize,serde::Deserialize)]
pub struct Episode{
    pub game:tavern_device_core::Game,pub index:u64,pub seed:u64,pub finished:[bool;8],
    pub controlled:[u8;8],pub k:u8,
}
pub const ABI_VERSION:u64=1;
pub fn layout()->[u64;8]{[
    ABI_VERSION,core::mem::size_of::<Episode>() as u64,core::mem::align_of::<Episode>() as u64,
    core::mem::size_of::<tavern_device_core::Game>() as u64,
    core::mem::offset_of!(Episode,game) as u64,core::mem::offset_of!(Episode,index) as u64,
    core::mem::offset_of!(Episode,controlled) as u64,core::mem::offset_of!(Episode,k) as u64,
]}
