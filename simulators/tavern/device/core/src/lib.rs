#![cfg_attr(target_arch="nvptx64",no_std)]
#![cfg_attr(target_arch="nvptx64",feature(asm_experimental_arch))]
#![recursion_limit="256"]
extern crate alloc;
#[cfg(target_arch="nvptx64")]
extern crate self as std;
#[cfg(target_arch="nvptx64")]
pub mod collections { pub use crate::maps::HashMap; pub use alloc::collections::{BTreeMap,BTreeSet}; }
#[cfg(target_arch="nvptx64")]
pub mod sync { pub use alloc::rc::Rc as Arc; pub use crate::device_once::OnceLock; }
#[cfg(target_arch="nvptx64")]
pub use core::{fmt,hash,mem,ops,iter,cmp,hint,cell,slice};
#[cfg(target_arch="nvptx64")]
mod device_once;
pub mod maps;
pub mod sorting;
#[cfg(feature="arena")]
pub mod arena;
#[cfg(target_arch="nvptx64")]
pub mod device_thread;
pub mod actions;
pub mod bot;
pub mod cards;
pub mod combat;
pub mod config;
pub mod content;
#[cfg(not(target_arch="nvptx64"))]
pub mod debug;
pub mod engine;
pub mod event;
pub mod experimental;
pub mod flow;
#[cfg(not(target_arch="nvptx64"))]
pub mod guard;
#[cfg(target_arch="nvptx64")]
#[path="device_guard.rs"]
pub mod guard;
#[cfg(not(target_arch="nvptx64"))]
pub mod invariants;
pub mod obs;
pub mod recruit;
pub mod rng;
pub mod script;
pub mod state;
pub mod types;
pub const VERSION:&str="0.1.0-device-port";
pub const GAME_VERSION:&str="37.0.2.256589";
pub const SERVER_RULES_VERSION:&str="37.0.2";
pub use engine::Game;
pub use config::RulesConfig;
pub use actions::{Action,N_ACTIONS};
