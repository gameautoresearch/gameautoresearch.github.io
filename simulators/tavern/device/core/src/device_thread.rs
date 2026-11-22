//! Per-game reentrancy cells for one-writer-per-game device kernels.
//! Kernel index is the immutable game lane; operations on an episode use one stream.
use core::cell::{Cell,UnsafeCell};
pub const MAX_DEVICE_GAMES:usize=16384;
pub struct ThreadBool{cells:UnsafeCell<[Cell<bool>;MAX_DEVICE_GAMES]>}
unsafe impl Sync for ThreadBool{}
impl ThreadBool{
    pub const fn new()->Self{Self{cells:UnsafeCell::new([const{Cell::new(false)};MAX_DEVICE_GAMES])}}
    pub fn with<R>(&self,f:impl FnOnce(&Cell<bool>)->R)->R{
        let i=cuda_std::thread::index_1d() as usize;
        assert!(i<MAX_DEVICE_GAMES);
        // Each kernel has exactly one writer per game index, and calls are stream-ordered.
        unsafe{f(&(*self.cells.get())[i])}
    }
}
