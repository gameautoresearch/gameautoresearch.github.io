//! Device-owned immutable once values, published after complete construction.
//! No spinning/hostcalls: duplicate racing constructors lose a CAS and free their value.
use alloc::boxed::Box;
use core::{cell::UnsafeCell,marker::PhantomData};

/// A table initialized on a single thread before any game kernel on the owning
/// ordered stream. Ordinary global loads suffice across the kernel boundary;
/// constructors are unreachable from getters and no read-only-cache asm is used.
pub struct Prewarmed<T>{ptr:UnsafeCell<*const T>}
unsafe impl<T:Send+Sync> Sync for Prewarmed<T>{}
impl<T> Prewarmed<T>{
    pub const fn new()->Self{Self{ptr:UnsafeCell::new(core::ptr::null())}}
    #[inline(always)]
    pub fn get(&self)->&T{
        let ptr=unsafe{self.ptr.get().read()};
        assert!(!ptr.is_null(),"table used before ordered prewarm");
        unsafe{&*ptr}
    }
    pub unsafe fn initialize(&self,value:&'static T){
        assert_eq!(cuda_std::thread::index_1d(),0);
        let old=unsafe{self.ptr.get().read()};
        assert!(old.is_null()||core::ptr::eq(old,value));
        unsafe{self.ptr.get().write(value)}
    }
}

/// An immutable singleton reference published only by the one-thread prewarm
/// kernel. Its owner is retained for the module lifetime. All readers run after
/// prewarm on the same ordered CUDA stream; no reader races the publication.
pub struct Published<T>{ptr:UnsafeCell<u64>,marker:PhantomData<T>}
unsafe impl<T:Send+Sync> Sync for Published<T>{}
impl<T> Published<T>{
    pub const fn new()->Self{Self{ptr:UnsafeCell::new(0),marker:PhantomData}}
    #[inline(always)]
    pub fn get(&self)->Option<&T>{
        let value:u64;
        unsafe{core::arch::asm!("ld.global.nc.u64 {}, [{}];",out(reg64)value,
            in(reg64)self.ptr.get() as u64,options(readonly,nostack))}
        if value==0{None}else{Some(unsafe{&*(value as *const T)})}
    }
    /// Requires the owning ordered stream and its single-thread prewarm phase.
    pub unsafe fn publish(&self,value:&'static T){
        assert_eq!(cuda_std::thread::index_1d(),0);
        let pointer=value as *const T as u64;
        let old=unsafe{*self.ptr.get()};assert!(old==0||old==pointer);
        // A published pointer never changes. A cached earlier zero only takes
        // the existing acquire/once fallback, preserving correctness.
        if old==0{unsafe{self.ptr.get().write(pointer)}}
    }
}

pub struct OnceLock<T> { ptr:UnsafeCell<u64>, marker:PhantomData<T> }
unsafe impl<T:Send+Sync> Sync for OnceLock<T> {}
impl<T> OnceLock<T> {
    pub const fn new()->Self {Self{ptr:UnsafeCell::new(0),marker:PhantomData}}
    #[inline(always)]
    pub fn get_or_init(&self,init:impl FnOnce()->T)->&T {
        unsafe {
            let ready:u64;
            core::arch::asm!("ld.acquire.gpu.global.u64 {}, [{}];",
                out(reg64) ready,in(reg64) self.ptr.get() as u64);
            if ready!=0{return &*(ready as *const T)}
            self.initialize(init)
        }
    }
    // Keep allocation, racing construction and publication out of every hot
    // database/script/card-id read. The acquire/release protocol is unchanged.
    #[cold]
    #[inline(never)]
    unsafe fn initialize(&self,init:impl FnOnce()->T)->&T {
        unsafe {
            #[cfg(feature="arena")]
            let _persistent=crate::arena::PersistentScope::new();
            let value=Box::into_raw(Box::new(init()));
            let prior:u64;
            core::arch::asm!("atom.acq_rel.gpu.global.cas.b64 {}, [{}], {}, {};",
                out(reg64) prior,in(reg64) self.ptr.get() as u64,in(reg64) 0u64,in(reg64) value as u64);
            if prior==0 {&*value} else {drop(Box::from_raw(value));&*(prior as *const T)}
        }
    }
}
