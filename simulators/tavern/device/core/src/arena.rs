//! Per-game reusable size-class storage. CUDA spills preserve complete mechanics.
//! One writer per lane, one ordered stream; no allocator locks on the common path.
use core::{alloc::Layout,ptr};
const CLASSES:usize=24;
const HEADER_BYTES:usize=32;
const MAGIC:u32=0x47505541;
#[repr(C)]
struct Header{owner:*mut Arena,next:usize,class:u32,magic:u32,raw:*mut u8}
#[repr(C)]
pub struct Arena{
    base:usize,capacity:usize,bump:usize,live:usize,peak:usize,allocations:u64,spills:u64,
    free:[usize;CLASSES],
}
impl Arena{
    pub const fn new(base:usize,capacity:usize)->Self{Self{base,capacity,bump:0,live:0,peak:0,allocations:0,spills:0,free:[0;CLASSES]}}
    pub unsafe fn allocate(&mut self,layout:Layout)->Option<*mut u8>{
        if layout.align()>32{return None}
        let size=layout.size().max(1).checked_add(HEADER_BYTES)?.max(64).checked_next_power_of_two()?;
        let class=size.trailing_zeros() as usize;
        if class>=CLASSES{return None}
        let block=if self.free[class]!=0{
            let value=self.free[class];self.free[class]=unsafe{(*(value as *const Header)).next};value
        }else{
            let end=self.bump.checked_add(size)?;
            if end>self.capacity{return None}
            let value=self.base+self.bump;self.bump=end;value
        };
        unsafe{(block as *mut Header).write(Header{owner:self,next:0,class:class as u32,magic:MAGIC,raw:ptr::null_mut()})}
        self.live+=size;self.peak=self.peak.max(self.live);self.allocations+=1;
        Some((block+HEADER_BYTES) as *mut u8)
    }
    pub unsafe fn release(pointer:*mut u8){
        let h=unsafe{&mut *(pointer.sub(HEADER_BYTES) as *mut Header)};
        assert_eq!(h.magic,MAGIC);assert!(!h.owner.is_null());
        let arena=unsafe{&mut *h.owner};let class=h.class as usize;
        h.next=arena.free[class];arena.free[class]=h as *mut Header as usize;arena.live-=1usize<<class;
    }
    pub fn stats(&self)->[u64;4]{[self.bump as u64,self.peak as u64,self.allocations,self.spills]}
}

#[cfg(target_arch="nvptx64")]
mod cuda{
    use super::*;
    use core::{alloc::GlobalAlloc,cell::UnsafeCell};
    use crate::device_thread::{ThreadBool,MAX_DEVICE_GAMES};
    struct Bindings(UnsafeCell<[usize;MAX_DEVICE_GAMES]>);
    unsafe impl Sync for Bindings{}
    static BINDINGS:Bindings=Bindings(UnsafeCell::new([0;MAX_DEVICE_GAMES]));
    struct PersistentArena(UnsafeCell<usize>);
    unsafe impl Sync for PersistentArena{}
    static SHARED:PersistentArena=PersistentArena(UnsafeCell::new(0));
    struct Mutex(UnsafeCell<u32>);
    unsafe impl Sync for Mutex{}
    static SHARED_LOCK:Mutex=Mutex(UnsafeCell::new(0));
    unsafe fn lock_shared(){
        let address=SHARED_LOCK.0.get() as u64;
        loop{
            let old:u32;
            unsafe{core::arch::asm!("atom.acq_rel.gpu.global.cas.b32 {old}, [{address}], 0, 1;",
                old=out(reg32)old,address=in(reg64)address,options(nostack))}
            if old==0{return}
            core::hint::spin_loop();
        }
    }
    unsafe fn unlock_shared(){
        let address=SHARED_LOCK.0.get() as u64;
        unsafe{core::arch::asm!("st.release.gpu.global.u32 [{address}], 0;",address=in(reg64)address,options(nostack))}
    }
    pub unsafe fn bind_persistent(region:*mut u8,capacity:usize,initialize:bool){
        if cuda_std::thread::index_1d()!=0{return}
        let arena=region as *mut Arena;
        if initialize{
            let data=(region as usize+core::mem::size_of::<Arena>()+31)/32*32;
            unsafe{arena.write(Arena::new(data,capacity-(data-region as usize)))}
        }
        unsafe{*SHARED.0.get()=arena as usize}
    }
    pub unsafe fn persistent_stats()->[u64;4]{
        let ptr=unsafe{*SHARED.0.get()} as *mut Arena;
        if ptr.is_null(){[0;4]}else{unsafe{(*ptr).stats()}}
    }
    static PERSISTENT:ThreadBool=ThreadBool::new();
    pub struct PersistentScope(bool);
    impl PersistentScope{
        pub fn new()->Self{Self(PERSISTENT.with(|v|v.replace(true)))}
    }
    impl Drop for PersistentScope{fn drop(&mut self){PERSISTENT.with(|v|v.set(self.0))}}
    pub unsafe fn bind(base:*mut u8,stride:usize,initialize:bool){
        let i=cuda_std::thread::index_1d() as usize;
        let region=unsafe{base.add(i*stride)};let arena=region as *mut Arena;
        if initialize{
            let data=((region as usize+core::mem::size_of::<Arena>()+31)/32)*32;
            unsafe{arena.write(Arena::new(data,stride-(data-region as usize)))}
        }
        unsafe{(*BINDINGS.0.get())[i]=arena as usize}
    }
    pub unsafe fn lane()->*mut Arena{
        let i=cuda_std::thread::index_1d() as usize;
        unsafe{(*BINDINGS.0.get())[i] as *mut Arena}
    }
    struct DeviceAllocator;
    #[global_allocator]
    static ALLOCATOR:DeviceAllocator=DeviceAllocator;
    unsafe impl GlobalAlloc for DeviceAllocator{
        unsafe fn alloc(&self,layout:Layout)->*mut u8{
            let persistent=PERSISTENT.with(|v|v.get());
            let arena=if persistent{unsafe{*SHARED.0.get() as *mut Arena}}else{unsafe{lane()}};
            if !arena.is_null(){
                if persistent{unsafe{lock_shared()}}
                let result=unsafe{(*arena).allocate(layout)};
                if result.is_none(){unsafe{(*arena).spills+=1}}
                if persistent{unsafe{unlock_shared()}}
                if let Some(p)=result{return p}
            }
            // Persistent once values never point into a game's reclaimable storage.
            // Rare capacity/alignment spills use CUDA malloc, never truncate a game.
            let Some(size)=layout.size().max(1).checked_add(HEADER_BYTES).and_then(|x|x.checked_add(layout.align())) else{return ptr::null_mut()};
            let raw=unsafe{cuda_std::mem::malloc(size)} as *mut u8;
            if raw.is_null(){return raw}
            let data=((raw as usize+HEADER_BYTES+layout.align()-1)/layout.align())*layout.align();
            unsafe{((data-HEADER_BYTES) as *mut Header).write(Header{owner:ptr::null_mut(),next:0,class:0,magic:MAGIC,raw})}
            data as *mut u8
        }
        unsafe fn dealloc(&self,pointer:*mut u8,_:Layout){
            let h=unsafe{&*(pointer.sub(HEADER_BYTES) as *const Header)};
            assert_eq!(h.magic,MAGIC);
            if h.owner.is_null(){unsafe{cuda_std::mem::free(h.raw.cast())}}
            else{
                let shared=h.owner as usize==unsafe{*SHARED.0.get()};
                if shared{unsafe{lock_shared()}}
                unsafe{Arena::release(pointer)}
                if shared{unsafe{unlock_shared()}}
            }
        }
    }
}
#[cfg(target_arch="nvptx64")]
pub use cuda::{PersistentScope,bind,lane,bind_persistent,persistent_stats};

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn reused_allocations_preserve_alignment_contents_and_capacity(){
        let mut backing=alloc::vec![0u8;1<<20];let base=(backing.as_mut_ptr() as usize+31)/32*32;
        let mut arena=alloc::boxed::Box::new(Arena::new(base,backing.len()-32));
        let mut live:alloc::vec::Vec<(*mut u8,usize,u8)>=alloc::vec::Vec::new();let mut rng=12345u64;
        for step in 0..10000{
            rng=rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if !live.is_empty()&&(rng%3==0||live.len()>100){
                let index=(rng as usize)%live.len();let(p,len,value)=live.swap_remove(index);
                unsafe{for j in 0..len{assert_eq!(*p.add(j),value)}Arena::release(p)}
            }else{
                let size=(rng as usize%4095)+1;let align=1usize<<((rng>>32)%6);
                if let Some(p)=unsafe{arena.allocate(Layout::from_size_align(size,align).unwrap())}{
                    assert_eq!(p as usize%align,0);let value=step as u8;
                    unsafe{p.write_bytes(value,size)};live.push((p,size,value));
                }
            }
        }
        for(p,len,value)in live{unsafe{for j in 0..len{assert_eq!(*p.add(j),value)}Arena::release(p)}}
        assert_eq!(arena.live,0);let bump=arena.bump;
        for _ in 0..1000{let p=unsafe{arena.allocate(Layout::new::<u64>()).unwrap()};unsafe{Arena::release(p)}}
        assert!(arena.bump<=bump+64);
    }
}
