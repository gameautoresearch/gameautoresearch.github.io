//! Minimal standard-NVPTX launch/allocation primitives; contains no game rules.
#![no_std]
#![cfg_attr(target_arch="nvptx64",feature(asm_experimental_arch,alloc_error_handler,stdarch_nvptx))]
pub use tavern_native_macros::kernel;
pub mod thread{
    #[inline(always)]
    pub fn index_1d()->u32{
        #[cfg(target_arch="nvptx64")]
        unsafe{
            // LLVM must see its recognized divergence-source intrinsic. Opaque
            // inline assembly does not communicate that tid.x varies by lane.
            core::arch::nvptx::_block_idx_x()*core::arch::nvptx::_block_dim_x()
                +core::arch::nvptx::_thread_idx_x()
        }
        #[cfg(not(target_arch="nvptx64"))]
        panic!("GPU thread index requires NVPTX")
    }
}
pub mod mem{
    unsafe extern "C"{
        pub fn malloc(size:usize)->*mut core::ffi::c_void;
        pub fn free(pointer:*mut core::ffi::c_void);
    }
}
pub mod intrinsics{
    #[inline(always)]
    pub unsafe fn umul64hi(a:i64,b:i64)->i64{
        #[cfg(target_arch="nvptx64")]
        unsafe{
            let high:u64;
            core::arch::asm!("mul.hi.u64 {out}, {a}, {b};",out=out(reg64)high,
                a=in(reg64)a as u64,b=in(reg64)b as u64,options(nomem,nostack,pure));
            high as i64
        }
        #[cfg(not(target_arch="nvptx64"))]
        {(((a as u64 as u128)*(b as u64 as u128))>>64) as u64 as i64}
    }
}
#[cfg(target_arch="nvptx64")]
unsafe extern "C"{
    fn __assertfail(message:*const u8,file:*const u8,line:u32,function:*const u8,char_size:usize)->!;
}
#[cfg(target_arch="nvptx64")]
#[panic_handler]
fn panic(info:&core::panic::PanicInfo)->!{
    let mut file=[0u8;256];let mut line=0;
    if let Some(location)=info.location(){
        let bytes=location.file().as_bytes();let n=bytes.len().min(255);
        file[..n].copy_from_slice(&bytes[bytes.len()-n..]);line=location.line();
    }
    unsafe{__assertfail(b"Rust GPU panic\0".as_ptr(),file.as_ptr(),line,b"native Rust kernel\0".as_ptr(),1)}
}
#[cfg(target_arch="nvptx64")]
#[alloc_error_handler]
fn oom(_layout:core::alloc::Layout)->!{unsafe{core::arch::asm!("trap;",options(noreturn))}}
