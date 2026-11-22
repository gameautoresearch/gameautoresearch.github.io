use cuda_builder::{CudaBuilder,NvvmArch};
fn main(){
    let root=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    CudaBuilder::new(root.join("kernels")).arch(NvvmArch::Compute100)
        .fma_contraction(false)
        .copy_to(root.join("tavern.ptx")).build().expect("compile complete device rule engine");
}
