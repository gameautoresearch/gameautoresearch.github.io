use cuda_builder::{CudaBuilder,NvvmArch};
fn main(){
    let root=std::path::PathBuf::from(std::env::args_os().nth(1).expect("complete rules workspace path required"));
    let split=std::env::var("TAVERN_DEVICE_SPLIT").as_deref()==Ok("1");
    let modules=if split{vec![("kernels","tavern","main"),("state-kernels","tavern-state","state")]}else{vec![("kernels","tavern","main")]};
    for (source,output,tag_suffix) in modules{
        let mut builder=CudaBuilder::new(root.join(source)).arch(NvvmArch::Compute100)
            .fma_contraction(false).copy_to(root.join(format!("{output}.ptx")));
        let mut args=Vec::<String>::new();
        if split{args.push("--no-default-features".into())}
        if std::env::var("TAVERN_DEVICE_ARENA").as_deref()==Ok("1"){args.push("--features=arena".into())}
        if let Ok(level)=std::env::var("TAVERN_DEVICE_RUST_OPT"){
            assert!(["0","1","2","3"].contains(&level.as_str()));
            args.push(format!("--config=profile.release.opt-level={level}"));
            args.push("--config=profile.release.lto=false".into());
        }
        if std::env::var("TAVERN_DEVICE_NVVM_OPT").as_deref()==Ok("0"){builder=builder.nvvm_opts(false)}
        if let Ok(tag)=std::env::var("TAVERN_DEVICE_IR_TAG"){
            let path=std::path::PathBuf::from(format!("/compiler-cache/{tag}-{tag_suffix}.ll"));
            assert!(!path.exists(),"diagnostic IR must have a unique attempt path");
            builder=builder.final_module_path(path);
        }
        builder.build_args(&args).build().expect("compile complete device engine module");
    }
}
