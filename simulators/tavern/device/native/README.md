# Standard LLVM NVPTX compiler path (experimental)

These manifests point directly at the complete shared core, state ABI and kernel
sources; there is no second rule implementation. The runtime supplies only kernel
ABI, thread indexing, unsigned high multiply, CUDA malloc/free and trap handlers.
No Rust-CUDA custom codegen backend or libNVVM compiler is used. Actual CUDA oracle,
checkpoint, hard-mechanics and neural parity gates remain mandatory.
