// SPDX-License-Identifier: Apache-2.0
// Generic neural graph dispatch only; no simulator or game mechanics.
// CUDA 12.8 SWITCH API and allowed child-graph bodies follow NVIDIA's example:
// https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/
// https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html
#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <new>
#include <vector>
#if CUDART_VERSION < 12080
#error "device_branch requires CUDA 12.8 or newer SWITCH conditional nodes"
#endif
namespace {
constexpr int MAX_BRANCHES=16,SELECTOR_THREADS=256,MAX_ENTITIES=200;
struct Limits {int size=0;int bound[MAX_BRANCHES]{};};
struct Decision {int bucket,maximum,error;};
struct Branch {cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;int device=-1;};

cudaError_t validate(const int* obs,int rows,int64_t stride,int column,
        const int* limits,int count,Limits& out){
    if(!obs||rows<1||stride<1||column<0||column>=stride||!limits||count<1||count>MAX_BRANCHES)return cudaErrorInvalidValue;
    int previous=0;
    for(int i=0;i<count;++i){if(limits[i]<=previous||limits[i]>MAX_ENTITIES)return cudaErrorInvalidValue;out.bound[i]=limits[i];previous=limits[i];}
    if(previous!=MAX_ENTITIES)return cudaErrorInvalidValue;
    out.size=count;return cudaSuccess;
}
__device__ __forceinline__ Decision reduce_counts(const int* obs,int rows,
        int64_t stride,int column,Limits limits){
    __shared__ int maxima[SELECTOR_THREADS],errors[SELECTOR_THREADS];
    int maximum=0,error=0;
    for(int row=threadIdx.x;row<rows;row+=blockDim.x){
        int value=obs[int64_t(row)*stride+column];
        maximum=value>maximum?value:maximum;
        error|=value<0?1:value>MAX_ENTITIES?2:0;
    }
    maxima[threadIdx.x]=maximum;errors[threadIdx.x]=error;__syncthreads();
    for(int offset=SELECTOR_THREADS/2;offset;offset>>=1){
        if(threadIdx.x<offset){
            int other=maxima[threadIdx.x+offset];
            maxima[threadIdx.x]=other>maxima[threadIdx.x]?other:maxima[threadIdx.x];
            errors[threadIdx.x]|=errors[threadIdx.x+offset];
        }
        __syncthreads();
    }
    int bucket=0;while(bucket<limits.size&&maxima[0]>limits.bound[bucket])++bucket;
    return {errors[0]?-1:bucket,maxima[0],errors[0]};
}
__global__ void set_branch(cudaGraphConditionalHandle handle,const int* obs,
        int rows,int64_t stride,int column,Limits limits,int* status){
    Decision value=reduce_counts(obs,rows,stride,column,limits);
    if(threadIdx.x==0){
        status[0]=value.bucket;status[1]=value.maximum;status[2]=value.error;
        // Out-of-range default selects no child even if assertions are disabled.
        cudaGraphSetConditional(handle,value.error?unsigned(limits.size):unsigned(value.bucket));
        assert(value.error==0&&"public entity count must be between zero and 200");
    }
}
__global__ void probe_branch(const int* obs,int rows,int64_t stride,int column,
        Limits limits,int* status){
    Decision value=reduce_counts(obs,rows,stride,column,limits);
    if(threadIdx.x==0){status[0]=value.bucket;status[1]=value.maximum;status[2]=value.error;}
}
// A conditional body may contain these nodes recursively. This bridge only
// accepts unconditional children: another conditional graph cannot be cloned
// into a child node. RNG-free kernels remain a contract of the Python caller.
cudaError_t validate_child(cudaGraph_t graph,int depth=0,int* counts=nullptr){
    if(!graph||depth>64)return cudaErrorInvalidValue;
    size_t count=0;auto code=cudaGraphGetNodes(graph,nullptr,&count);if(code!=cudaSuccess)return code;
    std::vector<cudaGraphNode_t> nodes(count);code=cudaGraphGetNodes(graph,nodes.data(),&count);if(code!=cudaSuccess)return code;
    for(auto node:nodes){
        cudaGraphNodeType type;code=cudaGraphNodeGetType(node,&type);if(code!=cudaSuccess)return code;
        if(counts&&int(type)>=0&&int(type)<32)++counts[int(type)];
        switch(type){
            case cudaGraphNodeTypeKernel:case cudaGraphNodeTypeEmpty:
            case cudaGraphNodeTypeMemcpy:case cudaGraphNodeTypeMemset:break;
            case cudaGraphNodeTypeGraph:{cudaGraph_t nested=nullptr;code=cudaGraphChildGraphNodeGetGraph(node,&nested);if(code!=cudaSuccess)return code;code=validate_child(nested,depth+1,counts);if(code!=cudaSuccess)return code;break;}
            default:return cudaErrorNotSupported;
        }
    }
    return cudaSuccess;
}
cudaError_t release(Branch* branch){
    if(!branch)return cudaSuccess;
    auto result=cudaSuccess;
    if(branch->executable){auto code=cudaGraphExecDestroy(branch->executable);if(result==cudaSuccess)result=code;}
    if(branch->graph){auto code=cudaGraphDestroy(branch->graph);if(result==cudaSuccess)result=code;}
    delete branch;return result;
}
void CUDART_CB never_run(void*){}
} // namespace
extern "C" {
int cn_branch_runtime_version(){int version=0;return cudaRuntimeGetVersion(&version)==cudaSuccess?version:0;}
const char* cn_branch_error_string(int code){return cudaGetErrorString(cudaError_t(code));}
int cn_branch_validate_child(void* graph,int* counts,int capacity){
    if(!counts||capacity<32)return int(cudaErrorInvalidValue);
    for(int i=0;i<32;++i)counts[i]=0;
    return int(validate_child(static_cast<cudaGraph_t>(graph),0,counts));
}
// Qualification constructs forbidden nodes but never launches them.
int cn_branch_test_unsupported_children(){
    cudaGraph_t host_graph=nullptr,parent_graph=nullptr;
    auto finish=[&](cudaError_t result){if(parent_graph)cudaGraphDestroy(parent_graph);if(host_graph)cudaGraphDestroy(host_graph);return int(result);};
    auto code=cudaGraphCreate(&host_graph,0);if(code!=cudaSuccess)return finish(code);
    cudaHostNodeParams params{};params.fn=never_run;
    cudaGraphNode_t node;
    code=cudaGraphAddHostNode(&node,host_graph,nullptr,0,&params);if(code!=cudaSuccess)return finish(code);
    if(validate_child(host_graph)!=cudaErrorNotSupported)return finish(cudaErrorUnknown);
    code=cudaGraphCreate(&parent_graph,0);if(code!=cudaSuccess)return finish(code);
    code=cudaGraphAddChildGraphNode(&node,parent_graph,nullptr,0,host_graph);if(code!=cudaSuccess)return finish(code);
    if(validate_child(parent_graph)!=cudaErrorNotSupported)return finish(cudaErrorUnknown);
    return finish(cudaSuccess);
}
int cn_branch_create(const int* observations,int rows,int64_t stride,int entity_column,
        const int* limits,void*const* child_graphs,int count,int* status,void** result){
    if(!result)return int(cudaErrorInvalidValue);*result=nullptr;
    Limits bounds;auto code=validate(observations,rows,stride,entity_column,limits,count,bounds);
    if(code!=cudaSuccess||!child_graphs||!status)return int(code!=cudaSuccess?code:cudaErrorInvalidValue);
    for(int i=0;i<count;++i){code=validate_child(static_cast<cudaGraph_t>(child_graphs[i]));if(code!=cudaSuccess)return int(code);}
    auto* branch=new(std::nothrow) Branch;if(!branch)return int(cudaErrorMemoryAllocation);
    code=cudaGetDevice(&branch->device);if(code!=cudaSuccess){release(branch);return int(code);}
    code=cudaGraphCreate(&branch->graph,0);if(code!=cudaSuccess){release(branch);return int(code);}
    cudaGraphConditionalHandle handle;
    code=cudaGraphConditionalHandleCreate(&handle,branch->graph,unsigned(count),cudaGraphCondAssignDefault);
    if(code!=cudaSuccess){release(branch);return int(code);}
    void* arguments[]={&handle,&observations,&rows,&stride,&entity_column,&bounds,&status};
    cudaGraphNodeParams selector{};selector.type=cudaGraphNodeTypeKernel;
    selector.kernel.func=reinterpret_cast<void*>(set_branch);
    selector.kernel.gridDim=dim3(1);selector.kernel.blockDim=dim3(SELECTOR_THREADS);
    selector.kernel.kernelParams=arguments;
    cudaGraphNode_t selector_node,conditional_node;
    code=cudaGraphAddNode(&selector_node,branch->graph,nullptr,0,&selector);
    if(code!=cudaSuccess){release(branch);return int(code);}
    cudaGraphNodeParams conditional{};conditional.type=cudaGraphNodeTypeConditional;
    conditional.conditional.handle=handle;conditional.conditional.type=cudaGraphCondTypeSwitch;conditional.conditional.size=unsigned(count);
    code=cudaGraphAddNode(&conditional_node,branch->graph,&selector_node,1,&conditional);
    if(code!=cudaSuccess){release(branch);return int(code);}
    for(int i=0;i<count;++i){
        cudaGraphNode_t child_node;
        code=cudaGraphAddChildGraphNode(&child_node,conditional.conditional.phGraph_out[i],nullptr,0,static_cast<cudaGraph_t>(child_graphs[i]));
        if(code!=cudaSuccess){release(branch);return int(code);}
    }
    code=cudaGraphInstantiateWithFlags(&branch->executable,branch->graph,0);
    if(code!=cudaSuccess){release(branch);return int(code);}
    *result=branch;return int(cudaSuccess);
}
int cn_branch_launch(void* opaque,void* stream){
    auto* branch=static_cast<Branch*>(opaque);if(!branch)return int(cudaErrorInvalidValue);
    int device=-1;auto code=cudaGetDevice(&device);if(code!=cudaSuccess)return int(code);
    if(device!=branch->device)return int(cudaErrorInvalidDevice);
    cudaStreamCaptureStatus capturing;
    code=cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&capturing);if(code!=cudaSuccess)return int(code);
    if(capturing!=cudaStreamCaptureStatusNone)return int(cudaErrorStreamCaptureUnsupported);
    return int(cudaGraphLaunch(branch->executable,static_cast<cudaStream_t>(stream)));
}
int cn_branch_probe(const int* observations,int rows,int64_t stride,int entity_column,
        const int* limits,int count,int* status,void* stream){
    Limits bounds;auto code=validate(observations,rows,stride,entity_column,limits,count,bounds);
    if(code!=cudaSuccess||!status)return int(code!=cudaSuccess?code:cudaErrorInvalidValue);
    probe_branch<<<1,SELECTOR_THREADS,0,static_cast<cudaStream_t>(stream)>>>(observations,rows,stride,entity_column,bounds,status);
    return int(cudaGetLastError());
}
// Caller must order all launches before destruction; Python close synchronizes
// the owning device before releasing the executable and retained Torch children.
int cn_branch_destroy(void* opaque){return int(release(static_cast<Branch*>(opaque)));}
}
