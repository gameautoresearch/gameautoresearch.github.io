// SPDX-License-Identifier: Apache-2.0
// Packed PPO graph dispatch: one SWITCH over entity-bound x active-head-capacity.
// Based on the qualified device_branch.cu graph ownership/validation implementation.
// No simulator mechanics or sampling; branch bodies must be RNG-free.
// CUDA 12.8 SWITCH API and allowed child-graph bodies follow NVIDIA's example:
// https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/
// https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html
#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <new>
#include <vector>
#if CUDART_VERSION < 12080
#error "packed_head_branch requires CUDA 12.8 or newer SWITCH conditional nodes"
#endif
namespace {
constexpr int MAX_BRANCHES=16,SELECTOR_THREADS=256;
struct Limits {int entities=0,heads=0,entity[4]{},head[4]{};};
struct Decision {int bucket,active,maximum,entity,head,error;};
struct Branch {
 cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;int device=-1;
 int* activeFlags=nullptr;const int* obs=nullptr;const uint8_t* masks=nullptr;
 int rows=0,column=0,actions=0;int64_t obsStride=0,maskStride=0;Limits bounds{};
};
cudaError_t validate(const int* obs,const uint8_t* masks,int rows,int64_t obs_stride,int64_t mask_stride,int column,int actions,const int* entity,int ne,const int* head,int nh,Limits& out){
 if(!obs||!masks||rows<1||actions<2||obs_stride<1||column<0||column>=obs_stride||mask_stride<actions||!entity||!head||ne<1||ne>4||nh<1||nh>4||ne*nh>MAX_BRANCHES)return cudaErrorInvalidValue;
 int previous=0;for(int i=0;i<ne;++i){if(entity[i]<=previous||entity[i]>200)return cudaErrorInvalidValue;out.entity[i]=entity[i];previous=entity[i];}
 if(previous!=200)return cudaErrorInvalidValue;
 previous=0;for(int i=0;i<nh;++i){if(head[i]<=previous||head[i]>rows)return cudaErrorInvalidValue;out.head[i]=head[i];previous=head[i];}
 if(previous!=rows)return cudaErrorInvalidValue;out.entities=ne;out.heads=nh;return cudaSuccess;
}
// One warp per row: coalesced 32-column reads and warp-uniform early exit.
// Every valid row writes one flag on each launch; scratch is owned by Branch.
__global__ void prepare_active(const uint8_t* masks,int rows,int64_t stride,int actions,int* flags){
 int lane=threadIdx.x&31,row=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 if(row>=rows)return;const uint8_t* mask=masks+int64_t(row)*stride;
 unsigned found=__ballot_sync(0xffffffffu,lane==0&&mask[0]==0);
 for(int start=1;start<actions&&found==0;start+=32){int col=start+lane;found=__ballot_sync(0xffffffffu,col<actions&&mask[col]!=0);}
 if(lane==0)flags[row]=int(found!=0);
}
__device__ __forceinline__ Decision reduce_counts(const int* obs,const int* activeFlags,int rows,int64_t obs_stride,int column,Limits limits){
 __shared__ int maxima[SELECTOR_THREADS],counts[SELECTOR_THREADS],errors[SELECTOR_THREADS];
 int maximum=0,count=0,error=0;
 for(int row=threadIdx.x;row<rows;row+=blockDim.x){int n=obs[int64_t(row)*obs_stride+column];maximum=n>maximum?n:maximum;error|=n<0?1:n>200?2:0;count+=activeFlags[row];}
 maxima[threadIdx.x]=maximum;counts[threadIdx.x]=count;errors[threadIdx.x]=error;__syncthreads();
 for(int offset=SELECTOR_THREADS/2;offset;offset>>=1){if(threadIdx.x<offset){int other=maxima[threadIdx.x+offset];maxima[threadIdx.x]=other>maxima[threadIdx.x]?other:maxima[threadIdx.x];counts[threadIdx.x]+=counts[threadIdx.x+offset];errors[threadIdx.x]|=errors[threadIdx.x+offset];}__syncthreads();}
 int e=0,h=0;while(e<limits.entities&&maxima[0]>limits.entity[e])++e;while(h<limits.heads&&counts[0]>limits.head[h])++h;
 return {errors[0]?-1:e*limits.heads+h,counts[0],maxima[0],e,h,errors[0]};
}
__device__ __forceinline__ void store_decision(int* status,Decision d){status[0]=d.bucket;status[1]=d.active;status[2]=d.maximum;status[3]=d.entity;status[4]=d.head;status[5]=d.error;}
__global__ void set_branch(cudaGraphConditionalHandle handle,const int* obs,const int* activeFlags,int rows,int64_t obs_stride,int column,Limits limits,int* status){
 Decision d=reduce_counts(obs,activeFlags,rows,obs_stride,column,limits);
 if(threadIdx.x==0){store_decision(status,d);cudaGraphSetConditional(handle,d.error?unsigned(limits.entities*limits.heads):unsigned(d.bucket));assert(d.error==0&&"public entity count outside zero..200");}
}
__global__ void probe_branch(const int* obs,const int* activeFlags,int rows,int64_t obs_stride,int column,Limits limits,int* status){
 Decision d=reduce_counts(obs,activeFlags,rows,obs_stride,column,limits);if(threadIdx.x==0)store_decision(status,d);
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
    if(branch->activeFlags){auto code=cudaFree(branch->activeFlags);if(result==cudaSuccess)result=code;}
    delete branch;return result;
}
void CUDART_CB never_run(void*){}
} // namespace
extern "C" {
int cn_head_branch_runtime_version(){int version=0;return cudaRuntimeGetVersion(&version)==cudaSuccess?version:0;}
const char* cn_head_branch_error_string(int code){return cudaGetErrorString(cudaError_t(code));}
int cn_head_branch_validate_child(void* graph,int* counts,int capacity){
    if(!counts||capacity<32)return int(cudaErrorInvalidValue);
    for(int i=0;i<32;++i)counts[i]=0;
    return int(validate_child(static_cast<cudaGraph_t>(graph),0,counts));
}
// Qualification constructs forbidden nodes but never launches them.
int cn_head_branch_test_unsupported_children(){
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
int cn_head_branch_create(const int* observations,const uint8_t* masks,int rows,int64_t obs_stride,int64_t mask_stride,int entity_column,int actions,const int* entities,int ne,const int* heads,int nh,void*const* child_graphs,int* status,void** result){
    if(!result)return int(cudaErrorInvalidValue);*result=nullptr;
    Limits bounds;auto code=validate(observations,masks,rows,obs_stride,mask_stride,entity_column,actions,entities,ne,heads,nh,bounds);int count=ne*nh;
    if(code!=cudaSuccess||!child_graphs||!status)return int(code!=cudaSuccess?code:cudaErrorInvalidValue);
    for(int i=0;i<count;++i){code=validate_child(static_cast<cudaGraph_t>(child_graphs[i]));if(code!=cudaSuccess)return int(code);}
    auto* branch=new(std::nothrow) Branch;if(!branch)return int(cudaErrorMemoryAllocation);
    code=cudaGetDevice(&branch->device);if(code!=cudaSuccess){release(branch);return int(code);}
    code=cudaMalloc(&branch->activeFlags,sizeof(int)*size_t(rows));if(code!=cudaSuccess){release(branch);return int(code);}
    branch->obs=observations;branch->masks=masks;branch->rows=rows;branch->column=entity_column;branch->actions=actions;branch->obsStride=obs_stride;branch->maskStride=mask_stride;branch->bounds=bounds;
    code=cudaGraphCreate(&branch->graph,0);if(code!=cudaSuccess){release(branch);return int(code);}
    cudaGraphConditionalHandle handle;
    code=cudaGraphConditionalHandleCreate(&handle,branch->graph,unsigned(count),cudaGraphCondAssignDefault);
    if(code!=cudaSuccess){release(branch);return int(code);}
    // Two ordered kernel nodes: coalesced per-row flags, then small reduction.
    int* activeFlags=branch->activeFlags;
    void* prepare_arguments[]={&masks,&rows,&mask_stride,&actions,&activeFlags};
    cudaGraphNodeParams prepare{};prepare.type=cudaGraphNodeTypeKernel;prepare.kernel.func=reinterpret_cast<void*>(prepare_active);
    prepare.kernel.gridDim=dim3((rows+7)/8);prepare.kernel.blockDim=dim3(SELECTOR_THREADS);prepare.kernel.kernelParams=prepare_arguments;
    cudaGraphNode_t prepare_node,selector_node,conditional_node;
    code=cudaGraphAddNode(&prepare_node,branch->graph,nullptr,0,&prepare);if(code!=cudaSuccess){release(branch);return int(code);}
    void* arguments[]={&handle,&observations,&activeFlags,&rows,&obs_stride,&entity_column,&bounds,&status};
    cudaGraphNodeParams selector{};selector.type=cudaGraphNodeTypeKernel;
    selector.kernel.func=reinterpret_cast<void*>(set_branch);selector.kernel.gridDim=dim3(1);selector.kernel.blockDim=dim3(SELECTOR_THREADS);selector.kernel.kernelParams=arguments;
    code=cudaGraphAddNode(&selector_node,branch->graph,&prepare_node,1,&selector);if(code!=cudaSuccess){release(branch);return int(code);}
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
int cn_head_branch_launch(void* opaque,void* stream){
    auto* branch=static_cast<Branch*>(opaque);if(!branch)return int(cudaErrorInvalidValue);
    int device=-1;auto code=cudaGetDevice(&device);if(code!=cudaSuccess)return int(code);
    if(device!=branch->device)return int(cudaErrorInvalidDevice);
    cudaStreamCaptureStatus capturing;
    code=cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&capturing);if(code!=cudaSuccess)return int(code);
    if(capturing!=cudaStreamCaptureStatusNone)return int(cudaErrorStreamCaptureUnsupported);
    return int(cudaGraphLaunch(branch->executable,static_cast<cudaStream_t>(stream)));
}
int cn_head_branch_probe(void* opaque,int* status,void* stream){
 auto* b=static_cast<Branch*>(opaque);if(!b||!status)return int(cudaErrorInvalidValue);
 int device=-1;auto code=cudaGetDevice(&device);if(code!=cudaSuccess)return int(code);if(device!=b->device)return int(cudaErrorInvalidDevice);
 cudaStreamCaptureStatus capture;code=cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&capture);if(code!=cudaSuccess)return int(code);if(capture!=cudaStreamCaptureStatusNone)return int(cudaErrorStreamCaptureUnsupported);
 prepare_active<<<(b->rows+7)/8,SELECTOR_THREADS,0,static_cast<cudaStream_t>(stream)>>>(b->masks,b->rows,b->maskStride,b->actions,b->activeFlags);
 code=cudaGetLastError();if(code!=cudaSuccess)return int(code);
 probe_branch<<<1,SELECTOR_THREADS,0,static_cast<cudaStream_t>(stream)>>>(b->obs,b->activeFlags,b->rows,b->obsStride,b->column,b->bounds,status);return int(cudaGetLastError());
}
// Caller must order all launches before destruction; Python close synchronizes
// the owning device before releasing the executable and retained Torch children.
int cn_head_branch_destroy(void* opaque){return int(release(static_cast<Branch*>(opaque)));}
}
