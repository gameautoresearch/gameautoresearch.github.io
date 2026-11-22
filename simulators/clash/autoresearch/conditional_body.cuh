// SPDX-License-Identifier: Apache-2.0
// Included ahead of a frozen native graph bridge (see device_branch_flat.cu): from here on, adding a captured
// graph to a conditional body copies that graph's own nodes and dependency edges into the body, instead of
// adding one child-graph node that wraps it. No simulator or game mechanics; the kernels, their arguments and
// their order are exactly the captured ones.
//
// Why: instantiating the nested form segfaults inside libcuda (cuGraphInstantiateWithFlags) on NVIDIA driver
// 620.70.06 (CUDA UMD 13.5) whenever the wrapped graph holds a memcpy node, which Torch records for a
// same-device copy_. The same memcpy node placed directly in the body instantiates and runs there and on
// 610.57.04. Isolated with a standalone program on both drivers: reports/clash-driver-segfault-2026-11-13.md.
//
// The driver API is used because Torch's kernels belong to Torch's own CUDA runtime: a kernel node read back
// through the driver is a function handle valid for every runtime in the process; a runtime one is not.
// Its functions are looked up in the already loaded driver library at first use, so the bridge builds with the
// same compiler command as the native one and needs no driver library where the worker image is built.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <unordered_map>
#include <vector>
namespace conditional_body {
#define CONDITIONAL_BODY_DRIVER(X) X(cuCtxGetCurrent) X(cuGraphGetNodes) X(cuGraphGetEdges) X(cuGraphNodeGetType) \
    X(cuGraphChildGraphNodeGetGraph) X(cuGraphKernelNodeGetParams) X(cuGraphAddKernelNode) X(cuGraphKernelNodeCopyAttributes) \
    X(cuGraphMemcpyNodeGetParams) X(cuGraphAddMemcpyNode) X(cuGraphMemsetNodeGetParams) X(cuGraphAddMemsetNode) \
    X(cuGraphAddEmptyNode) X(cuGraphAddDependencies)
#define CONDITIONAL_BODY_TEXT2(name) #name
#define CONDITIONAL_BODY_TEXT(name) CONDITIONAL_BODY_TEXT2(name)
// cuda.h maps some names to versioned symbols; the member keeps the plain name and binds the mapped symbol.
#define CONDITIONAL_BODY_MEMBER(name) decltype(&name) name##_=nullptr;
#define CONDITIONAL_BODY_BIND(name) name##_=reinterpret_cast<decltype(&name)>(dlsym(library,CONDITIONAL_BODY_TEXT(name)));ready=ready&&name##_;
struct Driver {
    CONDITIONAL_BODY_DRIVER(CONDITIONAL_BODY_MEMBER)
    bool ready=false;
    Driver(){
        void* library=dlopen("libcuda.so.1",RTLD_NOW);if(!library)return;
        ready=true;CONDITIONAL_BODY_DRIVER(CONDITIONAL_BODY_BIND)
    }
};
inline const Driver& driver(){static const Driver value;return value;}
struct Ends {std::vector<CUgraphNode> roots,leaves;};
// True when every node, at any nesting depth, is one this file can copy.
inline bool copyable(CUgraph graph,int depth=0){
    const auto& cu=driver();
    if(!cu.ready||!graph||depth>64)return false;
    size_t count=0;if(cu.cuGraphGetNodes_(graph,nullptr,&count)!=CUDA_SUCCESS)return false;
    std::vector<CUgraphNode> nodes(count);
    if(count&&cu.cuGraphGetNodes_(graph,nodes.data(),&count)!=CUDA_SUCCESS)return false;
    for(auto node:nodes){
        CUgraphNodeType type;if(cu.cuGraphNodeGetType_(node,&type)!=CUDA_SUCCESS)return false;
        switch(type){
            case CU_GRAPH_NODE_TYPE_KERNEL:case CU_GRAPH_NODE_TYPE_MEMCPY:
            case CU_GRAPH_NODE_TYPE_MEMSET:case CU_GRAPH_NODE_TYPE_EMPTY:break;
            case CU_GRAPH_NODE_TYPE_GRAPH:{CUgraph nested=nullptr;if(cu.cuGraphChildGraphNodeGetGraph_(node,&nested)!=CUDA_SUCCESS||!copyable(nested,depth+1))return false;break;}
            default:return false;
        }
    }
    return true;
}
inline CUresult copy(CUgraph source,CUgraph body,CUcontext context,Ends& ends,int depth=0){
    const auto& cu=driver();
    if(!cu.ready||!source||!body||depth>64)return CUDA_ERROR_INVALID_VALUE;
    size_t count=0;auto code=cu.cuGraphGetNodes_(source,nullptr,&count);if(code)return code;
    std::vector<CUgraphNode> nodes(count),first(count),last(count);
    if(count){code=cu.cuGraphGetNodes_(source,nodes.data(),&count);if(code)return code;}
    std::unordered_map<CUgraphNode,size_t> index;
    for(size_t i=0;i<count;++i){
        index[nodes[i]]=i;
        CUgraphNodeType type;code=cu.cuGraphNodeGetType_(nodes[i],&type);if(code)return code;
        switch(type){
            case CU_GRAPH_NODE_TYPE_KERNEL:{CUDA_KERNEL_NODE_PARAMS params;code=cu.cuGraphKernelNodeGetParams_(nodes[i],&params);if(code)return code;
                code=cu.cuGraphAddKernelNode_(&first[i],body,nullptr,0,&params);if(code)return code;
                code=cu.cuGraphKernelNodeCopyAttributes_(first[i],nodes[i]);if(code)return code;last[i]=first[i];break;}
            case CU_GRAPH_NODE_TYPE_MEMCPY:{CUDA_MEMCPY3D params;code=cu.cuGraphMemcpyNodeGetParams_(nodes[i],&params);if(code)return code;
                code=cu.cuGraphAddMemcpyNode_(&first[i],body,nullptr,0,&params,context);if(code)return code;last[i]=first[i];break;}
            case CU_GRAPH_NODE_TYPE_MEMSET:{CUDA_MEMSET_NODE_PARAMS params;code=cu.cuGraphMemsetNodeGetParams_(nodes[i],&params);if(code)return code;
                code=cu.cuGraphAddMemsetNode_(&first[i],body,nullptr,0,&params,context);if(code)return code;last[i]=first[i];break;}
            case CU_GRAPH_NODE_TYPE_EMPTY:{code=cu.cuGraphAddEmptyNode_(&first[i],body,nullptr,0);if(code)return code;last[i]=first[i];break;}
            case CU_GRAPH_NODE_TYPE_GRAPH:{  // a nested graph keeps its place between an entry and an exit node
                CUgraph nested=nullptr;code=cu.cuGraphChildGraphNodeGetGraph_(nodes[i],&nested);if(code)return code;
                code=cu.cuGraphAddEmptyNode_(&first[i],body,nullptr,0);if(code)return code;
                code=cu.cuGraphAddEmptyNode_(&last[i],body,nullptr,0);if(code)return code;
                Ends inner;code=copy(nested,body,context,inner,depth+1);if(code)return code;
                if(inner.roots.empty()){code=cu.cuGraphAddDependencies_(body,&first[i],&last[i],1);if(code)return code;}
                for(auto root:inner.roots){code=cu.cuGraphAddDependencies_(body,&first[i],&root,1);if(code)return code;}
                for(auto leaf:inner.leaves){code=cu.cuGraphAddDependencies_(body,&leaf,&last[i],1);if(code)return code;}
                break;}
            default:return CUDA_ERROR_NOT_SUPPORTED;
        }
    }
    size_t edges=0;code=cu.cuGraphGetEdges_(source,nullptr,nullptr,&edges);if(code)return code;
    std::vector<CUgraphNode> from(edges),to(edges);std::vector<char> entered(count,0),left(count,0);
    if(edges){code=cu.cuGraphGetEdges_(source,from.data(),to.data(),&edges);if(code)return code;}
    for(size_t e=0;e<edges;++e){
        auto a=index.find(from[e]),b=index.find(to[e]);if(a==index.end()||b==index.end())return CUDA_ERROR_INVALID_VALUE;
        code=cu.cuGraphAddDependencies_(body,&last[a->second],&first[b->second],1);if(code)return code;
        left[a->second]=1;entered[b->second]=1;
    }
    for(size_t i=0;i<count;++i){if(!entered[i])ends.roots.push_back(first[i]);if(!left[i])ends.leaves.push_back(last[i]);}
    return CUDA_SUCCESS;
}
// Same signature as cudaGraphAddChildGraphNode. A child added without dependencies whose nodes can all be
// copied is copied (no node handle is returned for it); anything else, or a process whose driver library
// cannot be looked up, adds it as the runtime would.
inline cudaError_t add_child(cudaGraphNode_t* node,cudaGraph_t graph,const cudaGraphNode_t* dependencies,size_t count,cudaGraph_t child){
    const auto& cu=driver();CUcontext context=nullptr;
    if(count||!cu.ready||cu.cuCtxGetCurrent_(&context)!=CUDA_SUCCESS||!context||!copyable(child))
        return cudaGraphAddChildGraphNode(node,graph,dependencies,count,child);
    if(node)*node=nullptr;
    Ends ends;return cudaError_t(copy(child,graph,context,ends));  // CUresult and cudaError_t share their numbering
}
}  // namespace conditional_body
#define cudaGraphAddChildGraphNode(node,graph,dependencies,count,child) conditional_body::add_child(node,graph,dependencies,count,child)
