/**
 * Copyright 2020 Hung-Hsin Chen, LSA Lab, National Tsing Hua University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * The library to intercept applications' CUDA-related function calls.
 *
 * This hook library will try connecting to scheduling system when first intercepted function being
 * called (with information specified in environment variables). After then, all CUDA kernel
 * launches and some GPU memory-related activity will be controlled by this hook library.
 */

#define __USE_GNU
#include "hook.h"

#include <arpa/inet.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <netinet/in.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "nvml.h"
#include "comm.h"
#include "debug.h"
#include "predictor.h"
#include "util.h"

#define STRINGIFY(x) #x
#define CUDA_SYMBOL_STRING(x) STRINGIFY(x)
#define SYNCP_MESSAGE

typedef struct MemInfo {
  size_t total;
  size_t used;
} MemInfo;

MemInfo *get_mem_info() {
  static MemInfo memInfo = {(long unsigned) 1 << (long unsigned) 32, 0};
  return &memInfo;
}

typedef void *(*fnDlsym)(void *, const char *);

void resetDlInfo(Dl_info *info) {
    if(info == NULL) return;
    info->dli_fbase = NULL;
    info->dli_fname = NULL;
    info->dli_sname = NULL;
    info->dli_saddr = NULL;
}

/* Brute force method to guess the original dlsym address
   (we cannot easily find it because we have erased the dlsym
   symbol and recent libc versions do not expose
   dlsym like symbols anymore...)
*/
fnDlsym scan_address_space() {
#ifdef _DEBUG
    printf("Scan address space to get dlsym original address\n");
#endif
    int bytes_around = 1000;
    int b;

    char *candidate_addr = ((char *) dlopen) - bytes_around;

    Dl_info info;

    for(b=0; b < 2 * bytes_around + 1; b++){
        resetDlInfo(&info);
        candidate_addr += 1;
        dladdr((void *)candidate_addr, &info);
        if(info.dli_sname != NULL)
        {
#ifdef _DEBUG
            printf("Found symbol %s address %p!\n", info.dli_sname, (char *) info.dli_saddr);
#endif
            if(strcmp(info.dli_sname,"dlsym") == 0) {
#ifdef _DEBUG
                printf("Real dlsym found !! Address %p in file %s\n", (char *) info.dli_saddr, info.dli_fname);
#endif
                return (fnDlsym) info.dli_saddr;
            }
        }
    }
    return NULL;
}

static void *real_dlsym(void *handle, const char *symbol) {
    static fnDlsym o_dlsym = NULL;

#ifdef _DEBUG
    printf("OH YEAH dlsym called for symbol %s\n", symbol);
#endif
    if(o_dlsym == NULL)
    {
        o_dlsym = scan_address_space();

        if(o_dlsym == NULL)
        {
            perror("Not able to get original dlsym address !\n");
            exit(EXIT_FAILURE);
        }
    }
  return (*o_dlsym)(handle, symbol);
}

struct hookInfo {
  int debug_mode;
  void *preHooks[NUM_HOOK_SYMBOLS];
  void *postHooks[NUM_HOOK_SYMBOLS];
  int call_count[NUM_HOOK_SYMBOLS];

  hookInfo() {
    const char *envHookDebug;

    envHookDebug = getenv("CU_HOOK_DEBUG");
    if (envHookDebug && envHookDebug[0] == '1')
      debug_mode = 1;
    else
      debug_mode = 0;
  }
};

static struct hookInfo hook_inf;

/*
 ** interposed functions
 */
void *dlsym(void *handle, const char *symbol) {
  // Early out if not a CUDA driver symbol
  if ((strncmp(symbol, "cu", 2) != 0) && (strcmp(symbol, "nvmlDeviceGetMemoryInfo_v2") != 0)) {
    return (real_dlsym(handle, symbol));
  }

  if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemAlloc)) == 0) {
    return (void *)(&cuMemAlloc);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemAllocManaged)) == 0) {
    return (void *)(&cuMemAllocManaged);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemAllocPitch)) == 0) {
    return (void *)(&cuMemAllocPitch);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemFree)) == 0) {
    return (void *)(&cuMemFree);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuArrayCreate)) == 0) {
    return (void *)(&cuArrayCreate);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuArray3DCreate)) == 0) {
    return (void *)(&cuArray3DCreate);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuArrayDestroy)) == 0) {
    return (void *)(&cuArrayDestroy);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMipmappedArrayCreate)) == 0) {
    return (void *)(&cuMipmappedArrayCreate);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMipmappedArrayDestroy)) == 0) {
    return (void *)(&cuMipmappedArrayDestroy);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemGetInfo)) == 0) {
    return (void *)(&cuMemGetInfo);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuGetProcAddress)) == 0){
    return (void *)(&cuGetProcAddress);
  } else if (strcmp(symbol, "nvmlDeviceGetMemoryInfo_v2") == 0) {
    return (void *) (&nvmlDeviceGetMemoryInfo_v2);
  } else if (strcmp(symbol, "cuDeviceTotalMem") == 0) {
    return (void *) (&cuDeviceTotalMem);
  }
  // omit cuDeviceTotalMem here so there won't be a deadlock in cudaEventCreate when we are in
  // initialize(). Functions called by cliet are still being intercepted.
  return (real_dlsym(handle, symbol));
}

/* Support max gpu number and current number */
const int max_gpu_num = 8;
int current_gpu_num = 0;

static pthread_once_t init_done = PTHREAD_ONCE_INIT;

// GPU memory allocation information
pthread_mutex_t allocation_mutex = PTHREAD_MUTEX_INITIALIZER;

std::map<CUdeviceptr, size_t> allocation_map[max_gpu_num];
size_t gpu_mem_used[max_gpu_num] = {0};  // local accounting only


/**
 * Handle get current device id
 */
int get_current_device_id() {
  CUdevice device;
  CUresult rc = cuCtxGetDevice(&device);
  if (rc != CUDA_SUCCESS || device >= max_gpu_num) {
    ERROR("failed to get current device: %d", rc);
  }

  // DEBUG("Operation on device %d", device);
  return device;
}


/**
 * get available GPU memory from Pod manager/scheduler
 * assume user memory limit won't exceed hardware limit
 * @return remaining memory, memory limit
 */
std::pair<size_t, size_t> get_gpu_memory_info() {
  MemInfo *memInfo = get_mem_info();
  return std::make_pair(memInfo->total - memInfo->used, memInfo->total);
}

/**
 * send memory allocate/free information to Pod manager/scheduler
 * @param bytes memory size
 * @param is_allocate 1 for allocation, 0 for free
 * @return request succeed or not
 */
CUresult update_memory_usage(size_t bytes, int is_allocate) {

  MemInfo *memInfo = get_mem_info();
  if(is_allocate) {
    size_t freeMem = memInfo->total - memInfo->used;
#ifdef _DEBUG
    printf("Trying to allocate %lu bytes\n", bytes);
    printf("Free memory %lu bytes\n", freeMem);
    printf("Used memory %lu bytes\n", memInfo->used);
#endif
    if(freeMem < bytes) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    memInfo->used += bytes;
  }
  else {
    memInfo->used -= bytes;
    if(memInfo->used < 0) {
      memInfo->used = 0;
    }
  }
#ifdef _DEBUG
  printf("Success\n");
  pid_t tid = syscall(__NR_gettid);
  printf("Thread id %d Allocated %lu\n", tid, memInfo->used);
  printf("MemInfo address %p\n", memInfo);
#endif
  return CUDA_SUCCESS;
}

/**
 * pre-hooks and post-hooks
 */

// update memory usage
CUresult cuMemFree_prehook(CUdeviceptr ptr) {
  int device = get_current_device_id();
  pthread_mutex_lock(&allocation_mutex);
  if (allocation_map[device].find(ptr) == allocation_map[device].end()) {
    DEBUG("Freeing unknown memory! %zx", ptr);
  } else {
    gpu_mem_used[device] -= allocation_map[device][ptr];
    update_memory_usage(allocation_map[device][ptr], 0);
    allocation_map[device].erase(ptr);
  }
  pthread_mutex_unlock(&allocation_mutex);
  return CUDA_SUCCESS;
}

CUresult cuArrayDestroy_prehook(CUarray hArray) { return cuMemFree_prehook((CUdeviceptr)hArray); }

CUresult cuMipmappedArrayDestroy_prehook(CUmipmappedArray hMipmappedArray) {
  return cuMemFree_prehook((CUdeviceptr)hMipmappedArray);
}

// ask backend whether there's enough memory or not
CUresult cuMemAlloc_prehook(CUdeviceptr *dptr, size_t bytesize) {
  size_t remain, limit;

  std::tie(remain, limit) = get_gpu_memory_info();

  // block allocation request before over-allocate
  if (bytesize > remain) {
    ERROR("Allocate too much memory! (request: %lu B, remain: %lu B)", bytesize, remain);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  return CUDA_SUCCESS;
}

// push memory allocation information to backend
CUresult cuMemAlloc_posthook(CUdeviceptr *dptr, size_t bytesize) {
  int device = get_current_device_id();
  pthread_mutex_lock(&allocation_mutex);
  allocation_map[device][*dptr] = bytesize;
  // send memory usage update to backend
  CUresult res = update_memory_usage(bytesize, 1);
  if (res != CUDA_SUCCESS) {
    pthread_mutex_unlock(&allocation_mutex);
    ERROR("Allocate too much memory!");
    cuMemFree(*dptr);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  // allocation_map[device][*dptr] = bytesize;
  // gpu_mem_used[device] += bytesize;
  pthread_mutex_unlock(&allocation_mutex);

  return res;
}

CUresult cuMemAllocManaged_prehook(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) {
  // TODO: This function access the unified memory. Behavior needs clarification.
  return cuMemAlloc_prehook(dptr, bytesize);
}

CUresult cuMemAllocManaged_posthook(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) {
  // TODO: This function access the unified memory. Behavior needs clarification.
  return cuMemAlloc_posthook(dptr, bytesize);
}

CUresult cuMemAllocPitch_prehook(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes,
                                 size_t Height, unsigned int ElementSizeBytes) {
  return cuMemAlloc_prehook(dptr, (*pPitch) * Height);
}
CUresult cuMemAllocPitch_posthook(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes,
                                  size_t Height, unsigned int ElementSizeBytes) {
  return cuMemAlloc_posthook(dptr, (*pPitch) * Height);
}

inline size_t CUarray_format_to_size_t(CUarray_format Format) {
  switch (Format) {
    case CU_AD_FORMAT_UNSIGNED_INT8:
    case CU_AD_FORMAT_SIGNED_INT8:
      return 1;
    case CU_AD_FORMAT_UNSIGNED_INT16:
    case CU_AD_FORMAT_SIGNED_INT16:
    case CU_AD_FORMAT_HALF:
      return 2;
    case CU_AD_FORMAT_UNSIGNED_INT32:
    case CU_AD_FORMAT_SIGNED_INT32:
    case CU_AD_FORMAT_FLOAT:
      return 4;
    default:
      return -1;
  }
}

CUresult cuArrayCreate_prehook(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  size_t totalMemoryNumber =
      pAllocateArray->Width * pAllocateArray->Height * pAllocateArray->NumChannels;
  size_t formatSize = CUarray_format_to_size_t(pAllocateArray->Format);
  return cuMemAlloc_prehook((CUdeviceptr *)pHandle, totalMemoryNumber * formatSize);
}

CUresult cuArrayCreate_posthook(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  size_t totalMemoryNumber =
      pAllocateArray->Width * pAllocateArray->Height * pAllocateArray->NumChannels;
  size_t formatSize = CUarray_format_to_size_t(pAllocateArray->Format);
  return cuMemAlloc_posthook((CUdeviceptr *)pHandle, totalMemoryNumber * formatSize);
}

CUresult cuArray3DCreate_prehook(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  size_t totalMemoryNumber = pAllocateArray->Width * pAllocateArray->Height *
                             pAllocateArray->Depth * pAllocateArray->NumChannels;
  size_t formatSize = CUarray_format_to_size_t(pAllocateArray->Format);
  return cuMemAlloc_prehook((CUdeviceptr *)pHandle, totalMemoryNumber * formatSize);
}

CUresult cuArray3DCreate_posthook(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  size_t totalMemoryNumber = pAllocateArray->Width * pAllocateArray->Height *
                             pAllocateArray->Depth * pAllocateArray->NumChannels;
  size_t formatSize = CUarray_format_to_size_t(pAllocateArray->Format);
  return cuMemAlloc_posthook((CUdeviceptr *)pHandle, totalMemoryNumber * formatSize);
}

CUresult cuMipmappedArrayCreate_prehook(CUmipmappedArray *pHandle,
                                        const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                                        unsigned int numMipmapLevels) {
  // TODO: check mipmap array size
  return CUDA_SUCCESS;
}

CUresult cuMipmappedArrayCreate_posthook(CUmipmappedArray *pHandle,
                                         const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                                         unsigned int numMipmapLevels) {
  // TODO: check mipmap array size
  return CUDA_SUCCESS;
}

//FIXME: dependent on version -> 12+
// CUresult cuGetProcAddress_preHook(const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags,
//                                   CUdriverProcAddressQueryResult* symbolStatus){
//   printf("cuGetProcAddress called for symbol %s and cuda version %d !!!\n", symbol, cudaVersion);
//   return CUDA_SUCCESS;
// }

// Cuda version 11
// CUresult cuGetProcAddress_preHook(const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags){
//   printf("cuGetProcAddress called for symbol %s and cuda version %d !!!\n", symbol, cudaVersion);
//   return CUDA_SUCCESS;
// }

typedef CUresult CUDAAPI(*fnCuGetProcAddress) (const char*, void **, int, cuuint64_t);
// FIXME: handle cudaversion and flags!!!!
static fnCuGetProcAddress oCuGetProcAddress;



CUresult nestedCuGetProcAddress(const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags) {
  printf("Nested cuGetProcAddress called for symbol %s and cuda version %d and flags %lu !!!\n",
         symbol, cudaVersion, flags);

  if(strcmp(symbol, "cuMemAlloc") == 0) {
    *pfn = (void *) &cuMemAlloc;
    return CUDA_SUCCESS;
  } else if(strcmp(symbol, "cuMemFree") == 0) {
    *pfn = (void *) &cuMemFree;
    return CUDA_SUCCESS;
  } else if (strcmp(symbol, "cuMemAllocManaged") == 0) {
    *pfn = (void *) &cuMemAllocManaged;
    return CUDA_SUCCESS;
  } else if (strcmp(symbol, "cuMemAllocPitch") == 0) {
    *pfn = (void *) &cuMemAllocPitch;
    return CUDA_SUCCESS;
  } else if (strcmp(symbol, "cuArrayCreate") == 0) {
    *pfn = (void *) &cuArrayCreate;
    return CUDA_SUCCESS;
  } else if (strcmp(symbol, "cuArray3DCreate") == 0) {
    *pfn = (void *) &cuArray3DCreate;
    return CUDA_SUCCESS;
  } else if (strcmp(symbol, "cuMipmappedArrayCreate") == 0) {
    *pfn = (void *) &cuMipmappedArrayCreate;
    return CUDA_SUCCESS;
  } else if (strcmp(symbol, "cuArrayDestroy") == 0) {
    *pfn = (void *) &cuArrayDestroy;
    return CUDA_SUCCESS;
  } else if (strcmp(symbol, "cuMipmappedArrayDestroy") == 0) {
    *pfn = (void *) &cuMipmappedArrayDestroy;
    return CUDA_SUCCESS;
  }

  CUresult res = oCuGetProcAddress(symbol, pfn, cudaVersion, flags);
  return res;
}

CUresult cuGetProcAddress(const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags){
  printf("cuGetProcAddress called for symbol %s and cuda version %d and flags %lu !!!\n",
         symbol, cudaVersion, flags);

  static fnCuGetProcAddress real_func = NULL;
  if(real_func == NULL) {
    real_func = (fnCuGetProcAddress) real_dlsym(RTLD_NEXT, CUDA_SYMBOL_STRING(cuGetProcAddress));
  }

  if(strcmp(symbol, "cuGetProcAddress") == 0) {
    printf("cuGetProcAddress on self!!!\n");

    if(oCuGetProcAddress == NULL)
    {
      CUresult result = real_func(symbol, (void **)&oCuGetProcAddress, cudaVersion, flags);
      if(result != CUDA_SUCCESS){
        return result;
      }
    }

    *pfn = (void *) nestedCuGetProcAddress;
    return CUDA_SUCCESS;
  }

  CUresult result = real_func(symbol, pfn, cudaVersion, flags);
  return result;
}

void initialize() {
  hook_inf.postHooks[CU_HOOK_MEM_ALLOC] = (void *)cuMemAlloc_posthook;
  hook_inf.postHooks[CU_HOOK_MEM_ALLOC_MANAGED] = (void *)cuMemAllocManaged_posthook;
  hook_inf.postHooks[CU_HOOK_MEM_ALLOC_PITCH] = (void *)cuMemAllocPitch_posthook;
  hook_inf.postHooks[CU_HOOK_ARRAY_CREATE] = (void *)cuArrayCreate_posthook;
  hook_inf.postHooks[CU_HOOK_ARRAY3D_CREATE] = (void *)cuArray3DCreate_posthook;
  hook_inf.postHooks[CU_HOOK_MIPMAPPED_ARRAY_CREATE] = (void *)cuMipmappedArrayCreate_posthook;
  // place pre-hooks
  hook_inf.preHooks[CU_HOOK_MEM_FREE] = (void *)cuMemFree_prehook;
  hook_inf.preHooks[CU_HOOK_ARRAY_DESTROY] = (void *)cuArrayDestroy_prehook;
  hook_inf.preHooks[CU_HOOK_MIPMAPPED_ARRAY_DESTROY] = (void *)cuMipmappedArrayDestroy_prehook;
  hook_inf.preHooks[CU_HOOK_MEM_ALLOC] = (void *)cuMemAlloc_prehook;
  hook_inf.preHooks[CU_HOOK_MEM_ALLOC_MANAGED] = (void *)cuMemAllocManaged_prehook;
  hook_inf.preHooks[CU_HOOK_MEM_ALLOC_PITCH] = (void *)cuMemAllocPitch_prehook;
  hook_inf.preHooks[CU_HOOK_ARRAY_CREATE] = (void *)cuArrayCreate_prehook;
  hook_inf.preHooks[CU_HOOK_ARRAY3D_CREATE] = (void *)cuArray3DCreate_prehook;
  hook_inf.preHooks[CU_HOOK_MIPMAPPED_ARRAY_CREATE] = (void *)cuMipmappedArrayCreate_prehook;
}

#define CU_HOOK_GENERATE_INTERCEPT(hooksymbol, funcname, params, ...)                     \
  CUresult CUDAAPI funcname params {                                                      \
    pthread_once(&init_done, initialize);                                                 \
                                                                                          \
    static void *real_func = (void *)real_dlsym(RTLD_NEXT, CUDA_SYMBOL_STRING(funcname)); \
    CUresult result = CUDA_SUCCESS;                                                       \
                                                                                          \
    if (hook_inf.debug_mode) hook_inf.call_count[hooksymbol]++;                           \
                                                                                          \
    if (hook_inf.preHooks[hooksymbol])                                                    \
      result = ((CUresult CUDAAPI(*) params)hook_inf.preHooks[hooksymbol])(__VA_ARGS__);  \
    if (result != CUDA_SUCCESS) return (result);                                          \
    result = ((CUresult CUDAAPI(*) params)real_func)(__VA_ARGS__);                        \
    if (hook_inf.postHooks[hooksymbol] && result == CUDA_SUCCESS)                         \
      result = ((CUresult CUDAAPI(*) params)hook_inf.postHooks[hooksymbol])(__VA_ARGS__); \
                                                                                          \
    return (result);                                                                      \
  }

// cuda driver alloc/free APIs
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_MEM_ALLOC, cuMemAlloc, (CUdeviceptr * dptr, size_t bytesize),
                           dptr, bytesize)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_MEM_ALLOC_MANAGED, cuMemAllocManaged,
                           (CUdeviceptr * dptr, size_t bytesize, unsigned int flags), dptr,
                           bytesize, flags)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_MEM_ALLOC_PITCH, cuMemAllocPitch,
                           (CUdeviceptr * dptr, size_t *pPitch, size_t WidthInBytes, size_t Height,
                            unsigned int ElementSizeBytes),
                           dptr, pPitch, WidthInBytes, Height, ElementSizeBytes)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_MEM_FREE, cuMemFree, (CUdeviceptr dptr), dptr)

// cuda driver array/array_destroy APIs
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_ARRAY_CREATE, cuArrayCreate,
                           (CUarray * pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray),
                           pHandle, pAllocateArray)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_ARRAY3D_CREATE, cuArray3DCreate,
                           (CUarray * pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray),
                           pHandle, pAllocateArray)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_MIPMAPPED_ARRAY_CREATE, cuMipmappedArrayCreate,
                           (CUmipmappedArray * pHandle,
                            const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                            unsigned int numMipmapLevels),
                           pHandle, pMipmappedArrayDesc, numMipmapLevels)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_ARRAY_DESTROY, cuArrayDestroy, (CUarray hArray), hArray)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_MIPMAPPED_ARRAY_DESTROY, cuMipmappedArrayDestroy,
                           (CUmipmappedArray hMipmappedArray), hMipmappedArray)

// cuda driver mem info APIs
CUresult CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
  pthread_once(&init_done, initialize);
  std::pair<size_t, size_t> mem_info = get_gpu_memory_info();
  if (hook_inf.debug_mode) hook_inf.call_count[CU_HOOK_DEVICE_TOTOAL_MEM]++;
  *bytes = mem_info.second;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetInfo(size_t *gpu_mem_free, size_t *gpu_mem_total) {
  pthread_once(&init_done, initialize);
  std::pair<size_t, size_t> mem_info = get_gpu_memory_info();
  if (hook_inf.debug_mode) hook_inf.call_count[CU_HOOK_MEM_INFO]++;
  *gpu_mem_free = mem_info.first;
  *gpu_mem_total = mem_info.second;
  return CUDA_SUCCESS;
}

nvmlReturn_t DECLDIR nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t *memory)
{
#ifdef _DEBUG
  printf("HELLO MEM INFO\n");
#endif
  MemInfo *memInfo = get_mem_info();
  memory->version = 2;
  memory->total = memInfo->total;
  memory->reserved = 0;
  memory->used = memInfo->used;
  memory->free = memory->total - memory->reserved - memory->used;
  return NVML_SUCCESS;
}
