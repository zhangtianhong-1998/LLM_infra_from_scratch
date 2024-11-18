#ifndef __UTILS_TOOL_H__
#define __UTILS_TOOL_H__
#include <cuda_runtime.h>
#include <iostream>
#include <system_error>

#define CHECK_LAST_KERNEL() __kernelCheckError(__FILE__, __LINE__);

#define CHECK_CUDA_ERROR(call) __checkCudaError(call, __FILE__, __LINE__);

inline static void __checkCudaError(cudaError_t err, const char* file, const int line) 
{
    if (err != cudaSuccess) 
    {
        printf("ERROR: %s: %d", file, line);
        printf("Code: %s, Reason: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(-1);
    }
}

inline static void __kernelCheckError(const char* file, int line)
{
    // 这里使用cudaPeekAtLastError，因为这里不会重置错误码
    // 还有另一种API cudaGetLastError，但是会重置错误码
    cudaError_t err = cudaPeekAtLastError();
    if (cudaSuccess != err)
    {
        printf("ERROR: %s: %d", file, line);
        printf("Code: %s, Reason: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(-1);
    }
}
// 打印GPU信息的工具函数
inline static void getDeviceInfo()
{
    int deviceCount;
    int index = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    std::cout << "-------------------------GPU Information----------------------------"<< std::endl;
    for (index = 0; index < deviceCount; index++)
    {
        cudaDeviceProp deviceProp;
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, index));
        std::cout << "*************************Architecture Information*************************"<< std::endl;
        std::cout << "Device ID: " << index << std::endl;
        std::cout << "Device name: " << deviceProp.name << std::endl;
        std::cout << "Device Global memory: " << (float)deviceProp.totalGlobalMem / 1024 / 1024 / 1024 << " GB" << std::endl;
        std::cout << "Device L2 cache size: " << (float)deviceProp.l2CacheSize / 1024 / 1024 << " MB" << std::endl;

        std::cout << "Device Shared memory per block: " << (float)deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "Device Shared memory per SM: " << (float)deviceProp.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
        std::cout << "Device Clock rate: " << deviceProp.clockRate / 1E6 << " GHz" << std::endl;
        std::cout << "Device Memory clock rate: " << deviceProp.memoryClockRate / 1E6 << " GHz" << std::endl;
        std::cout << "Device Numebr of SMs: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "Device Warp size: " << deviceProp.warpSize << std::endl;
        std::cout << "Device Memory bus width: " << deviceProp.memoryBusWidth << " bit" << std::endl;
        std::cout << "Device Memory transfer rate: " << deviceProp.memoryClockRate / 1E6 << " GB/s" << std::endl;
        std::cout << "Device Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "*************************Parameter Information*************************"<< std::endl;
        std::cout << "Device Max block numbers: " << deviceProp.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "Device Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Device Max threads per SM: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Max block dim: " << " X: " << deviceProp.maxThreadsDim[0] << " Y: "<< deviceProp.maxThreadsDim[1] << " Z:" << deviceProp.maxThreadsDim[2] << std::endl;
        std::cout << "Max grid dim: " << " X: " << deviceProp.maxGridSize[0] << " Y: "<< deviceProp.maxGridSize[1] << " Z:" << deviceProp.maxGridSize[2] << std::endl;

        std::cout << "Device Registers per block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "-------------------------------------------------------------------"<< std::endl;
    }
}


#endif


