// #include <glog/logging.h>
// #include <cstdlib>
// #include "base/cpu_allocator.h"

// namespace mbase
// {
//     // 显示实例化模板类，以避免链接错误
//     template class CPUAllocatorManager<int>;
//     template class CPUAllocatorManager<float>;
//     template class CPUAllocatorManager<double>;

//     template <typename T>
//     CPUAllocatorManager<T>::CPUAllocatorManager() : AllocatorManager<T>(DeviceType::HOST) {}
    
//     template <typename T>
//     std::unique_ptr<T[]> CPUAllocatorManager<T>::allocate(size_t byte_size) const
//     {

//         if (!memory_pool_.empty())
//         {
//             auto ptr = std::move(memory_pool_.back());
//             memory_pool_.pop_back();
//             return ptr;
//         }
//         return std::unique_ptr<T[]>(new T[byte_size]);

//         if (byte_size == 0) 
//         {
//             return nullptr;  // 请求大小为0，返回空指针
//         }

//         return std::unique_ptr<T[]>(new T[byte_size]);
//     }

//     template <typename T>
//     void CPUAllocatorManager<T>::release(T* data) const
//     {
//         if (data) 
//         {
//             delete[] data;
//         }
//     }

//     template <typename T>
//     std::unique_ptr<T[], void(*)(void*)> CPUAllocatorManager<T>::allocate_aligned(size_t count, size_t alignment) const
//     {
//         T* data;
//         if (posix_memalign(reinterpret_cast<void**>(&data), alignment, count * sizeof(T)) != 0)
//         {
//             throw std::bad_alloc();
//         }
//         return std::unique_ptr<T[], void(*)(void*)>(data, free);
//     }

// }

#include <glog/logging.h>
#include <cstdlib>
#include "base/cpu_allocator.h"

namespace mbase
{
    // 显式实例化模板类，以避免链接错误
    template class CPUAllocatorManager<int>;
    template class CPUAllocatorManager<float>;
    template class CPUAllocatorManager<double>;

    template <typename T>
    CPUAllocatorManager<T>::CPUAllocatorManager() : AllocatorManager<T>(DeviceType::HOST) {}
    
    template <typename T>
    void CPUAllocatorManager<T>::release(T* data) const
    {
        // 将内存放回父类的 memory_pool_ 中
        if (data) 
        {
            AllocatorManager<T>::release(data); // 调用父类的 release 方法
        }
    }
}
