
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
    void CPUAllocatorManager<T>::release(T* data) const
    {
        // 将内存放回父类的 memory_pool_ 中
        if (data) 
        {
            AllocatorManager<T>::release(data); // 调用父类的 release 方法
        }
    }
}
