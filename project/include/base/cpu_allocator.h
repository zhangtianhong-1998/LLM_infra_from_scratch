#ifndef INFER_BASE_CPU_ALLOC_H_
#define INFER_BASE_CPU_ALLOC_H_

#include "allocator.h"

namespace mbase {

    template <typename T>
    class CPUAllocatorManager : public AllocatorManager<T>
    {

        public:
            // 无参构造函数
            CPUAllocatorManager() : AllocatorManager<T>() {}

            // 带参构造函数（无默认值）
            CPUAllocatorManager(DeviceType device, int device_id)
                : AllocatorManager<T>(device, device_id) {}
            // std::unique_ptr<T[]> allocate(size_t byte_size) const override;
            // std::unique_ptr<T[], void(*)(void*)> allocate_aligned(size_t count, size_t alignment) const override;

            void release(T* data) const override;

    };

} // namespace mbase

#endif