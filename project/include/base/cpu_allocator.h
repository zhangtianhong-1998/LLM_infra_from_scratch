#ifndef INFER_BASE_CPU_ALLOC_H_
#define INFER_BASE_CPU_ALLOC_H_

#include "allocator.h"

namespace mbase {

    template <typename T>
    class CPUAllocatorManager : public AllocatorManager<T>
    {

        public:
            explicit CPUAllocatorManager();
            // std::unique_ptr<T[]> allocate(size_t byte_size) const override;
            // std::unique_ptr<T[], void(*)(void*)> allocate_aligned(size_t count, size_t alignment) const override;

            void release(T* ptr) const override;

    };

} // namespace mbase

#endif