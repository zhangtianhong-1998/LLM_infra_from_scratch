#ifndef INFER_BASE_BUFFER_H_
#define INFER_BUFFER_H_
#include <memory>
#include "base.h"
#include "base/allocator.h"
#include "base/cpu_allocator.h"


namespace mbase
{
    // std::enable_shared_from_this 提供了一种安全的方式，使得在类内部能够获取指向当前对象的 shared_ptr
    // 同时确保该指针与外部已有的 shared_ptr 共享相同的引用计数。
    template <typename T>
    class MemoryManager : public mbase::NoCopyable, public std::enable_shared_from_this<MemoryManager<T>> 
    {
        private:
            size_t byte_size_ = 0;                // 空间大小
            DeviceType device_;                 // 数据存储的设备
            // T* data_ = nullptr;                      
            std::unique_ptr<T[]> data_;           // 使用智能指针管理数据
            std::shared_ptr<AllocatorManager<T>> allocator_manager_; 
            bool use_external_ = false; //是否使用外部传入的地址， 否则就需要自己内部对维护指针的生命周期
        public:
            explicit MemoryManager() = default; // 显式构造

            explicit MemoryManager(size_t byte_size, std::shared_ptr<AllocatorManager<T>> allocator_manager,
                            T* data=nullptr, bool use_external=false);


            virtual ~MemoryManager();
            
            // 分配函数
            bool allocate();

            //对外的数据指针  访问
            T* data();

            std::shared_ptr<AllocatorManager<T>> allocator_manager() const;
    };
    
    
} // namespace name
#endif