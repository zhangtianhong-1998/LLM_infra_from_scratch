#ifndef INFER_BASE_ALLOC_H_
#define INFER_BASE_ALLOC_H_
#include <map>
#include <memory>
#include "base.h"

namespace mbase
{
    enum class MemcpyDirection
    {
        CPU2CPU = 0,
        CPU2CUDA = 1,
        CUDA2CPU = 2,
        CUDA2CUDA = 3,
    };
    
    /******************************内存管理基类***************************************** */
    template <typename T>
    class AllocatorManager
    {
        protected:
            mutable std::vector<std::unique_ptr<T[]>> memory_pool_;
            DeviceType device_ = DeviceType::Unknown; 

            int device_id_ = 0;

        public:
            // 无参构造函数，设置默认值
            AllocatorManager(DeviceType device = DeviceType::HOST, int device_id = 0)
                : device_(device), device_id_(device_id) {}

            // 虚函数，拿到设备
            virtual DeviceType get_device_type() const { return device_; }
            // 对于多卡的设备编号
            int get_device_id() const { return device_id_; }

            // 分配内存方法，支持内存池优化
            virtual std::unique_ptr<T[]> allocate(size_t byte_size) const;

            // 内存对齐分配
            virtual std::unique_ptr<T[], void(*)(void*)> allocate_aligned(size_t byte_size, size_t alignment) const;

            // 释放内存，将内存放回内存池
            virtual void release(T* data) const;


            virtual void memcpy(const T* src_data, T* dest_data, size_t byte_size,
                                MemcpyDirection memcpy_kind = MemcpyDirection::CPU2CPU, void* stream = nullptr,
                                bool need_sync = false
                                ) const;

            virtual void reset_zero(T* data, size_t byte_size, void* stream = nullptr) const;
            
            virtual ~AllocatorManager() = default;
    };



} // namespace base
#endif