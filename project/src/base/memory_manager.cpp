#include "base/memory_manager.h"
#include <glog/logging.h>

namespace mbase
{
    template class MemoryManager<int>;
    template class MemoryManager<float>;
    template class MemoryManager<double>;
    
    template <typename T>
    MemoryManager<T>::MemoryManager(size_t byte_size, std::shared_ptr<AllocatorManager<T>> allocator_manager,
                                    T* data, bool use_external)
        : byte_size_(byte_size), allocator_manager_(allocator_manager), use_external_(use_external)
    {
        if (data) 
        {
            // 如果使用外部传入的数据，将裸指针转换成 unique_ptr，但不负责管理释放
            data_.reset(data); // 使用 reset 将裸指针转换成 unique_ptr
            use_external_ = true;
            
        } 
        else if (allocator_manager) 
        {
            // 使用分配器进行内存分配，并包装为 unique_ptr
            device_ = allocator_manager_->get_device_type();
            data_ = allocator_manager_->allocate(byte_size_);
            use_external_ = false;

            if (!data_) 
            {
                LOG(ERROR) << "Memory allocation failed in constructor";
            }
        }
    }
    
    template <typename T>
    T* MemoryManager<T>::data() 
    {
        return data_.get();
    }

    template <typename T>
    const T* MemoryManager<T>::data() const 
    {
        return data_.get();
    }

    template <typename T>
    bool MemoryManager<T>::allocate() 
    {
        if (allocator_manager_ && byte_size_ != 0 && !data_) 
        {
            // 直接将分配返回的 unique_ptr 赋值给 data_
            data_ = allocator_manager_->allocate(byte_size_);

            if (!data_) 
            {
                LOG(ERROR) << "Memory allocation failed in allocate() method";
                return false;
            }
            return true;
        } 
        return false;
    }

    template <typename T>
    std::shared_ptr<AllocatorManager<T>> MemoryManager<T>::allocator_manager() const
    {
        return allocator_manager_;            
    }
    
    template <typename T>
    MemoryManager<T>::~MemoryManager(){}

    template <typename T>
    void MemoryManager<T>::memory_copy(const MemoryManager<T>& memory) const 
    {
        memory_copy_impl(&memory);
    }
    template <typename T>
    void MemoryManager<T>::memory_copy(const MemoryManager<T>* memory) const 
    {
        if (!memory) 
        {
            throw std::runtime_error("Null memory pointer provided for copy operation.");
        }
        memory_copy_impl(memory);
    }

    template <typename T>
    void MemoryManager<T>::memory_copy_impl(const MemoryManager<T>* memory) const 
    {
        if (!memory->data() || !data_) 
        {
            throw std::runtime_error("Invalid memory for copy operation.");
        }

        size_t copy_size = std::min(byte_size_, memory->byte_size());

        const DeviceType& source_device = memory->device_;
        const DeviceType& target_device = this->device_;

        if (source_device == DeviceType::Unknown || target_device == DeviceType::Unknown) 
        {
            throw std::runtime_error("Unknown device type in memory copy.");
        }

        // Determine the memory copy direction
        MemcpyDirection direction = MemcpyDirection::CUDA2CUDA; // Default to CUDA2CUDA
        if (source_device == DeviceType::HOST && target_device == DeviceType::HOST) 
        {
            // std::cout<< "Memory copy direction: HOST2HOST";
            direction = MemcpyDirection::CPU2CPU;
        } 
        else if (source_device == DeviceType::Device && target_device == DeviceType::HOST) 
        {
            direction = MemcpyDirection::CUDA2CPU;
        } 
        else if (source_device == DeviceType::HOST && target_device == DeviceType::Device) 
        {
            direction = MemcpyDirection::CPU2CUDA;
        }

        // Perform the memory copy 
        allocator_manager_->memcpy(this->data(), const_cast<T*>(memory->data()), copy_size, direction);
    }  


    template <typename T>
    void MemoryManager<T>::set_slice_offsets(const std::vector<int64_t>& offsets, const std::vector<size_t>& new_dims) 
    {
        slice_offsets_ = offsets;
        slice_dims_ = new_dims;
    }
}
