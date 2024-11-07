#include "base/allocator.h"
#include <cuda_runtime_api.h>


namespace mbase 
{
    // 需要显式声明
    template class AllocatorManager<int>;
    template class AllocatorManager<float>;
    template class AllocatorManager<double>;

    template <typename T>
    std::unique_ptr<T[]> AllocatorManager<T>::allocate(size_t byte_size) const
    {
        if (!memory_pool_.empty())
        {
            auto ptr = std::move(memory_pool_.back());
            memory_pool_.pop_back();
            return ptr;
        }

        return std::unique_ptr<T[]>(new T[byte_size]);
    }


    template <typename T>
    void AllocatorManager<T>::release(T* data) const
    {
        memory_pool_.emplace_back(data);
    }

    template <typename T>
    void AllocatorManager<T>::memcpy(const void* src_data, void* dest_data, size_t byte_size,
                                     MemcpyDirection memcpy_kind, void* stream) const
    {
        if (!src_data || !dest_data || byte_size == 0)
            return;

        int device_id = get_device_id();
        cudaStream_t stream_ = nullptr;
        if (stream)
            stream_ = static_cast<cudaStream_t>(stream);

        if (memcpy_kind == MemcpyDirection::CPU2CPU) 
        {
            std::memcpy(dest_data, src_data, byte_size);
        } 
        else if (memcpy_kind == MemcpyDirection::CPU2CUDA) 
        {
            cudaSetDevice(device_id);               
            if (!stream_) 
            {
                cudaMemcpy(dest_data, src_data, byte_size, cudaMemcpyHostToDevice);
            } 
            else 
            { 
                cudaMemcpyAsync(dest_data, src_data, byte_size, cudaMemcpyHostToDevice, stream_);
                cudaDeviceSynchronize();
            }
        } 
        else if (memcpy_kind == MemcpyDirection::CUDA2CPU) 
        {
            cudaSetDevice(device_id);        
            if (!stream_) 
            {
                cudaMemcpy(dest_data, src_data, byte_size, cudaMemcpyDeviceToHost);
            } 
            else 
            {
                cudaMemcpyAsync(dest_data, src_data, byte_size, cudaMemcpyDeviceToHost, stream_);
                cudaDeviceSynchronize();
            }
        } 
        else if (memcpy_kind == MemcpyDirection::CUDA2CUDA) 
        {
            if (!stream_) 
            {
                cudaMemcpy(dest_data, src_data, byte_size, cudaMemcpyDeviceToDevice);
            } 
            else 
            {
                cudaMemcpyAsync(dest_data, src_data, byte_size, cudaMemcpyDeviceToDevice, stream_);
                cudaDeviceSynchronize();
            }
        } 
        else 
        {
            throw std::runtime_error("Unknown Memcpy Direction");
        }
    }

    template <typename T>
    void AllocatorManager<T>::reset_zero(T* data, size_t byte_size, void* stream) const
    {
        if (device_ == DeviceType::HOST) 
        {
            std::memset(data, 0, byte_size);
        } 
        else 
        {
            if (stream) 
            {
                cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
                cudaMemsetAsync(data, 0, byte_size, stream_);
                cudaDeviceSynchronize();
            } 
            else 
            {
                cudaMemset(data, 0, byte_size);
            }
        }
    }

    // 对齐内存
    template <typename T>
    std::unique_ptr<T[], void(*)(void*)> AllocatorManager<T>::allocate_aligned(size_t byte_size, size_t alignment) const
    {
        T* ptr = nullptr;
        if (posix_memalign(reinterpret_cast<void**>(&ptr), alignment, byte_size * sizeof(T)) != 0)
        {
            throw std::bad_alloc();
        }
        return std::unique_ptr<T[], void(*)(void*)>(ptr, free);
    }

}  // namespace base