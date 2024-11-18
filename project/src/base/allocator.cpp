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
    void AllocatorManager<T>::memcpy(const T* src_data, T* dest_data, size_t byte_size,
                                    MemcpyDirection memcpy_kind, void* stream, bool need_sync) const 
    {
        if (!src_data || !dest_data || byte_size == 0)
        {
            return;
        }

        int device_id = get_device_id();
        cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        // enum __device_builtin__ cudaMemcpyKind
        // {
        //     cudaMemcpyHostToHost          =   0,      /**< Host   -> Host */
        //     cudaMemcpyHostToDevice        =   1,      /**< Host   -> Device */
        //     cudaMemcpyDeviceToHost        =   2,      /**< Device -> Host */
        //     cudaMemcpyDeviceToDevice      =   3,      /**< Device -> Device */
        //     cudaMemcpyDefault             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
        // };

        auto performMemcpy = [=](cudaMemcpyKind cuda_kind) {
            cudaSetDevice(device_id);
            if (stream_) 
            {
                cudaMemcpyAsync(dest_data, src_data, byte_size, cuda_kind, stream_);
            } 
            else 
            {
                cudaMemcpy(dest_data, src_data, byte_size, cuda_kind);
            }
        };

        switch (memcpy_kind) 
        {
            case MemcpyDirection::CPU2CPU:
                std::memcpy(dest_data, src_data, byte_size);
                break;

            case MemcpyDirection::CPU2CUDA:
                performMemcpy(cudaMemcpyHostToDevice);
                break;

            case MemcpyDirection::CUDA2CPU:
                performMemcpy(cudaMemcpyDeviceToHost);
                break;

            case MemcpyDirection::CUDA2CUDA:
                performMemcpy(cudaMemcpyDeviceToDevice);
                break;

            default:
                throw std::runtime_error("Unknown Memcpy Direction");
        }

        if (stream_)
        {
            cudaStreamSynchronize(stream_);
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