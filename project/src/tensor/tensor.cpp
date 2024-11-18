#include "tensor.h"
#include <cstring> // for std::memcpy
#include <glog/logging.h>

namespace tensor 
{
    template class Tensor<float>;  // 显式实例化模板
    template class Tensor<int>;  // 显式实例化模板
    template class Tensor<double>;  // 显式实例化模板

    // 辅助函数：解析设备字符串
    static void parse_device_string(const char* device_str, mbase::DeviceType& device_type, int& device_id)
    {
        if (strcmp(device_str, "cpu") == 0)
        {
            device_type = mbase::DeviceType::HOST;
            device_id = 0;
        }
        else if (strncmp(device_str, "cuda", 4) == 0)
        {
            device_type = mbase::DeviceType::Device;
            device_id = 0; // 默认设备 ID 为 0

            // 检查是否指定了设备 ID
            if (device_str[4] == ':' || device_str[4] == ' ')
            {
                const char* id_str = device_str + 5; // 跳过 "cuda:" 或 "cuda "
                device_id = std::atoi(id_str);
            }
        }
        else
        {
            throw std::invalid_argument("Invalid device string. Expected 'cpu' or 'cuda[:id]'.");
        }
    }

    // 1 维    
    template <typename T>
    Tensor<T>::Tensor(size_t dim0, mbase::DeviceType device_type,
                        mbase::DataType dtype,
                        std::shared_ptr<mbase::AllocatorManager<T>> allocator_manager,
                        T* data,
                        bool deep_copy_data
                    )
        : dimensions_{dim0}, device_(device_type), dtype_(dtype)
    {
        size_ = dim0;
        
        memory_initialize(allocator_manager, data, deep_copy_data);
    }
    // 2 维
    template <typename T>
    Tensor<T>::Tensor(size_t dim0,
                        size_t dim1,
                        mbase::DeviceType device_type,
                        mbase::DataType dtype,
                        std::shared_ptr<mbase::AllocatorManager<T>> allocator_manager,
                        T* data,
                        bool deep_copy_data
                        )
        : dimensions_{dim0, dim1}, device_(device_type), dtype_(dtype)
    {
        size_ = dim0 * dim1;
        memory_initialize(allocator_manager, data, deep_copy_data);
    }
    // 3 维

    template <typename T>
    Tensor<T>::Tensor(
                        size_t dim0,
                        size_t dim1,
                        size_t dim2,
                        mbase::DeviceType device_type,
                        mbase::DataType dtype,
                        std::shared_ptr<mbase::AllocatorManager<T>> allocator_manager,
                        T* data,
                        bool deep_copy_data
                    )
        : dimensions_{dim0, dim1, dim2}, device_(device_type), dtype_(dtype)
    {
        size_ = dim0 * dim1 * dim2;
        memory_initialize(allocator_manager, data, deep_copy_data);

    }
    template <typename T>
    Tensor<T>::Tensor(                        
                        size_t dim0,
                        size_t dim1,
                        size_t dim2,
                        size_t dim3,
                        mbase::DeviceType device_type,
                        mbase::DataType dtype,
                        std::shared_ptr<mbase::AllocatorManager<T>> allocator_manager,
                        T* data,
                        bool deep_copy_data
                    )
        : dimensions_{dim0, dim1, dim2, dim3}, device_(device_type), dtype_(dtype)
    {
        size_ = dim0 * dim1 * dim2 * dim3;
        memory_initialize(allocator_manager, data, deep_copy_data);
    }

    template <typename T>
    Tensor<T>::Tensor(
                        const std::vector<size_t> dimensions,
                        mbase::DeviceType device_type,
                        mbase::DataType dtype,
                        std::shared_ptr<mbase::AllocatorManager<T>> allocator_manager,
                        T* data,
                        bool deep_copy_data
                    )
        : dimensions_(dimensions), device_(device_type), dtype_(dtype)
    {
        size_ = 1;
        for (auto dim : dimensions_)
        {
            size_ *= dim;
        }
        memory_initialize(allocator_manager, data, deep_copy_data);
    }



    template <typename T>
    void Tensor<T>::to(const char* device_str)
    {
        // 解析设备字符串
        mbase::DeviceType target_device_;
        int target_device_id = 0; // 默认设备 ID 为 0

        parse_device_string(device_str, target_device_, target_device_id);

        // 检查是否需要移动数据
        if (device_ == target_device_ && device_id_ == target_device_id)
        {
            // 设备相同，无需移动
            LOG(INFO) << "Tensor is already on the target device.";
            return;
        }

        // 执行数据移动
        if (!data_memory_)
        {
            throw std::runtime_error("Data memory is not initialized.");
        }
        
        // 执行数据移动 todo
        switch (target_device_)
        {
            case mbase::DeviceType::Device:
                // move_data_to_device(target_device_type, target_device_id);
                std::cout << "move data to device: " << target_device_id << std::endl;
                break;
            case mbase::DeviceType::HOST:
                std::cout << "move data to cpu" << std::endl;
            default:
                break;
        }
        // 更新张量的设备信息
        device_ = target_device_;
        device_id_ = target_device_id;
    }


    template <typename T>
    bool Tensor<T>::is_empty() const
    {
        return size_ == 0 || !data_memory_;
    }

    template <typename T>
    void Tensor<T>::reshape(const std::vector<size_t>& new_dimensions)
    {
        size_t new_size = 1;

        for (auto dim : new_dimensions)
        {
            new_size *= dim;
        }

        if (new_size != size_)
        {
            throw std::invalid_argument("Total size of new dimensions must be the same as the original size.");
        }

        dimensions_ = new_dimensions;
    }

    /// @brief 重置张量
    /// @tparam T 
    /// @param dtype 
    /// @param dimensions 
    template <typename T>
    void Tensor<T>::reset(mbase::DataType dtype, const std::vector<size_t>& dimensions)
    {
        dtype_ = dtype;
        dimensions_ = dimensions;
        size_ = 1;
        for (auto dim : dimensions_)
        {
            size_ *= dim;
        }
        // 重新分配内存
        memory_initialize(nullptr, nullptr, false);
    }

    // 数据内存初始化函数
    template <typename T>
    void Tensor<T>::memory_initialize(std::shared_ptr<mbase::AllocatorManager<T>> allocator_manager, T* data, bool copy_data) 
    {

        size_t byte_size = size_ * sizeof(T);
        // 空的话
        if (!allocator_manager) 
        {
            // 未提供分配器，根据设备类型创建默认分配器
            switch (device_) 
            {
                case mbase::DeviceType::HOST:
                    allocator_manager = std::make_shared<mbase::CPUAllocatorManager<T>>();
                case mbase::DeviceType::Device:
                    // 需要实现 CUDA 分配器 TODO
                    allocator_manager = std::make_shared<mbase::AllocatorManager<T>>(device_, 0);
                    // allocator_manager = std::make_shared<mbase::CudaAllocatorManager<T>>();
                default:
                    allocator_manager = nullptr;
            }
        }

        // 创建数据内存管理器
        data_memory_ = std::make_shared<mbase::MemoryManager<T>>(byte_size, allocator_manager, data, copy_data);
    }


    // 添加 clone 函数
    template <typename T>
    Tensor<T> Tensor<T>::clone() const
    {
        Tensor<T> new_tensor;
        new_tensor.dimensions_ = this->dimensions_;
        new_tensor.size_ = this->size_;
        new_tensor.device_ = this->device_;
        new_tensor.device_id_ = this->device_id_;
        new_tensor.dtype_ = this->dtype_;

        // 创建新的内存管理器，并深拷贝数据
        size_t byte_size = size_ * sizeof(T);

        // 创建目标设备的分配器
        std::shared_ptr<mbase::AllocatorManager<T>> allocator_manager;

        switch (device_) 
        {
            case mbase::DeviceType::HOST:
                allocator_manager = std::make_shared<mbase::CPUAllocatorManager<T>>();
                break;
            case mbase::DeviceType::Device:
                // 需要实现 CUDA 分配器 TODO
                LOG(INFO) << "需要实现 CUDA 分配器 TODO";
                allocator_manager = std::make_shared<mbase::CPUAllocatorManager<T>>();
                break;
            default:
                throw std::invalid_argument("Unsupported device type.");
        }

        new_tensor.data_memory_ = std::make_shared<mbase::MemoryManager<T>>(byte_size, allocator_manager);

        if (data_memory_)
        {
            // 复制数据
            // LOG(INFO) << "数据拷贝";
            data_memory_->memory_copy(new_tensor.data_memory_.get());

        }
        else
        {
            throw std::runtime_error("Data memory is not initialized.");
        }

        return new_tensor;
    }

    template <typename T>
    const T& Tensor<T>::index(const std::vector<int64_t>& indices) const 
    {
        if (indices.size() != dimensions_.size()) 
        {
            std::cout << "indices.size() " << indices.size() << std::endl;
            std::cout << "dimensions_.size() " << dimensions_.size() << std::endl;

            throw std::invalid_argument("Number of indices must match tensor dimensions.");
        }

        size_t flat_index = 0;
        size_t stride = 1;

        for (int64_t i = dimensions_.size() - 1; i >= 0; --i) 
        {
            int64_t idx = indices[i];

            // Handle negative indexing
            if (idx < 0) idx += dimensions_[i];

            // Validate index
            if (idx < 0 || idx >= static_cast<int64_t>(dimensions_[i])) {
                throw std::out_of_range("Index out of range.");
            }

            flat_index += idx * stride;
            stride *= dimensions_[i];
        }

        return data()[flat_index];
    }
    template <typename T>
    T& Tensor<T>::index(const std::vector<int64_t>& indices) {
        if (indices.size() != dimensions_.size()) {
            throw std::invalid_argument("Number of indices must match tensor dimensions.");
        }

        size_t flat_index = 0;
        size_t stride = 1;

        for (int64_t i = dimensions_.size() - 1; i >= 0; --i) {
            int64_t idx = indices[i];

            // Handle negative indexing
            if (idx < 0) idx += dimensions_[i];

            // Validate index
            if (idx < 0 || idx >= static_cast<int64_t>(dimensions_[i])) {
                throw std::out_of_range("Index out of range.");
            }

            flat_index += idx * stride;
            stride *= dimensions_[i];
        }

        return data()[flat_index];
    }

    template <typename T>
    Tensor<T> Tensor<T>::slice(const std::vector<std::pair<int64_t, int64_t>>& ranges) const 
    {
        if (ranges.size() != dimensions_.size()) {
            throw std::invalid_argument("Number of slicing ranges must match tensor dimensions.");
        }

        std::vector<size_t> new_dimensions;
        size_t new_size = 1;
        std::vector<int64_t> offsets(dimensions_.size(), 0);

        for (size_t i = 0; i < ranges.size(); ++i) 
        {
            auto [start, end] = ranges[i];

            // Handle negative indexing
            if (start < 0) start += dimensions_[i];
            if (end < 0) end += dimensions_[i];

            // Validate range
            if (start < 0 || end <= start || end > static_cast<int64_t>(dimensions_[i])) 
            {
                throw std::out_of_range("Invalid slicing range.");
            }

            offsets[i] = start;
            new_dimensions.push_back(static_cast<size_t>(end - start));
            new_size *= new_dimensions.back();
        }

        // Create a new tensor to hold the sliced view
        Tensor sliced_tensor;
        sliced_tensor.dimensions_ = new_dimensions;
        sliced_tensor.size_ = new_size;
        sliced_tensor.dtype_ = dtype_;
        sliced_tensor.device_ = device_;
        sliced_tensor.device_id_ = device_id_;
        sliced_tensor.data_memory_ = data_memory_;

        // Adjust memory manager to include offsets
        sliced_tensor.data_memory_->set_slice_offsets(offsets, new_dimensions);

        return sliced_tensor;
    }


    // 模仿pytorch 实现高维张量 打印 
    template <typename T>
    void Tensor<T>::print() const 
    {
        if (dimensions_.empty()) 
        {
            std::cout << "Empty Tensor" << std::endl;
            return;
        }

        std::cout << "Tensor dimensions: [";

        for (size_t i = 0; i < dimensions_.size(); ++i) 
        {
            std::cout << dimensions_[i];
            if (i < dimensions_.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";

        if (size_ == 0) 
        {
            std::cout << "Tensor is empty." << std::endl;
            return;
        }

        print_recursive(0, {});
    }

} // namespace tensor
