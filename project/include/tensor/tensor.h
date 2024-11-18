#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>
#include <iostream>
#include <stdexcept>
#include "../base/base.h"
#include "../base/memory_manager.h"
namespace tensor 
{
    template <typename T>
    // Tensor Class Template
    class Tensor 
    {
        public:
            /**********************构造函数***************************/ 
            explicit Tensor() = default;
            /***********************一维构造***************************/
            explicit Tensor(size_t dim0,
                            mbase::DeviceType device_type,
                            mbase::DataType dtype,
                            std::shared_ptr<mbase::AllocatorManager<T>> allocator_manager,
                            T* data=nullptr,
                            bool deep_copy_data=false
                            );
            explicit Tensor(size_t dim0,
                            size_t dim1, 
                            mbase::DeviceType device_type,
                            mbase::DataType dtype,
                            std::shared_ptr<mbase::AllocatorManager<T>> allocator_manager,
                            T* data=nullptr,
                            bool deep_copy_data=false
                            );

            explicit Tensor(size_t dim0, size_t dim1, size_t dim2, mbase::DeviceType device_type, mbase::DataType dtype,
                            std::shared_ptr<mbase::AllocatorManager<T>> allocator_manager,
                            T* data=nullptr,
                            bool deep_copy_data=false
                            );

            explicit Tensor(size_t dim0, size_t dim1, size_t dim2, size_t dim3, mbase::DeviceType device_type, mbase::DataType dtype,
                            std::shared_ptr<mbase::AllocatorManager<T>> allocator_manager,
                            T* data=nullptr,
                            bool deep_copy_data=false
                            );
            // 多维构造
            explicit Tensor(const std::vector<size_t> dimensions, mbase::DeviceType device_type, mbase::DataType dtype, 
                            std::shared_ptr<mbase::AllocatorManager<T>> allocator_manager,
                            T* data=nullptr,
                            bool deep_copy_data=false
                            );

            // 模仿pytorch的tensor 的 to() 函数
            void to(const char* device);

            // 判断张量是否为空
            bool is_empty() const;
            // Resize Tensor
            void reshape(const std::vector<size_t>& new_dimensions);

            // 获取张量的数据指针

            T* data();

            const T* data() const; // 常量版本

            T& index(int64_t offset);
            // 常量版本，返回指定偏移量的数据的常量引用。

            const T& index(int64_t offset) const;
            
            // 获取张量的元素总数量。
            size_t size() const {return size_;}

            // 获取张量的维度数量。
            size_t get_dims() const {return dimensions_.size();}

            // 获取张量的数据类型。
            mbase::DataType dtype() const { return dtype_; }

            // 根据index进行索引

            // 获取维度
            const std::vector<size_t>& dimensions() const { return dimensions_; }

            // 重置张量的类型和维度信息，并清除已有的数据缓冲区。
            void reset(mbase::DataType dtype, const std::vector<size_t>& dimensions);
            // 获取张量所处的设备类型。
            mbase::DeviceType device() const { return device_; }

            void memory_initialize(std::shared_ptr<mbase::AllocatorManager<T>> allocator_manager = nullptr, T* data=nullptr, bool copy_data=false);

            // 填充张量数据
            void fill(const T& value);
            // 运算符重载，方便索引
            T& operator()(const std::vector<size_t>& indices);
            
            const T& operator()(const std::vector<size_t>& indices) const;
            // 创建并返回当前张量的深拷贝，包括数据和元数据。
            Tensor clone() const;

            const T& index(const std::vector<int64_t>& indices) const;
            T& index(const std::vector<int64_t>& indices);

            Tensor<T> slice(const std::vector<std::pair<int64_t, int64_t>>& ranges) const;

            T& operator[](const std::vector<int64_t>& indices);
            const T& operator[](const std::vector<int64_t>& indices) const;

            void print() const;


        private:
            void print_recursive(size_t dim, std::vector<int64_t> indices) const;
            // size_ 存储张量的元素总数量，初始化为 0。
            size_t size_ = 0;
            // 维度大小
            std::vector<size_t> dimensions_;
            // 内存/显存管理, 实际的数据存储位置
            std::shared_ptr<mbase::MemoryManager<T>> data_memory_;
            mbase::DataType dtype_ = mbase::DataType::Unknown;
            // 张量默认在CPU(HOST)
            mbase::DeviceType device_ = mbase::DeviceType::HOST;

            size_t device_id_ = 0;

    };

    // ----------------部分，源于模板类的成员函数实现----------------------------------------------
    template <typename T>
    const T* Tensor<T>::data() const 
    {
        if (!data_memory_) 
        {
            return nullptr;
        }
        // reinterpret_cast<T*>(buffer_->ptr())：将缓冲区指针转换为 T*。
        // const_cast<const T*>：确保返回的是常量指针。
        return const_cast<const T*>(reinterpret_cast<T*>(data_memory_->data()));
    }

    template <typename T>
    T* Tensor<T>::data() 
    {
        if (!data_memory_) 
        {
            return nullptr;
        }
        return reinterpret_cast<T*>(data_memory_->data());
    }

    // 定义 operator[]，支持直接通过 std::vector<int64_t> 索引
    template <typename T>
    T& Tensor<T>::operator[](const std::vector<int64_t>& indices) 
    {
        // std::cout<<"indices.size(): "<< indices.size()<<std::endl;
        return this->index(indices);
    }
    template <typename T>
    const T& Tensor<T>::operator[](const std::vector<int64_t>& indices) const 
    {
        return this->index(indices);
    }

    template <typename T>
    void Tensor<T>::print_recursive(size_t dim, std::vector<int64_t> indices) const 
    {
        if (dim == dimensions_.size()) {
            std::cout << (*this)[indices] << " ";
            return;
        }

        std::cout << "[";
        for (size_t i = 0; i < dimensions_[dim]; ++i) {
            indices.push_back(static_cast<int64_t>(i));
            print_recursive(dim + 1, indices);
            indices.pop_back();
            if (i < dimensions_[dim] - 1) std::cout << ", ";
        }
        std::cout << "]";

        if (dim == 0) std::cout << std::endl;
    }


}
#endif // TENSOR_H
