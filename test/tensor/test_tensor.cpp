#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <tensor/tensor.h>

#include "cpu_allocator.h"


TEST(test_tensor, create_tensor) 
{

    // 创建共享指针形式的 CPUAllocatorManager<int> 实例
    auto float_allocator = std::make_shared<mbase::CPUAllocatorManager<float>>();

    tensor::Tensor<float> ts(32, 32, mbase::DeviceType::HOST, mbase::DataType::Float32, float_allocator);
    ASSERT_EQ(ts.is_empty(), false);
}


TEST(test_tensor, clone_cpu) 
{

    auto float_allocator = std::make_shared<mbase::CPUAllocatorManager<float>>();
    tensor::Tensor<float> t1_cpu(32, 32, mbase::DeviceType::HOST, mbase::DataType::Float32, float_allocator);
    ASSERT_EQ(t1_cpu.is_empty(), false);

    std::cout<<"T1_CPU \n";
    
    for (int i = 0; i < 32; ++i) 
    {
        for (int j = 0; j < 32; ++j) 
        {
            t1_cpu[{i, j}] = 1.0f; // 赋值
        }
    }
    
    t1_cpu.print();

    tensor::Tensor<float> t2_cpu = t1_cpu.clone();

    std::cout<<"T2_CPU \n";

    t2_cpu.print();

    float* p2 = new float[32 * 32];

    std::memcpy(p2, t2_cpu.data(), sizeof(float) * 32 * 32);
    
    for (int i = 0; i < 32 * 32; ++i) 
    {
        ASSERT_EQ(p2[i], 1.f);
    }

    std::memcpy(p2, t1_cpu.data(), sizeof(float) * 32 * 32);


    for (int i = 0; i < 32 * 32; ++i) 
    {
        ASSERT_EQ(p2[i], 1.f);
    }

    delete[] p2;
}



TEST(test_tensor, 3d_print) 
{
    auto float_allocator = std::make_shared<mbase::CPUAllocatorManager<float>>();
    tensor::Tensor<float> t1_cpu(3, 3, 3, mbase::DeviceType::HOST, mbase::DataType::Float32, float_allocator);

    ASSERT_EQ(t1_cpu.is_empty(), false);

    std::cout << "T1_CPU\n";

    for (int i = 0; i < 3; ++i) 
    {
        for (int j = 0; j < 3; ++j) 
        {
            for (int k = 0; k < 3; ++k) 
            {
                t1_cpu[{i, j, k}] = 1.0f; // 赋值
            }
        }
    }

    t1_cpu.print();
}

TEST(test_tensor, slice) 
{
    auto float_allocator = std::make_shared<mbase::CPUAllocatorManager<float>>();
    tensor::Tensor<float> t1_cpu(3, 3,  mbase::DeviceType::HOST, mbase::DataType::Float32, float_allocator);


    // 填充初始值
    for (int i = 0; i < 3; ++i) 
    {
        for (int j = 0; j < 3; ++j) 
        {
            t1_cpu[{i, j}] = static_cast<float>(i * 3 + j); // 赋值
        }
    }

    std::vector<std::pair<int64_t, int64_t>> ranges = 
    {
        {0, 3},  
        {0, 3}, 

    };

    tensor::Tensor<float> sliced_tensor = t1_cpu.slice(ranges);

    sliced_tensor.print();

}