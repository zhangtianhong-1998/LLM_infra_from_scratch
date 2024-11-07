#include "cpu_allocator.h"
#include <iostream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "utils.h"
#include "memory_manager.h"

TEST(test_cpu_allocator, cpu_allocatorManager) 
{
    // 创建共享指针形式的 CPUAllocatorManager<int> 实例
    auto int_allocator = std::make_shared<mbase::CPUAllocatorManager<int>>();

    // 使用共享指针 int_allocator 创建 MemoryManager 实例，分配100个 int 类型的空间
    mbase::MemoryManager<int> managed_memory(100 * sizeof(int), int_allocator);

    // 获取数据指针
    int* data = managed_memory.data();


    managed_memory.allocator_manager()->reset_zero(data, 100 * sizeof(int));

    // // 打印第一个元素以检查是否已初始化为零
    for(int i=0; i<100; i++)
    {
        std::cout << "Element after memset_zero: " << data[i] << std::endl;
    }

}