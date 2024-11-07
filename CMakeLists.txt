cmake_minimum_required(VERSION 3.16)
# 导出编译命令
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 指定cuda编译路径
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.1/bin/nvcc")

# 使用 C++ 和 CUDA 语言
project(cuda_learn CXX CUDA)

include(cmake/cuda.cmake)
# C++ 标准为 C++17，CUDA 标准为 14。
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Eigen3 REQUIRED)# 新增

include_directories(${EIGEN3_INCLUDE_DIRS}) # 新增

# 显式指定源文件
set(DIR_SRC
    ${CMAKE_SOURCE_DIR}/project/src/base/allocator.cpp
    ${CMAKE_SOURCE_DIR}/project/src/base/base.cpp
    ${CMAKE_SOURCE_DIR}/project/src/base/cpu_allocator.cpp
    ${CMAKE_SOURCE_DIR}/project/src/base/gpu_allocator.cpp
    ${CMAKE_SOURCE_DIR}/project/src/base/memory_manager.cpp
)


set(DIR_OP
    ${CMAKE_SOURCE_DIR}/project/src/operators/gpu/matrix_operator.cu
    ${CMAKE_SOURCE_DIR}/project/src/operators/gpu/stencil_1d_operator.cu
    ${CMAKE_SOURCE_DIR}/project/src/operators/gpu/vec_add_operator.cu
    ${CMAKE_SOURCE_DIR}/project/src/operators/interface.cpp
)




# aux_source_directory(project/src/ DIR_MAIN)
# aux_source_directory(project/src/operators/gpu/ DIR_KERNEL_CUDA)
# aux_source_directory(project/src/operators/ DIR_OPERATORS)
# aux_source_directory(project/src/base/ DIR_BASE)

add_library(LLMInfer SHARED ${DIR_SRC} ${DIR_OP})


target_link_libraries(LLMInfer glog::glog gtest gtest_main pthread cudart)

target_link_directories(LLMInfer PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

target_include_directories(LLMInfer PUBLIC ${EIGEN3_INCLUDE_DIRS}) # 新增
target_include_directories(LLMInfer PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(LLMInfer PUBLIC ${PROJECT_SOURCE_DIR}/project/include)
target_include_directories(LLMInfer PUBLIC ${PROJECT_SOURCE_DIR}/project/include/base)
target_include_directories(LLMInfer PUBLIC ${PROJECT_SOURCE_DIR}/project/src/operators)
target_include_directories(LLMInfer PUBLIC ${PROJECT_SOURCE_DIR}/project/src/operators/gpu)
target_include_directories(LLMInfer PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# 启用了 CUDA 的可分离编译功能（Separable Compilation），以便分模块编译 CUDA 代码：
set_target_properties(LLMInfer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(test)

# -----------------------以下是编译主程序--------------------------
set(link_ext_lib glog::glog GTest::gtest)

add_executable(demo project/src/main.cpp)

target_include_directories(demo PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(demo PUBLIC ${GTest_INCLUDE_DIR})
# 这一行可能有点重复引用
target_include_directories(demo PUBLIC ${PROJECT_SOURCE_DIR}/project/include)

target_link_directories(demo PUBLIC ${PROJECT_SOURCE_DIR}/lib)

target_link_libraries(demo ${link_ext_lib})
target_link_libraries(demo LLMInfer)

set_target_properties(demo PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)