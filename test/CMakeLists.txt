include(../cmake/cuda.cmake)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Eigen3 REQUIRED) # 新增

set(link_ext_lib glog::glog GTest::gtest)

# 显式指定源文件
set(DIR_TEST
    ${CMAKE_SOURCE_DIR}/test/main.cpp
    ${CMAKE_SOURCE_DIR}/test/utils.cpp
)



set(DIR_TEST_OP
    ${CMAKE_SOURCE_DIR}/test/operators_test/operator_matrix.cpp
    ${CMAKE_SOURCE_DIR}/test/operators_test/operators_test.cpp
    ${CMAKE_SOURCE_DIR}/test/operators_test/operator_stencil_1d.cpp
    ${CMAKE_SOURCE_DIR}/test/cuda_api_test/cuda_api_test.cpp
    ${CMAKE_SOURCE_DIR}/test/eigen_test/hello_eigen.cpp
    ${CMAKE_SOURCE_DIR}/test/base/cpu_allocator_test.cpp
    ${CMAKE_SOURCE_DIR}/test/tensor/test_memory.cpp
    ${CMAKE_SOURCE_DIR}/test/tensor/test_tensor.cpp
)

include_directories(${EIGEN3_INCLUDE_DIRS}) # 新增

message(STATUS "Eigen3 include directory: ${EIGEN3_INCLUDE_DIRS}")
# 添加可执行文件
add_executable(test_main ${DIR_TEST} ${DIR_TEST_OP})

# 链接库
target_link_libraries(test_main ${link_ext_lib})

# 包含库和头文件路径
target_include_directories(test_main PUBLIC ${EIGEN3_INCLUDE_DIRS}) # 新增
target_include_directories(test_main PUBLIC ${glog_INCLUDE_DIRS})
target_include_directories(test_main PUBLIC ${GTest_INCLUDE_DIRS})
target_include_directories(test_main PUBLIC ${CMAKE_SOURCE_DIR}/test)

# 链接库文件的目录
target_link_directories(test_main PUBLIC ${PROJECT_SOURCE_DIR}/lib)

# 链接 cudalearn 库
target_link_libraries(test_main LLMInfer)

# 设置属性
set_target_properties(test_main PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
