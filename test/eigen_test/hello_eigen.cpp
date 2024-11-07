#include <glog/logging.h>
#include <gtest/gtest.h>
#include <Eigen/Eigen>
#include "utils.h"
#include <iostream>


TEST(test_eigen, hello_eigen) 
{
    double a;
    
    Eigen::Vector3i index1(11, 21, 31);

    a = index1.norm();

    std::cout << "a is " << a << std::endl;


}
