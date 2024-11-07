#ifndef BASE_H_
#define BASE_H_
#include <glog/logging.h>
#include <cstdint>
#include <string>


namespace mbase 
{
    // 设备类型
    enum class DeviceType : uint8_t 
    {
        Unknown = 0,
        HOST = 1,
        Device = 2,
    };

    enum class DataType : uint8_t {
        Unknown = 0,
        Float32 = 1,
        Int8 = 2,
        Int32 = 3,
        Float64 = 4,
    };

    // 防止对象被拷贝或赋值的基类 这个类通过删除拷贝构造函数和赋值运算符，使得任何继承自它的类都不能被拷贝或赋值。
    class NoCopyable 
    {
        // 在类中的构造函数和析构函数被标记为 protected，这意味着 NoCopyable 不能被直接实例化，但可以通过继承来创建实例。这使得它只能作为基类使用，而不会被误用来创建实际对象。
        protected:
            NoCopyable() = default;

            ~NoCopyable() = default;
            // 删除拷贝构造
            NoCopyable(const NoCopyable&) = delete;
            // 删除 赋值拷贝
            NoCopyable& operator=(const NoCopyable&) = delete;
    };


}


#endif  // BASE_H_
