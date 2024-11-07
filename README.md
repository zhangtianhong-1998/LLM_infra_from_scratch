# 这是一个从零学习Cuda的记录
这是我得bilibili主页
https://space.bilibili.com/347417212?spm_id_from=333.999.0.0

- 代码全部开源，持续更新，教程会通过充电或者课堂形式开放

## 0. cmake安装
```
sudo apt install g++
sudo apt install gcc
sudo apt install cmake
```
## 1. googletest 安装
https://github.com/google/googletest
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8 
sudo make install 
```
## 2. googletest 安装
https://github.com/google/glog
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_GFLAGS=OFF -DWITH_GTEST=OFF ..
make -j8 
sudo make install 
```

## 3. cuda安装
略

## 4. Eigen 配置
### 源代码下载
- https://gitlab.com/libeigen/eigen/-/releases/3.4.0
### 解压
- 按以下命令进行安装
```
mkdir build
cd build
cmake ..
sudo make install
//拷贝头文件到系统的用户头文件中，方便后期管理
sudo cp -r /home/【你的路径】/eigen-3.4.0  /usr/local/include/eigen3 
```
### 更新CMakeLists.txt文件（两处）
## 参考项目
- https://github.com/zjhellofss/KuiperLLama
- https://github.com/zjhellofss/KuiperInfer