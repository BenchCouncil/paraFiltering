#!/bin/bash

# 下载 Boost 源码
wget https://archives.boost.io/release/1.71.0/source/boost_1_71_0.tar.gz

# 解压源码包
tar -xzvf boost_1_71_0.tar.gz

# 进入解压后的目录
cd boost_1_71_0

# 配置 Boost 安装路径
./bootstrap.sh --prefix=/usr/local

# 配置编译选项
./b2 cxxflags="-std=c++17 -fopenmp" linkflags="-fopenmp"

# 安装 Boost
./b2 install

# 清理
cd ..
rm -rf boost_1_71_0 boost_1_71_0.tar.gz

echo "Boost 1.71.0 安装完成！"