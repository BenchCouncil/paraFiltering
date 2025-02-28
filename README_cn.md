# paraFiltering
ParaFiltering: A vector retrieval algorithm strategy based on conditional filtering. It incorporates diversified filtering constraints as extra vector dimensions when building the index, uses a hierarchical recall approach during search, and adopts a GPU - optimized method for parallel filtering vector query processing. 

## Install

### Clone the repository

```bash
git clone https://github.com/BenchCouncil/paraFiltering.git
```

### Make Environment


- Python 3.10 and dependencies
  - Create a new conda environment and install dependencies from `requirements.txt`:

```bash
conda create -n parafiltering  python=3.10 -y
conda activate parafiltering
pip3 install -r requirements.txt
```

## Install PyTorch
### 检查cuda版本

```bash
nvidia-smi
# 会显示cuda版本 例如：CUDA Version: 11.2，以及GPU型号
nvcc --version
# 显示cuda版本 例如：Cuda compilation tools, release 11.2, V11.2.152
```

### 参考pytorch与cuda版本对应关系
- https://pytorch.org/get-started/previous-versions/
  
  (仅供参考) pytorch安装指令：
```bash
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

- 成功安装后，测试pytorch是否安装成功：
```bash
python -c "import torch; print(torch.cuda.is_available())"
# 预期 ： True
```

### boost安装
- 用于C++代码链接到python代码，主要作用与第二轮检索的筛选和第三轮检索的排序
```bash
python setup.py build_ext --inplace
```

- 安装boost库
  - boost版本需求为：1.71.0
  - 通过以下指令检查boost版本
  ```bash
  dpkg -l | grep libboost
  ```
  - 如果没有安装boost库，可以通过boost官网下载安装包进行安装
  - 安装boost库
  ```bash
  wget https://archives.boost.io/release/1.71.0/source/boost_1_71_0.tar.gz
  tar -xzvf boost_1_71_0.tar.gz
  cd boost_1_71_0
  ./bootstrap.sh --prefix=/usr/local
  ./b2 cxxflags="-std=c++17 -fopenmp" linkflags="-fopenmp"
  ./b2 install
  ```
  - 安装成功后，可以通过以下指令检查boost库是否安装成功
  ```bash
  ls /usr/local/include/boost
  ls /usr/local/lib | grep "boost_python310\|boost_numpy310"
  ```
  - 清理（可选）
  ```bash
  cd ..
  rm -rf boost_1_71_0 boost_1_71_0.tar.gz
  ```

## 运行
```bash
bash run.sh
```
