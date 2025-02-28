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

### Install PyTorch
- Check cuda version

```bash
nvidia-smi
# The cuda version will be displayed, for example: CUDA Version: 11.2, and the GPU model
nvcc --version
# Display cuda versions, for example: Cuda compilation tools, release 11.2, V11.2.152
```

- Reference pytorch and cuda version correspondence
  - https://pytorch.org/get-started/previous-versions/
  
- (For reference only) pytorch installation instructions:
```bash
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

- After a successful installation, test whether PyTorch is successfully installed.
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Expect : True
```

### Install Boost
- Used to link C++ code to Python code, the main role is to filter the second round of search and sort the third round of search
- After installing the boost library
```bash
python src/setup.py build_ext --inplace
```

- Install the boost library
  - Boost version requirements are: 1.71.0
  - Check the boost version with the following instructions
  ```bash
  dpkg -l | grep libboost
  ```
  - If the boost library is not installed, you can execute the following command to install it
  ```bash
  bash install_boost.sh
  ```

## run
```bash
bash run.sh
```

## Configuration
- `--data` : The path of the dataset file
- `--device` : The device used for training, default is `cuda`
- `--gpu_num` : The number of GPUs used for training, default is `1`
- `--subv_dim` : The dimension of the subvectors, default is `8`
- `--num_codebook` : The number of codebooks, default is `256`
- `--topk` : The number of topk, default is `100`



## Data Preparation
### Benchmark Datasets
- Download the benchmark datasets from the following link:
  - [BigVectorBench](https://github.com/BenchCouncil/BigVectorBench)
    | Dataset | Data / Query Points |  Dimension  | Filtering ratio |  Labels |   Distance |  Download | Raw Data |
    | :------: | :---: | :---: | :---: | :---: | :---: | :---: |  :---: | 
    | ag_news-384-euclidean-filter | 120,000 / 7,600 | 384 | 25% | 1 | Euclidean | [HDF5](https://huggingface.co/datasets/Patrickcode/BigVectorBench/resolve/main/ag_news-384-euclidean-filter.hdf5)(208M) | [BVB](https://github.com/BenchCouncil/BigVectorBench?tab=readme-ov-file) |
    | cc_news-384-euclidean-filter | 620,643 / 10,000 | 384 | < 10% | 1 | Euclidean | [HDF5](https://huggingface.co/datasets/Patrickcode/BigVectorBench/resolve/main/cc_news-384-euclidean-filter.hdf5)(994M) | [BVB](https://github.com/BenchCouncil/BigVectorBench?tab=readme-ov-file) |
    | app_reviews-384-euclidean-filter | 277,936 / 10,000 | 384 | < 1% | 3 | Euclidean | [HDF5](https://huggingface.co/datasets/Patrickcode/BigVectorBench/resolve/main/app_reviews-384-euclidean-filter.hdf5)(457M) | [BVB](https://github.com/BenchCouncil/BigVectorBench?tab=readme-ov-file) |
    | amazon-384-euclidean-5filter|	15,928,208 / 10,000 | 384 | < 0.1% | 5 | Euclidean | [HDF5](https://huggingface.co/datasets/Patrickcode/BigVectorBench/resolve/main/amazon-384-euclidean-5filter.hdf5)(24.8G) | [BVB](https://github.com/BenchCouncil/BigVectorBench?tab=readme-ov-file) |


- The manually generated dataset, the data format is consistent with BigVectorBench, and the dataset download link is as follows:
  - [paraFiltering](https://huggingface.co/datasets/1253828034HJP/paraFiltering)
    | Dataset | Data / Query Points |  Dimension  | Filtering ratio |  Labels |   Distance |  Download | Raw Data |
    | :------: | :---: | :---: | :---: | :---: | :---: | :---: |  :---: | 
    | msong-1filter-80a.hdf5  | 990,000 / 10,000 | 420 | 80%| 1  | Euclidean | [HDF5](https://huggingface.co/datasets/1253828034HJP/paraFiltering/blob/main/msong-1filter-80a.hdf5)(3.4G) | [SIFT](http://corpus-texmex.irisa.fr/) |
    | deep1M-2filter-50a.hdf5 | 1,000,000 / 10,000 | 256 | 50% | 2 | Euclidean | [HDF5](https://huggingface.co/datasets/1253828034HJP/paraFiltering/blob/main/deep1M-2filter-50a.hdf5)(2.12G) | [DEEP1M](https://github.com/matsui528/deep1b_gt) |
    | tiny5m-6filter-12a.hdf5 | 5,000,000 / 10,000 | 384 | 12% | 6 | Euclidean | [HDF5](https://huggingface.co/datasets/1253828034HJP/paraFiltering/blob/main/tiny5m-6filter-12a.hdf5)(15.7G) | [TINY5M](https://huggingface.co/timm/tiny_vit_5m_224.in1k) |
    | sift10m-6filter-6a.hdf5 | 10,000,000  / 10,000 | 128 | 6% | 6 | Euclidean | [HDF5](https://huggingface.co/datasets/1253828034HJP/paraFiltering/blob/main/sift10m-6filter-6a.hdf5)(10.8G) | [SIFT10M](https://www.bing.com/search?q=sift10m&qs=n&form=QBRE&sp=-1&lq=0&pq=sift10m&sc=17-7&sk=&cvid=1324959AF80D4796A7EA1C1A39F0FA97&ghsh=0&ghacc=0&ghpl=) |


    




