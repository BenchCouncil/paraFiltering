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
### Check cuda version

```bash
nvidia-smi
# The cuda version will be displayed, for example: CUDA Version: 11.2, and the GPU model
nvcc --version
# Display cuda versions, for example: Cuda compilation tools, release 11.2, V11.2.152
```

### Reference pytorch and cuda version correspondence
- https://pytorch.org/get-started/previous-versions/
  
  (For reference only) pytorch installation instructions:
```bash
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

- After a successful installation, test whether PyTorch is successfully installed.
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Expect : True
```

### Boost
- Used to link C++ code to Python code, the main role is to filter the second round of search and sort the third round of search
```bash
python setup.py build_ext --inplace
```
## run
```bash
bash run.sh
```
