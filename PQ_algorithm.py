import numpy as np
import warnings
import datetime
import torch.nn as nn
import torch
from concurrent.futures import ThreadPoolExecutor
import time

def l2_distance(obs: torch.Tensor, codebook: torch.Tensor):
    ''' attention need!!! without sqrt, not real distance, just for comparison '''
    ''' obs:(N,D) - codebook:(k,D)  ---extend--->  (N,1,D) - (1,K,D) ---->  
        (N,K,D) : each vector N & each center K --> quare(**2) & sum in D dimention
        calculate approximate distance in demension D  --->  (N,K)  '''
    dis = ((obs.unsqueeze(dim=1) - codebook.unsqueeze(dim=0)) ** 2.0).sum(dim=-1).squeeze()
    return dis

def l2_norm_distance(obs: torch.Tensor, codebook: torch.Tensor, norm: torch.Tensor):
    ''' attention need!!! without sqrt, not real distance, just for comparison '''
    ''' obs:(N,D) - codebook:(k,D)  ---extend--->  (N,1,D) - (1,K,D) ---->  
        (N,K,D) : each vector N & each center K --> quare(**2) & sum in D dimention
        calculate approximate distance in demension D  --->  (N,K)  '''
    dis = ((obs.unsqueeze(dim=1) - codebook.unsqueeze(dim=0)) / norm.unsqueeze(dim=1) ** 2.0).sum(dim=-1).squeeze() # 除以norm
    return dis

def _kmeans_batch(obs: torch.Tensor, 
                  k: int, device,
                  distance_function,
                  batch_size=0,
                #   thresh=1e-1,
                  thresh=1,
                  norm_center=False):
    '''return codebook, distance'''
    codebook = obs[torch.randperm(obs.size(0))[:k]].clone()
    history_distances = [float('inf')]
    if batch_size == 0:
        batch_size = obs.shape[0]
    while True:
        # (N x D, k x D) -> N x k
        segs = torch.split(obs, batch_size)
        seg_center_dis = []
        seg_center_ids = []
        for seg in segs:
            distances = distance_function(seg, codebook)
            center_dis, center_ids = distances.min(dim=1)
            seg_center_ids.append(center_ids)
            seg_center_dis.append(center_dis)
        obs_center_dis_mean = torch.cat(seg_center_dis).mean()
        obs_center_ids = torch.cat(seg_center_ids)
        history_distances.append(obs_center_dis_mean.item())
        diff = history_distances[-2] - history_distances[-1]
        if diff < thresh:
            if diff < 0:
                warnings.warn("Distance diff < 0, distances: " + ", ".join(map(str, history_distances)))
            break
        for i in range(k):
            obs_id_in_cluster_i = obs_center_ids == i
            if obs_id_in_cluster_i.sum() == 0:
                continue
            obs_in_cluster = obs.index_select(0, obs_id_in_cluster_i.nonzero().squeeze())
            c = obs_in_cluster.mean(dim=0)
            if norm_center:
                c /= c.norm()
            codebook[i] = c
    return codebook, history_distances[-1]


def kmeans(obs: torch.Tensor, k: int,
           device = 'cuda',
           distance_function=l2_distance,
           iter=20,
           batch_size=0,
        #    thresh=1e-1,
           thresh=1,
           norm_center=False):
    '''return codebook, distance'''
    best_distance = float("inf")
    # best_codebook = None
    for i in range(iter):
        if batch_size == 0:
            batch_size = obs.shape[0]  # Fix: Assign the value of obs.shape[0] to batch_size
        codebook, distance = _kmeans_batch(obs, k, device,
                                          norm_center=norm_center,
                                          distance_function=distance_function,
                                          batch_size=batch_size,
                                          thresh=thresh)
        if distance < best_distance:
            best_codebook = codebook
            best_distance = distance
    return best_codebook, best_distance


def process_sub_data(data, i, sub_vector_size, dim, k, device, kwargs):
    if i // sub_vector_size % 12 == 0:
        print(f"Start time : {datetime.datetime.now()} ; product_quantization: {format(i)}/{format(dim)}")
    sub_data = data[:, i:min(i + sub_vector_size, dim)].to(device)
    sub_codebook, _ = kmeans(sub_data, k=k, device=device, **kwargs)
    return sub_codebook.cpu()


def product_quantization(data: torch.Tensor, dim: int, 
                         sub_vector_size: int, k: int, 
                         device: str = "cuda",
                         device_ids: list = ['0', '1', '2', '3', '4', '5', '6', '7'],
                         **kwargs):
    '''default use [0, 1, 2, 3, 4, 5, 6, 7] cuda devices, return codebook (m, k, d)'''
    tasks = []
    codebook = []
    # # --------------------- 测试代码 ---------------------  start
    # # 测试阶段 trainning 强制使用5块卡跑
    # device = "cuda"
    # device_ids = ['0', '1', '2', '3', '4']
    # # --------------------- 测试阶段 ---------------------  end
    num_works = len(device_ids) if device == "cuda" else 1
    with ThreadPoolExecutor(max_workers=num_works) as executor:
        for i in range(0, dim, sub_vector_size):
            gpu_id = device_ids[i // sub_vector_size % num_works]
            sub_device = f"cuda:{gpu_id}" if device == "cuda" else "cpu"        # CPU: 执行cpu， cuda：执行多卡训练
            tasks.append(executor.submit(process_sub_data, data, i, sub_vector_size, dim, k, sub_device, kwargs))
        for future in tasks:
            codebook.append(future.result())    # 将每个线程的结果添加到codebook中
    return codebook                             # (dim/sub_vector_size, k, sub_vector_size) -> (m, k, d) -> (48, 256, 8)


def data_to_pq(data: torch.Tensor, 
               centers: torch.Tensor, 
               device : str = "cuda", 
               batch_size = 4096*10):
    '''return (N, K)'''
    assert (len(centers) > 0)
    assert (data.shape[1] == sum([c.shape[1] for c in centers]))
    m = len(centers)
    syb_size = centers[0].shape[1]
    ret = torch.zeros(data.shape[0], m, dtype=torch.uint8, device=device)

    for idx, sub_vec in enumerate(torch.split(data, syb_size, dim=1)):
        idx = torch.tensor(idx)  # Convert idx to a tensor
        if device == "cuda":
            sub_vec = sub_vec.cuda()
        segs = torch.split(sub_vec, batch_size)
        seg_center_dis = []
        for seg in segs:
            dis = l2_distance(seg, centers[idx])
            seg_center_dis.append(dis.argmin(dim=1).to(dtype=torch.uint8))
        ret[:, idx] = torch.cat(seg_center_dis)
    return ret.cpu()
    

def asymmetric_table(query, centers, device = "cuda"):
    m = len(centers)
    sub_size = centers[0].shape[1]
    ret = torch.zeros(
        query.shape[0], m, centers[0].shape[0],
        # device = "cpu")
        device=device)
    assert (query.shape[1] == sum([cb.shape[1] for cb in centers]))
    for i, offset in enumerate(range(0, query.shape[1], sub_size)):
        sub_query = query[:, offset: offset + sub_size]
        ret[:, i, :] = l2_distance(sub_query, centers[i])
    return ret


def asymmetric_distance_slow(asymmetric_tab, pq_data):
    ret = torch.zeros(asymmetric_tab.shape[0], pq_data.shape[0])
    for i in range(asymmetric_tab.shape[0]):
        for j in range(pq_data.shape[0]):
            dis = 0
            for k in range(pq_data.shape[1]):
                sub_dis = asymmetric_tab[i, k, pq_data[j, k].item()]
                dis += sub_dis
            ret[i, j] = dis
    return ret


def asymmetric_distance(asymmetric_tab, pq_data):
    pq_db = pq_data.long()
    dd = [torch.index_select(asymmetric_tab[:, i, :], 1, pq_db[:, i]) for i in range(pq_data.shape[1])]
    return sum(dd)


# --------------------- 训练代码 强制GPU ---------------------  start
def pq_training_precomputing(train_X: np.ndarray, test_X: np.ndarray, db_size: int,
                             dim: int = 384, subv_dim: int = 48, num_codebook: int = 256, 
                             device: str = "cuda", device_ids: list = ['0', '1', '2', '3', '4', '5', '6', '7']):
    '''for PQ training(Dataset) and precomputing(Query to pq_query)'''
    s = time.time()
    training_data = torch.from_numpy(train_X)
    codebook = product_quantization(
        data = training_data, dim = dim,
        sub_vector_size = subv_dim, k = num_codebook, 
        device = device, device_ids = device_ids,
        batch_size = 4096*10 , iter = 3
    )
    print(f"PQ end time : {datetime.datetime.now()}")
    if device == "cuda":
        torch.cuda.synchronize()
        codebook = [i.cuda() for i in codebook]
    codebook = torch.stack(codebook)
    print(f"centers shape: {codebook.shape}")
    pq_data = data_to_pq(training_data, codebook, device)
    print("pq_data shape: {}".format(pq_data.shape))
    print(f"trainning end time : {datetime.datetime.now()}")
    e = time.time()
    Training = e - s
    print("Training: {} s".format(Training))

    s = time.time()
    query = torch.from_numpy(test_X).to(device)
    pq_query = asymmetric_table(query, codebook, device)
    print("pq_query shape: {}".format(pq_query.shape))
    ep = time.time()
    Precomputing = ep - s
    print("Precomputing: {} s".format(Precomputing))

    if device == "cuda":
        torch.cuda.synchronize()
        
    idx = torch.from_numpy(np.arange(db_size))

    return pq_data, pq_query, idx, Precomputing
# --------------------- 测试代码 强制GPU ---------------------  end