# 5-3版本已经完成以下实现：
# 1. PQ+asymmetric distance的实现（L2距离）
# 2. label数据的归一化方法
# 3. GPU的并行实现
# 4. 放弃label的PQ操作，优化融合计算方法
# 5. 重写gpu并发调度逻辑，单次查询batch在cpu上进行，对数据集进行分块后再    （mul GPU 控制 --gpu）
# 6. 取消idx，默认indices作为索引
# 7. 分割label与KNN代码     √（由于分支对gpu影响，还是完整分割了
# 8. 分割gpu与cpu执行代码   （测试中）
# 9. refine方法(python)，逼近精度上限 
# 10. 改进9：访存密集优化，Refine C++实现，扩大召回+Refine效果很好，时间大幅降低（10s）
# 11. C++兼容框架出现问题：函数传递确定类型需要二次确认（强制转换），否则会出现计算错误

# 6-1尝试实现目标：
# 6-2：修改欧几里得距离计算方式 √\sum{(a-b)^2} = a^2 + b^2 - 2ab 在顺序上具有一致性；   假设query为a，计算优化为：b^2 - 2ab，仅需保留b^2，计算点积
# n. 尝试IVF实现，学习使用

import os
import numpy as np
import math
import time
import argparse
import random
from tqdm import tqdm 
import h5py
import warnings
import datetime
import torch.nn as nn
import torch
from concurrent.futures import ThreadPoolExecutor
from PQ_algorithm import (
    l2_distance, asymmetric_distance, pq_training_precomputing, l2_norm_distance
)
from utils import (
    parse_arguments, args_check, set_cuda_info,
    imformation_print, recall_at_k, data_normalization,
)
import boost_filter_refine as bfr

fmt = "\n=====       {:15} =====\n"

def read_hdf5(data: str = "msong-small-1filter-80a"):
    # data_file_path = f"/home/xyzhang/mywork/parafiltering/groundtruth/data/real-datasets/{data}.hdf5"
    data_file_path = f"/HDD03/xyzhang/12-14dataset/{data}.hdf5"


    print("data_file_path: ", data_file_path)
    print("Start reading data from hdf5 file...")

    f = h5py.File(data_file_path, "r")
    train_vec = np.array(f["train_vec"],    dtype = np.float32)
    test_vec =  np.array(f["test_vec"],     dtype = np.float32)
    neighbors = np.array(f["neighbors"],    dtype = np.int32)
    # distances = np.array(f["distances"],    dtype = np.float32)
    train_label = np.array(f["train_label"],    dtype = np.int32)
    test_label  = np.array(f["test_label"],     dtype = np.int32)

    f.close()
    print("Data reading is complete!")

    # return train_vec, test_vec, neighbors, distances, train_label, test_label
    return train_vec, test_vec, neighbors, train_label, test_label



def process_batch_query_with_label(batch_query, pq_data, batch_label, batch_label_scope, data_label, idx_slt, 
                                    sub_device, topk=100, batch_size=4096*10, exp_recall=10, fsc=0.05):
    tmp_dis = []
    tmp_label = []
    tmp_idx = []
    len = batch_query.shape[0]
    batch_query = batch_query.to(sub_device)
    batch_label = batch_label.to(sub_device)
    batch_label_scope = batch_label_scope.to(sub_device)
    # batch_label_scope = (batch_label_scope**(-2.0)).unsqueeze(1)    # (batch, 5) -> (batch, 1, 5)
    pq_data = pq_data.to(sub_device)
    data_label = data_label.to(sub_device)
    idx_slt = idx_slt.to(sub_device)
    
    # | (batch, m) -> (batch, 1, m) | * | (N, m) -> (1, N, m) |  ->  (Batch, 1, m) - (1, N, m) -> (Batch, N, m)
    for batch in torch.split(pq_data, batch_size, dim=0):
        tmp_dis.append(asymmetric_distance(batch_query, batch))
    # del pq_data, batch_query

    # print(f"batch_label: {batch_label.shape}, data_label: {data_label.shape}, batch_label_scope: {batch_label_scope.shape}")
    for batch in torch.split(data_label, batch_size, dim=0):
        # tmp_label.append(l2_distance(batch_label, batch))
        # tmp_label.append(l2_distance(batch_label, batch) * batch_label_scope)
        tmp_label.append(l2_norm_distance(batch_label, batch, batch_label_scope))
    # | (batch, 5) -> (batch, 1, 5) | * | (N, 5) -> (1, N, 5) |  ->  (Batch, 1, 5) - (1, N, 5) -> (Batch, N, 5)
    # del data_label, batch_label

    tmp_dis = torch.cat(tmp_dis, dim=1)
    tmp_label = torch.cat(tmp_label, dim=1)

    # all_dis = tmp_dis + tmp_label * 0.05
    all_dis = tmp_dis + tmp_label * fsc

    _, indices = all_dis.topk(topk * exp_recall, dim=1, largest=False)
    tmp_dis_cpu = torch.gather(tmp_dis, 1, indices).cpu()
    tmp_label_cpu = torch.gather(tmp_label, 1, indices).cpu()
    tmp_idx = torch.gather(idx_slt.unsqueeze(0).repeat(len, 1), 1, indices).cpu()
    return tmp_dis_cpu, tmp_label_cpu, tmp_idx



def distance_calculate_with_label(pq_query, pq_data, idx, 
                                  train_Y, test_Y, test_Y_scope,
                                  db_size, query_size, topk, device, device_ids, num_gpus, 
                                  batch_split=100, batch_size=4096*10, exp_recall=10, fsc=0.05):
    final_dis = []
    final_label = []
    final_idx = []

    # if device == "cuda":
    #     batch_split = 25 * num_gpus

    db_split_size = db_size // (num_gpus * 1) + 1
    for i, (batch_query, batch_label, batch_label_scope) in enumerate(zip(  torch.split(pq_query,   batch_split,    dim=0), \
                                                                            torch.split(test_Y,     batch_split,    dim=0), \
                                                                            torch.split(test_Y_scope, batch_split,  dim=0))):    
        dis_in = []
        dis_in_label = []
        dis_in_idx = []
        # if i % 10 == 0:
        #     print(f"Start time : {datetime.datetime.now()} ; process_batch_query: {format(i * batch_split)}/{format(query_size)}")
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            # for j, (pq_data_sp, train_Y_sp) in enumerate(zip(torch.split(pq_data, db_split_size, dim=0), torch.split(train_Y, db_split_size, dim=0))):
            for j, (pq_data_sp, train_Y_sp, idx_sp) in enumerate(zip(torch.split(pq_data, db_split_size, dim=0), \
                                                                     torch.split(train_Y, db_split_size, dim=0), \
                                                                     torch.split(idx, db_split_size, dim=0))):
                gpu_id = device_ids[j % num_gpus]
                sub_device = f"cuda:{gpu_id}" if device == "cuda" else "cpu"
                futures.append(executor.submit(process_batch_query_with_label, 
                                               batch_query, pq_data_sp, batch_label, batch_label_scope, train_Y_sp, idx_sp, 
                                               sub_device, topk, batch_size, exp_recall, fsc))
            for future in futures:
                tmp_dis, tmp_label, tmp_idx = future.result()
                dis_in.append(tmp_dis)
                dis_in_label.append(tmp_label)
                dis_in_idx.append(tmp_idx)

        design_device = "cuda:0" if device == "cuda" else "cpu"
        dis_in = [dis_in_tensor.to(design_device) for dis_in_tensor in dis_in]
        dis_in_label = [dis_in_label_tensor.to(design_device) for dis_in_label_tensor in dis_in_label]
        dis_in_idx = [dis_in_idx_tensor.to(design_device) for dis_in_idx_tensor in dis_in_idx]
        dis_in = torch.cat(dis_in, dim=1)
        dis_in_label = torch.cat(dis_in_label, dim=1)
        dis_in_idx = torch.cat(dis_in_idx, dim=1)
        # all_dis = dis_in + dis_in_label * 0.05
        all_dis = dis_in + dis_in_label * fsc
        _, indices = all_dis.topk(topk * exp_recall, dim=1, largest=False)
        dis_in = torch.gather(dis_in, 1, indices)
        dis_in_label = torch.gather(dis_in_label, 1, indices)
        dis_in_idx = torch.gather(dis_in_idx, 1, indices)
        final_dis.append(dis_in.cpu())
        final_label.append(dis_in_label.cpu())
        final_idx.append(dis_in_idx.cpu())

    distances = torch.cat(final_dis, dim=0)
    distances_label = torch.cat(final_label, dim=0)
    distances_idx = torch.cat(final_idx, dim=0)
    return distances, distances_label, distances_idx




def process_batch_query(batch_query, pq_data, batch_label, data_label, batch_label_scope, idx_slt, 
                        sub_device, topk=100, batch_size=4096*10, exp_recall=10, fsc=0.05):
    tmp_dis = []
    tmp_label = []
    tmp_idx = []
    len = batch_query.shape[0]
    batch_query = batch_query.to(sub_device)
    batch_label = batch_label.to(sub_device)
    batch_label_scope = batch_label_scope.to(sub_device)
    # batch_label_scope = (batch_label_scope**(-2.0)).unsqueeze(1)    # (batch, 5) -> (batch, 1, 5)
    pq_data = pq_data.to(sub_device)
    data_label = data_label.to(sub_device)
    idx_slt = idx_slt.to(sub_device)
    
    # | (batch, m) -> (batch, 1, m) | * | (N, m) -> (1, N, m) |  ->  (Batch, 1, m) - (1, N, m) -> (Batch, N, m)
    for batch in torch.split(pq_data, batch_size, dim=0):
        tmp_dis.append(asymmetric_distance(batch_query, batch))
    # del pq_data, batch_query

    for batch in torch.split(data_label, batch_size, dim=0):
        tmp_label.append(l2_norm_distance(batch_label, batch, batch_label_scope))
    # | (batch, 5) -> (batch, 1, 5) | * | (N, 5) -> (1, N, 5) |  ->  (Batch, 1, 5) - (1, N, 5) -> (Batch, N, 5)

    tmp_dis = torch.cat(tmp_dis, dim=1)
    tmp_label = torch.cat(tmp_label, dim=1)

    # all_dis = tmp_dis + tmp_label * 0.05
    all_dis = tmp_dis + tmp_label * fsc

    _, indices = all_dis.topk(topk * exp_recall, dim=1, largest=False)
    tmp_dis_cpu = torch.gather(tmp_dis, 1, indices).cpu()
    tmp_label_cpu = torch.gather(tmp_label, 1, indices).cpu()
    tmp_idx = torch.gather(idx_slt.unsqueeze(0).repeat(len, 1), 1, indices).cpu()
    return tmp_dis_cpu, tmp_label_cpu, tmp_idx




def distance_calculate(pq_query, pq_data, idx, 
                        train_Y, test_Y, test_Y_scope,
                        db_size, query_size, topk, device, device_ids, num_gpus, 
                        batch_split=100, batch_size=4096*10, exp_recall=10, fsc=0.05):
    
    final_dis = []
    final_label = []
    final_idx = []

    # batch_split = 100
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for i, (pq_query_sp, test_Y_sp, test_Y_scope_sp) in enumerate(zip(  torch.split(pq_query,  batch_split, dim=0), \
                                                                            torch.split(test_Y,    batch_split, dim=0), \
                                                                            torch.split(test_Y_scope, batch_split,  dim=0))):
            gpu_id = device_ids[i % num_gpus]
            sub_device = f"cuda:{gpu_id}" if device == "cuda" else "cpu"
            futures.append(executor.submit(process_batch_query,
                                           pq_query_sp, pq_data, test_Y_sp, train_Y, test_Y_scope_sp, idx, 
                                           sub_device, topk, batch_size, exp_recall, fsc))            
        for future in futures:
            tmp_dis, tmp_label, tmp_idx = future.result()
            final_dis.append(tmp_dis)
            final_label.append(tmp_label)
            final_idx.append(tmp_idx)
    
    design_device = "cuda:0" if device == "cuda" else "cpu"
    final_dis = [dis_in_tensor.to(design_device) for dis_in_tensor in final_dis]
    final_label = [dis_in_label_tensor.to(design_device) for dis_in_label_tensor in final_label]
    final_idx = [dis_in_idx_tensor.to(design_device) for dis_in_idx_tensor in final_idx]
    final_dis = torch.cat(final_dis, dim=0)
    final_label = torch.cat(final_label, dim=0)
    final_idx = torch.cat(final_idx, dim=0)
    # all_dis = final_dis + final_label * 0.05
    all_dis = final_dis + final_label * fsc
    _, indices = all_dis.topk(topk * exp_recall, dim=1, largest=False)
    final_dis = torch.gather(final_dis, 1, indices)
    final_label = torch.gather(final_label, 1, indices)
    final_idx = torch.gather(final_idx, 1, indices)
    return final_dis, final_label, final_idx




if __name__ == "__main__":
    print(f"All Process Start Time : {datetime.datetime.now()}")
    args = parse_arguments()
    args_check(args)
    data_name, subv_dim, num_codebook, topk, ExpRecall = args.data, args.subv_dim, args.num_codebook, args.topk, args.exp_recall
    # truly data in original dataset
    train_X, test_X, neighbors, train_label, test_label = read_hdf5(data_name)
    query_size  = test_X.shape[0]
    db_size     = train_X.shape[0]
    dim         = train_X.shape[1]
    sub_num     = dim // subv_dim
    # subv_dim    = dim // sub_num
    assert sub_num * subv_dim == dim
    # imformation_print(data_type, topk, query_size, db_size, dim, subv_dim, num_codebook, sub_num, ExpRecall)
    device, device_ids, num_gpus = set_cuda_info(device = args.device, gpu_num = args.gpu_num)

    label_dim = test_label.shape[1]
    test_label_center = np.array((test_label[:, :, 0] + test_label[:, :, 1]) * 0.5, dtype=np.float32)
    # test_label_center = (test_label[:, :, 0] + test_label[:, :, 1]) * 0.5
    test_label_scope = np.abs(test_label[:, :, 1] - test_label[:, :, 0], dtype=np.float32)

    pq_data, pq_query, idx, Precomputing = pq_training_precomputing(
        train_X, test_X, db_size, 
        dim, subv_dim, num_codebook, 
        # device, device_ids
    )


    # fsc = args.fsc
    batch_split = 200
    batch_size = 4096*10
    # for fsc in [0.035, 0.1, 0.35, 1, 3.5, 10, 35, 100, 1000]:
    for fsc in [0.035]:
        print(fmt.format(f"\n\n\nfusion coefficient: {fsc}"))
        for ExpRecall in [1, 5, 10, 15, 30, 50]:
            s = time.time()
            
            if db_size>=2000000:
                distances, distances_label, distances_idx = distance_calculate_with_label(
                    pq_query, pq_data, idx,
                    torch.from_numpy(train_label), torch.from_numpy(test_label_center), torch.from_numpy(test_label_scope),
                    db_size, query_size, topk, device, device_ids, num_gpus,
                    batch_split=batch_split, batch_size=batch_size, exp_recall=ExpRecall, fsc = fsc
                )
            else:
                distances, distances_label, distances_idx = distance_calculate(
                    pq_query, pq_data, idx,
                    torch.from_numpy(train_label), torch.from_numpy(test_label_center), torch.from_numpy(test_label_scope),
                    db_size, query_size, topk, device, device_ids, num_gpus,
                    batch_split=batch_split, batch_size=batch_size, exp_recall=ExpRecall, fsc = fsc
                )
                    

            # idx_ = idx.cpu().numpy()
            distances = distances.cpu().numpy()
            distances_idx = distances_idx.cpu().numpy()

            # idx_ = idx_.reshape(-1)
            distances = distances.reshape(query_size, -1)
            distances_idx = distances_idx.reshape(query_size, -1)

            # idx_ = np.array(idx_, dtype=np.int32)
            distances = np.array(distances, dtype=np.float32)
            distances_idx = np.array(distances_idx, dtype=np.int32)

            es = time.time()
            Searching = es - s
            if(ExpRecall==1):
                print("\nNamespace(ExpRecall={}, filter={}, refine={}, subv_dim={}, num_codebook={}, data={})".format(1, 0, 0, subv_dim, num_codebook, data_name))
                recall_at_k(distances_idx, neighbors, topk, num_codebook, dim, subv_dim)  
                # print("Searching: {} s".format(Searching))
                print(f"Searching QPS: {query_size / (Searching + Precomputing)}")  
                print("(expRecall) Searching Time: ", Searching)
                print("Round 1 End Time : ", Searching + Precomputing)

            for cc in [1, 3, 5, 7]:
                if cc >= ExpRecall:
                    break

                print("\nNamespace(ExpRecall={}, filter={}, refine={}, subv_dim={}, num_codebook={}, data={})".format(ExpRecall, cc, 0, subv_dim, num_codebook, data_name))
                
                label_test_left     = np.array(test_label[:, :, 0], dtype=np.int32)
                label_test_right    = np.array(test_label[:, :, 1], dtype=np.int32)

                select_idx, select_dis = bfr.process_mul(
                    distances_idx, distances,
                    label_test_left, label_test_right, train_label,
                    int(query_size), int(topk), int(cc), int(label_dim)
                )
        
                e2 = time.time()
                FIlter_time = e2 - es
                # print("FIlter_time: {} s".format(FIlter_time))
                recall_at_k(select_idx, neighbors, topk, num_codebook, dim, subv_dim)
                print("Filtering QPS: {}".format(query_size / (FIlter_time + Searching + Precomputing)))
                print("Filtering Searching Time: ", FIlter_time)
                print("Round 2 End Time : ", FIlter_time + Searching + Precomputing)



                print("\nNamespace(ExpRecall={}, filter={}, refine={}, subv_dim={}, num_codebook={}, data={}".format(ExpRecall, cc, 1, subv_dim, num_codebook, data_name))

                s = time.time()    

                refine_idx = bfr.Refine(
                    select_idx, train_X, test_X, 
                    int(topk)
                )

                e = time.time()
                Refine_time = e - s
                # print("Refine_time: {} s".format(Refine_time))
                recall_at_k(refine_idx, neighbors, topk, num_codebook, dim, subv_dim)
                print("Refine QPS: {}".format(query_size / (Refine_time + Precomputing + Searching + FIlter_time)))
                print("Refine Searching Time: ", Refine_time)
                print("Round 3 End Time : ", Refine_time + Precomputing + Searching + FIlter_time)

    print("END Time : ", datetime.datetime.now())