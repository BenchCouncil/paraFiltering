import numpy as np
import torch
import time
import datetime
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="mul", help="Specify the data file path(e.g.,'ann' or 'label' or 'mul')")
    parser.add_argument("--device", type=str, default="cuda", help="Specify the device (e.g., 'cpu' or 'cuda')")
    parser.add_argument("--gpu_num", type=int, default=1, help="Specify the number of GPU, if more than, the number will be set to 5")
    parser.add_argument("--subv_dim", type=int, default=8, help="The dimension of the subvectors")
    # parser.add_argument("--m", type=int, default=8, help="Specify the sub vector number of each vector")
    parser.add_argument("--num_codebook", type=int, default=256, help="Specify the number of codebook")
    parser.add_argument("--topk", type=int, default=100, help="Specify the number of nearest neighbors")
    parser.add_argument("--is_shift", type=bool, default=True, help="Specify the shift time")
    parser.add_argument("--exp_recall", type=int, default=10, help="Expanded the recall scope")
    parser.add_argument("--fsc", type=float, default=0.035, help="Specify the fusion coefficient")
    args = parser.parse_args()
    return args


def imformation_print(data_type, topk, query_size, db_size, dim, subv_dim, num_codebook, sub_num, ExpRecall):
    print("data type : {}".format(data_type))
    print("TOP k     : {}".format(topk))
    print("query_size: {}, db_size:  {}".format(query_size, db_size))
    print("demension : {}, subv_dim: {}".format(dim, subv_dim))
    print("codebook  : {}".format(num_codebook))
    print("sub_num(m): {}".format(sub_num))
    print("ExpRecall : {}".format(ExpRecall))
    return


def set_cuda_info(device:str = "cuda", gpu_num:int = 5):
    count = torch.multiprocessing.cpu_count()
    print("set cpu count : {}".format(count))
    torch.set_num_threads(count)
    if torch.cuda.is_available() and device == "cuda":
        device = "cuda"
        # device_ids = list(range(torch.cuda.device_count()))
        device_ids = ['0', '1', '2', '3', '4', '5', '6', '7']   # 0-7 : V100-16G;
        num_gpus = min(gpu_num, len(device_ids))
        device_ids = device_ids[:num_gpus]
    else:
        device = "cpu"
        num_gpus = 1
        device_ids = ['0']
    print("cuda available:  {}".format(torch.cuda.is_available()))
    print("device:          {}".format(device))
    if device == "cuda":
        print("device_ids:      {}".format(device_ids))
    return device, device_ids, num_gpus


def args_check(args):
    assert args.device in ["cpu", "cuda"]
    # print(f"args: {args}")
    print("Namespace(sub_num: {}, num_codebook: {}, data: {}, device: {}, gpu_num: {}, topk: {}, is_shift: {}, fsc: {})".format(
        args.subv_dim, args.num_codebook, args.data, args.device, args.gpu_num, args.topk, args.is_shift, args.fsc))
    return


def data_normalization(train_label, test_label, left: float, right: float, is_shift: bool = True, is_time: bool = False):
    train_Y = np.array(train_label, dtype = float)
    test_Y = np.array(test_label, dtype = float)
    MAX_train = np.max(train_label)
    MIN_train = np.min(train_label)
    MAX_test = np.max(test_label)
    MIN_test = np.min(test_label)
    Max = max(MAX_train, MAX_test)
    Min = min(MIN_train, MIN_test)
    if is_time:
        Max_view = (np.datetime64('1970-01-01') + np.timedelta64(int(Max), 's')).astype(str)
        Min_view = (np.datetime64('1970-01-01') + np.timedelta64(int(Min), 's')).astype(str)
        print("Max:  {:<10}, Min:  {:<10}".format(Max_view, Min_view))
    else:
        print("Max:  {:<10}, Min:  {:<10}".format(Max, Min))
    div = left + right
    coff = 2.0 / div
    shift_time = (left - right) if is_shift else 0
    # print("coff: ", coff)
    # print("shift_time: ", shift_time)
    train_Y = (train_label - Min) * coff
    test_Y = (test_label - Min - shift_time) * coff
    for i in range(len(test_Y)):
        if test_Y[i] < 0:
            test_Y[i] = 0
    return train_Y, test_Y


def recall_at_k(idx:np.ndarray, neighbors:np.ndarray, 
                topk:int = 100, num_centers:int = 256, 
                dim:int = 384, subv_dim:int = 8):
    # num = 0
    # hit = 0
    # id_num = 0
    # try:
    #     pre_select = len(idx[0])
    # except:
    #     pre_select = 0
    # for id, nb in zip(idx, neighbors[:, :topk]):    
    #     num += len(set(nb) - set([-1]))             # remove -1
    #     id_num += len(set(id) - set([-1]))
    #     hit += len(set(nb) & set(id) - set([-1]))
    # print(f"hit num : {hit}, total num : {num}, id has num : {id_num}")
    # print("Pre-select in {} for top {} hit rate: {} ".format(pre_select, topk, hit / num))
    hit = 0
    num = 0
    id_num = 0
    for id, nb in zip(idx[:, :topk], neighbors[:, :topk]):
        num += len(set(nb) - set([-1]))             # remove -1
        id_num += len(set(id) - set([-1]))
        hit += len(set(nb) & set(id) - set([-1]))
    # print(f"hit num : {hit}, total num : {num}, id has num : {id_num}")
    # print("Recall: {} [M={}, centers={} @ top {}]".format(hit / num, dim//subv_dim, num_centers, topk))
    print(f"Recall: {hit / num} [hit num: {hit}, total num: {num}, id has num: {id_num}]")

    return
