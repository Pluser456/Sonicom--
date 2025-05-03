import os
from re import S
import h5py
import numpy as np
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist

def split_dataset(image_dir: str, hrtf_dir: str, test_indices: list = None) -> dict:
    """
    将数据集分割为训练集和测试集
    
    Args:
        image_dir (str): 图像目录路径
        hrtf_dir (str): HRTF文件目录路径
        test_indices (list, optional): 测试集索引列表. 默认为None时使用预定义的索引
        
    Returns:
        dict: 包含训练集和测试集路径的字典
    """
    cwd = os.path.dirname(os.path.realpath(__file__))

    parent_dir = os.path.dirname(cwd)
    image_dir = os.path.join(parent_dir, image_dir)
    hrtf_dir = os.path.join(parent_dir, hrtf_dir)
    if test_indices is None:
        test_indices = [7, 14, 27, 30, 31, 52, 54, 55, 70, 82, 143, 184]
        # print(sorted(np.random.choice(190, size=12, replace=False)))
    
    # 获取并排序图像列表
    image_list = os.listdir(image_dir)
    image_list.sort(key=lambda x: int(x.split('.')[0].split('_')[0][1:]))
    
    # 分离左右耳图像
    left_image_list = [img for img in image_list if img.endswith('left.png')]
    right_image_list = [img for img in image_list if img.endswith('right.png')]
    
    # 分割训练集和测试集
    left_train = [x for i, x in enumerate(left_image_list) if i not in test_indices]
    right_train = [x for i, x in enumerate(right_image_list) if i not in test_indices]
    left_test = [x for i, x in enumerate(left_image_list) if i in test_indices]
    right_test = [x for i, x in enumerate(right_image_list) if i in test_indices]
    
    # 从图像名称中提取编号
    train_image_numbers = [int(img.split('_')[0][1:]) for img in left_train]
    test_image_numbers = [int(img.split('_')[0][1:]) for img in left_test]
    
    # 过滤HRTF文件列表
    train_hrtf_list = [x for x in os.listdir(hrtf_dir) if int(x.split('.')[0][1:]) in train_image_numbers]
    train_hrtf_list = [os.path.join(hrtf_dir, f) for f in train_hrtf_list if os.path.isfile(os.path.join(hrtf_dir, f))]
    
    test_hrtf_list = [x for x in os.listdir(hrtf_dir) if int(x.split('.')[0][1:]) in test_image_numbers]
    test_hrtf_list = [os.path.join(hrtf_dir, f) for f in test_hrtf_list if os.path.isfile(os.path.join(hrtf_dir, f))]
    
    # 获取完整路径
    left_train = [os.path.join(image_dir, img) for img in left_train]
    right_train = [os.path.join(image_dir, img) for img in right_train]
    left_test = [os.path.join(image_dir, img) for img in left_test]
    right_test = [os.path.join(image_dir, img) for img in right_test]
    
    return {
        'train_hrtf_list': train_hrtf_list,
        'test_hrtf_list': test_hrtf_list,
        'left_train': left_train,
        'right_train': right_train,
        'left_test': left_test,
        'right_test': right_test
    }

def calculate_hrtf_mean(hrtf_file_names, whichear=None):
    # 初始化累加器和计数器
    hrtf_sum = None
    total_samples = 0
    
    for file_path in hrtf_file_names:
        with h5py.File(file_path, 'r') as data:
            # 读取当前文件所有位置的HRTF数据
            hrtfs = data[f'F_{whichear}'][:]  # 形状为 (num_positions, num_freq_bins)
            
            # 如果是第一次读取，初始化累加器
            if hrtf_sum is None:
                hrtf_sum = np.zeros(hrtfs.shape, dtype=np.float64)
            
            # 累加当前文件所有位置的HRTF
            hrtf_sum += hrtfs
            total_samples += 1
    
    # 计算全局平均
    hrtf_mean = hrtf_sum / total_samples
    return hrtf_mean  # 保持与原始数据相同的精度

def train_one_epoch(model, optimizer, data_loader, device, epoch, rank=0):
    model.train()
    loss_function = nn.MSELoss()
    accu_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # 仅在主进程显示进度条
    if rank == 0:
        data_loader = tqdm(data_loader, file=sys.stdout)
    else:
        data_loader = data_loader

    for step, sample_batch in enumerate(data_loader):
        # Extract data (ensure keys match your DataLoader output)
        left_image = sample_batch["left_image"]
        right_image = sample_batch["right_image"]
        pos = sample_batch["position"]
        hrtf = sample_batch["hrtf"]

        # Model now returns prediction and target
        mu, target_y_sel = model(left_image, right_image, pos, hrtf, device=device, is_training=True, auxiliary_data=None)

        # Ensure target is on the correct device
        target_y_sel = target_y_sel.to(device)

        # Handle cases where model returns zero tensors due to insufficient points
        if mu.shape[0] == 0:
            print(f"Warning: Skipping step {step} in epoch {epoch} due to zero points.")
            continue

        loss = loss_function(mu, target_y_sel)

        accu_loss += loss.detach()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        if rank == 0:
            data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        optimizer.step()
        optimizer.zero_grad()

    # Sync loss across GPUs if using distributed training
    if dist.is_initialized():
        dist.all_reduce(accu_loss, op=dist.ReduceOp.SUM)
        accu_loss = accu_loss / dist.get_world_size()

    num_steps = step + 1
    final_loss = accu_loss.item() / num_steps if num_steps > 0 else 0.0

    return final_loss

def evaluate(model, data_loader, device, epoch, rank=0, auxiliary_loader=None):
    model.eval()
    loss_function = nn.MSELoss()
    accu_loss = torch.zeros(1).to(device)

    if auxiliary_loader is None:
        raise ValueError("Auxiliary loader must be provided for evaluation")

    try:
        auxiliary_batch = next(iter(auxiliary_loader))
    except StopIteration:
        print("Error: Auxiliary loader is empty.")
        return 0.0

    if rank == 0:
        data_loader = tqdm(data_loader, file=sys.stdout)

    step_count = 0
    with torch.no_grad():
        for step, sample_batch in enumerate(data_loader):
            step_count += 1
            left_image = sample_batch["left_image"]
            right_image = sample_batch["right_image"]
            pos = sample_batch["position"]
            hrtf = sample_batch["hrtf"]

            mu, _ = model(left_image, right_image, pos, hrtf, device=device, is_training=False, auxiliary_data=auxiliary_batch)

            target = hrtf.to(device)

            loss = loss_function(mu, target)
            accu_loss += loss.detach()

            if rank == 0:
                data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / step_count)

    if dist.is_initialized():
        dist.all_reduce(accu_loss, op=dist.ReduceOp.SUM)
        accu_loss = accu_loss / dist.get_world_size()

    final_loss = accu_loss.item() / step_count if step_count > 0 else 0.0
    return final_loss