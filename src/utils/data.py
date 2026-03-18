"""
数据处理工具函数
"""
import os
import h5py
import numpy as np
from typing import Dict, List, Optional


def split_dataset(
    voxel_dir: str,
    hrtf_dir: str,
    test_indices: List[int] = None,
    inputform: str = "voxel",
    n_folds: int = 1,
    val_fold: int = 0,
    seed: int = 42
) -> Dict:
    """
    将数据集分割为训练集和测试集，支持k折交叉验证

    Args:
        voxel_dir (str): 体素目录路径
        hrtf_dir (str): HRTF文件目录路径
        test_indices (list, optional): 测试集索引列表. 默认为None时使用预定义的索引
        inputform (str): 输入形式，"voxel" 或 "image"
        n_folds (int): k折交叉验证的折数，默认为1（不使用交叉验证）
        val_fold (int): 验证折的序号（0到n_folds-1），默认为0
        seed (int): 随机种子，用于k折划分

    Returns:
        dict: 包含训练集和测试集路径的字典
    """
    # 设置随机种子
    np.random.seed(seed)

    # 获取并排序体素列表
    voxel_list = os.listdir(voxel_dir)
    voxel_list.sort(key=lambda x: int(x.split('.')[0].split('_')[0][1:]))

    # 分离左右耳体素
    if inputform == "voxel":
        left_voxel_list = [vox for vox in voxel_list if vox.endswith('left.npy')]
        right_voxel_list = [vox for vox in voxel_list if vox.endswith('right.npy')]
    elif inputform == "image":
        left_voxel_list = [vox for vox in voxel_list if vox.endswith('left.png')]
        right_voxel_list = [vox for vox in voxel_list if vox.endswith('right.png')]

    # 如果指定了k折交叉验证
    if n_folds > 1:
        n_samples = len(right_voxel_list)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        # 将索引均匀分成k折
        base_size = n_samples // n_folds  # 基础大小
        remainder = n_samples % n_folds   # 余数
        folds = []
        start = 0
        for i in range(n_folds):
            # 前 remainder 折多分配一个样本
            end = start + base_size + (1 if i < remainder else 0)
            folds.append(indices[start:end])
            start = end

        # 确定验证折的索引
        val_indices_set = set(folds[val_fold])
        test_indices = [i for i in range(n_samples) if i in val_indices_set]

        print(f"Using fold {val_fold} for validation, with {len(test_indices)}/{n_samples} samples in the test set.")

    elif test_indices is None:
        test_indices = [7, 14, 27, 30, 31, 52, 54, 55, 70, 82, 143, 184]

    # 分割训练集和测试集
    left_train = [x for i, x in enumerate(left_voxel_list) if i not in test_indices]
    right_train = [x for i, x in enumerate(right_voxel_list) if i not in test_indices]
    left_test = [x for i, x in enumerate(left_voxel_list) if i in test_indices]
    right_test = [x for i, x in enumerate(right_voxel_list) if i in test_indices]

    # 从体素名称中提取编号
    train_voxel_numbers = [int(vox.split('_')[0][1:]) for vox in right_train]
    test_voxel_numbers = [int(vox.split('_')[0][1:]) for vox in right_test]

    # 过滤HRTF文件列表
    train_hrtf_list = [x for x in os.listdir(hrtf_dir) if int(x.split('.')[0][1:]) in train_voxel_numbers]
    train_hrtf_list = [os.path.join(hrtf_dir, f) for f in train_hrtf_list if os.path.isfile(os.path.join(hrtf_dir, f))]

    test_hrtf_list = [x for x in os.listdir(hrtf_dir) if int(x.split('.')[0][1:]) in test_voxel_numbers]
    test_hrtf_list = [os.path.join(hrtf_dir, f) for f in test_hrtf_list if os.path.isfile(os.path.join(hrtf_dir, f))]

    # 获取完整路径
    left_train = [os.path.join(voxel_dir, vox) for vox in left_train]
    right_train = [os.path.join(voxel_dir, vox) for vox in right_train]
    left_test = [os.path.join(voxel_dir, vox) for vox in left_test]
    right_test = [os.path.join(voxel_dir, vox) for vox in right_test]

    return {
        'train_hrtf_list': train_hrtf_list,
        'test_hrtf_list': test_hrtf_list,
        'left_train': left_train,
        'right_train': right_train,
        'left_test': left_test,
        'right_test': right_test
    }


def calculate_hrtf_mean(hrtf_file_names, whichear: str = None):
    """
    计算HRTF均值

    Args:
        hrtf_file_names: HRTF文件路径列表
        whichear: 'left' 或 'right'

    Returns:
        HRTF均值数组
    """
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
    return hrtf_mean