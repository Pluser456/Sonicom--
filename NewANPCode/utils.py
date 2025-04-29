import os
from re import S
import h5py
import numpy as np
import sys
from tqdm import tqdm

import torch

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

# model related
# model_path = "model.pth"
# model_path = os.path.join(cwd, model_path)

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
    loss_function = torch.nn.MSELoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()
 
    # 仅在主进程显示进度条
    if rank == 0:
        data_loader = tqdm(data_loader, file=sys.stdout)
    contexttargetmanager = ContextTargetManager(device, model, reuse_num=10)
    for step, sample_batch in enumerate(data_loader):
        # 数据迁移到设备
        # 修改这里使用特征而不是图像
        # left_image = sample_batch["left_image"]
        # right_image = sample_batch["right_image"]        
        # pos = sample_batch["position"].squeeze(1).to(device)
        # target = sample_batch["hrtf"].squeeze(1)[:, :].to(device)
        
        (target_x, target_y), (context_x, context_y) = contexttargetmanager.get_contexttarget(sample_batch)

        # 直接使用 prediction_net 而不是完整模型
        # output = model.prediction_net(img_features, pos)
        mu = model.anp(context_x, context_y, target_x)
        loss = loss_function(mu, target_y)
        accu_loss += loss.detach()

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # 仅在主进程更新进度条描述
        if rank == 0:
            data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))
        
        optimizer.step()
        optimizer.zero_grad()
 
    # 同步所有GPU上的损失
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(accu_loss)
        accu_loss = accu_loss / torch.distributed.get_world_size()
    
    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, rank=0, auxiliary_loader=None):
    model.eval()
    loss_function = torch.nn.MSELoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
 
    # 仅在主进程显示进度条
    if rank == 0:
        data_loader = tqdm(data_loader, file=sys.stdout)

    auxiliary_batch = next(iter(auxiliary_loader))
    
    contexttargetmanager = ContextTargetManager(device, model)
    for step, sample_batch in enumerate(data_loader):
        # 数据迁移到设备
        (target_x, target_y), (context_x, context_y) = contexttargetmanager.get_contexttarget(sample_batch, auxiliary=auxiliary_batch)

        # 直接使用 prediction_net
        mu = model.anp(context_x, context_y, target_x)
        loss = loss_function(mu, target_y)
        accu_loss += loss.detach()

        # 仅在主进程更新进度条描述
        if rank == 0:
            data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(
                epoch, accu_loss.item() / (step + 1)
            )
    
    # 同步所有GPU上的损失
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(accu_loss)
        accu_loss = accu_loss / torch.distributed.get_world_size()
 
    return accu_loss.item() / (step + 1)

class ContextTargetManager():
    """管理上下文和目标数据"""
    def __init__(self, device, model, reuse_num=10, target_num=100):
        self.reuse_time = reuse_num
        self.device = device
        self.model = model
        self.target_num = target_num
        
    def get_contexttarget(self, sample_batch, auxiliary=None):
        """管理上下文和目标数据"""
        if auxiliary is None:
            features, target = self._get_target_and_feature(sample_batch)
            # 选择上下文和目标数据
            batch_indices = np.zeros((self.reuse_time, target.shape[0]), dtype=int)
            for i in range(self.reuse_time):
                # 打乱顺序
                # batch_indices[i] = batch_indices[i][torch.randperm(batch_indices[i].shape[0])]
                batch_indices[i] = np.random.permutation(target.shape[0])
            target_indices = batch_indices[:, :self.target_num]
            context_indices = batch_indices[:, self.target_num:]
            target_x = features[target_indices]
            target_y = target[target_indices]
            context_x = features[context_indices]
            context_y = target[context_indices]
            return (target_x, target_y), (context_x, context_y)
        else:
            # 使用辅助数据, 即让上下文点来自辅助数据
            auxi_features, auxi_target = self._get_target_and_feature(auxiliary, status="train")
            test_features, test_target = self._get_target_and_feature(sample_batch, status="test")
            # 选择上下文和目标数据
            target_x = test_features.unsqueeze(0)
            target_y = test_target.unsqueeze(0)
            context_x = auxi_features.unsqueeze(0)
            context_y = auxi_target.unsqueeze(0)
            return (target_x, target_y), (context_x, context_y)
            
    
    def _get_target_and_feature(self, sample_batch, status="train"):
        """获取目标和特征"""
        if status == "train":
            left_image = sample_batch["left_image"]
            right_image = sample_batch["right_image"]
            pos = sample_batch["position"].squeeze(1).to(self.device)
            image_feature = self.model.feature_extractor(left_image, right_image)
            image_feature = image_feature.unsqueeze(1).repeat(1, pos.shape[1], 1)
            features = torch.cat([image_feature, pos], dim=2)
            features = features.reshape(-1, features.shape[-1])
            target = sample_batch["hrtf"].squeeze(1).to(self.device)
            target = target.reshape(-1, target.shape[-1])
            return features, target
        else:
            # 这里是测试集的处理
            left_image = sample_batch["left_image"]
            right_image = sample_batch["right_image"]
            pos = sample_batch["position"].squeeze(1).to(self.device)
            image_feature = self.model.feature_extractor(left_image, right_image)
            features = torch.cat([image_feature, pos], dim=1)
            target = sample_batch["hrtf"].squeeze(1).to(self.device)
            return features, target