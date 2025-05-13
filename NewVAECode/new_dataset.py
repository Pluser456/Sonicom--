import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from utils import calculate_hrtf_mean
import h5py
import os
from torchvision import transforms
from PIL import Image

class SonicomDataSet(Dataset):
    """使用预计算特征的数据集"""
    def __init__(self, hrtf_files, left_voxels, right_voxels, 
                 status="train", positions_chosen_num=793,
                 transform=None, 
                 calc_mean=True, use_diff=True, inputform="image",
                 mode="both", provided_mean_left=None, provided_mean_right=None):
        """
        Args:
            hrtf_files (list): HRTF文件路径列表
            left_voxels (list): 左耳体素路径列表
            right_voxels (list): 右耳体素路径列表
            device (str): 设备类型 - "cpu"/"cuda"
            status (str): 输出数据集模式 - "train"/"test"
            calc_mean (bool): 是否计算HRTF均值
            mode (str): 输出模式 - "left"/"right"/"both"
            positions_chosen_num (int): 每个文件选择的方位数
        """
        self.hrtf_files = self._validate_hrtf_files(hrtf_files)
        self.mode = mode
        self.status = status
        self.positions_chosen_num = positions_chosen_num  # 训练集每个文件选择的方位数
        # self.model = model
        if inputform == "image":
            if transform is None:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
            else:
                self.transform = transform
            self.left_tensor, self.right_tensor = self._get_image_tensor(left_voxels, right_voxels)
        elif inputform == "voxel":
            self.left_tensor, self.right_tensor = self._get_voxel_tensor(left_voxels, right_voxels)
        self.use_diff = use_diff  # 是否使用当前HRTF和平均HRTF之间的差值作为预测目标
        # 计算HRTF均值
        if calc_mean:
            self.log_mean_hrtf_left = 20 * np.log10(calculate_hrtf_mean(self.hrtf_files, whichear='left'))
            self.log_mean_hrtf_right = 20 * np.log10(calculate_hrtf_mean(self.hrtf_files, whichear='right'))
        else:
            self.log_mean_hrtf_left = provided_mean_left
            self.log_mean_hrtf_right = provided_mean_right
            
        # 获取方位数
        with h5py.File(self.hrtf_files[0], 'r') as f:
            self.positions_per_subject = f["F_left"].shape[0]
            

    def __len__(self):
        if self.status == "cvae":
            return len(self.hrtf_files)
            #return len(self.hrtf_files)*self.positions_per_subject
        else:
            return len(self.hrtf_files)

    def __getitem__(self, idx):
        # 计算文件索引和方位索引

        if self.status == "train":
            file_idx = idx
            position_idx = sorted(np.random.choice(self.positions_per_subject, self.positions_chosen_num, replace=False))
        elif self.status == "test":
            file_idx = idx
            position_idx = np.arange(self.positions_per_subject)  # 测试集使用所有方位
        elif self.status == "cvae":
            file_idx = idx // self.positions_per_subject
            position_idx = idx % self.positions_per_subject


        # 读取HRTF数据
        with h5py.File(self.hrtf_files[file_idx], 'r') as data:
            if self.status == "cvae":
                # 获取HRTF
                hrtf = self._get_hrtf(data, position_idx)
                # 获取方位角
                original_position_rad = torch.deg2rad(torch.tensor(data["theta"][:, position_idx].T).type(torch.float32))
                position = torch.tensor([
                    torch.sin(original_position_rad[0]),  # sin(azimuth)
                    torch.cos(original_position_rad[0]),  # cos(azimuth)
                    torch.sin(original_position_rad[1])   # sin(elevation)
                ])
            else:
                # 获取HRTF
                hrtf = self._get_hrtf(data, position_idx)
                # 获取方位角
                original_position_rad = torch.deg2rad(torch.tensor(data["theta"][:, position_idx].T).type(torch.float32))
                position = torch.stack([
                    torch.sin(original_position_rad[:, 0]), # sin(azimuth)
                    torch.cos(original_position_rad[:, 0]), # cos(azimuth)
                    torch.sin(original_position_rad[:, 1])  # sin(elevation)
                ], dim=1)

        left_voxel = self.left_tensor[file_idx, :, :, :]
        right_voxel = self.right_tensor[file_idx, :, :, :]

        return {
            "hrtf": hrtf,
            "position": position,
            "left_image": left_voxel,
            "right_image": right_voxel,
        }
        
    def _get_hrtf(self, data, position_idx):
        """获取指定模式的HRTF数据"""
        if self.mode == "left":
            hrtf_data = data["F_left"][position_idx, :]
            mean_hrtf = self.log_mean_hrtf_left[position_idx, :]
            hrtf = torch.tensor(20 * np.log10(hrtf_data) - mean_hrtf).type(torch.float32) if self.use_diff else torch.tensor(20 * np.log10(hrtf_data)).type(torch.float32)
        elif self.mode == "right":
            hrtf_data = data["F_right"][position_idx, :]
            mean_hrtf = self.log_mean_hrtf_right[position_idx, :]
            hrtf = torch.tensor(20 * np.log10(hrtf_data) - mean_hrtf).type(torch.float32) if self.use_diff else torch.tensor(20 * np.log10(hrtf_data)).type(torch.float32)
        else:  # both
            left_data = 20 * np.log10(data["F_left"][position_idx, :]) - self.log_mean_hrtf_left[position_idx, :]
            right_data = 20 * np.log10(data["F_right"][position_idx, :]) - self.log_mean_hrtf_right[position_idx, :]
            left_hrtf = torch.tensor(left_data).type(torch.float32) if self.use_diff else torch.tensor(left_data).type(torch.float32)
            right_hrtf = torch.tensor(right_data).type(torch.float32) if self.use_diff else torch.tensor(right_data).type(torch.float32)
            hrtf = torch.cat([left_hrtf, right_hrtf], dim=1)
        return hrtf
    
    def _get_voxel_tensor(self, left_voxel_path, right_voxel_path):
        """获取体素张量"""
        left_tensors = []
        right_tensors = []
        for _, (left_path, right_path) in enumerate(zip(left_voxel_path, right_voxel_path)):
            left_voxel = np.load(left_path)
            right_voxel = np.load(right_path)
            right_voxel = np.flip(right_voxel, axis=1).copy()
            left_voxel_tensor = torch.tensor(left_voxel, dtype=torch.float32).unsqueeze(0)
            right_voxel_tensor = torch.tensor(right_voxel, dtype=torch.float32).unsqueeze(0)
            left_tensors.append(left_voxel_tensor)
            right_tensors.append(right_voxel_tensor)
        left_tensors = torch.cat(left_tensors, dim=0)
        right_tensors = torch.cat(right_tensors, dim=0)
        return left_tensors, right_tensors
    
    def _get_image_tensor(self, left_image_path, right_image_path):
        """获取图像张量"""
        left_tensors = []
        right_tensors = []
        for _, (left_path, right_path) in enumerate(zip(left_image_path, right_image_path)):
            left_image = Image.open(left_path).convert('L')
            right_image = Image.open(right_path).convert('L').transpose(Image.FLIP_LEFT_RIGHT)
            left_image_tensor = self.transform(left_image).unsqueeze(0)
            right_image_tensor = self.transform(right_image).unsqueeze(0)
            left_tensors.append(left_image_tensor)
            right_tensors.append(right_image_tensor)
        left_tensors = torch.cat(left_tensors, dim=0)
        right_tensors = torch.cat(right_tensors, dim=0)
        return left_tensors, right_tensors
    
    def turn_auxiliary_mode(self, mode: bool):
        """切换为辅助测试集模式"""
        if mode:
            self.positions_chosen_num = self.positions_per_subject
        else:
            self.positions_chosen_num = self.positions_chosen_num
            
        

    @staticmethod
    def _validate_hrtf_files(hrtf_files):
        """验证HRTF文件的有效性"""
        valid_files = []
        for file_path in hrtf_files:
            with h5py.File(file_path, 'r') as data:
                left_hrtf = torch.tensor(data["F_left"][:])
                right_hrtf = torch.tensor(data["F_right"][:])
                if not np.isnan(np.sum(left_hrtf.cpu().data.numpy()) + np.sum(right_hrtf.cpu().data.numpy())):
                    valid_files.append(file_path)
                else:
                    print(f"Warning: Invalid HRTF file found: {file_path}")
        return valid_files

    @staticmethod
    def collate_fn(batch):
        """自定义批处理函数"""
        hrtfs = torch.stack([item["hrtf"] for item in batch])
        positions = torch.stack([item["position"] for item in batch])
        left_voxels = torch.stack([item["left_image"] for item in batch]).unsqueeze(1) # [B, 1, D, H, W]
        right_voxels = torch.stack([item["right_image"] for item in batch]).unsqueeze(1) # [B, 1, D, H, W]
        
        return {
            "hrtf": hrtfs,
            "position": positions,
            "left_image": left_voxels,
            "right_image": right_voxels,
        }


class SingleSubjectFeatureDataset(SonicomDataSet):
    """单个受试者的特征数据集"""
    def __init__(
            self, 
            hrtf_files,
            left_images, 
            right_images,
            feature_extractor,
            train_log_mean_hrtf_left,
            train_log_mean_hrtf_right,
            subject_id,
            transform=None,
            mode="both",
    ):
        # 确保subject_id有效
        if not (1 <= subject_id <= len(hrtf_files)):
            raise ValueError(f"Invalid subject_id: {subject_id}")
            
        # 只保留目标受试者的数据
        target_idx = subject_id - 1
        single_hrtf = [hrtf_files[target_idx]]
        single_left = [left_images[target_idx]]
        single_right = [right_images[target_idx]]
        
        # 调用父类初始化
        super().__init__(
            hrtf_files=single_hrtf,
            left_images=single_left,
            right_images=single_right,
            feature_extractor=feature_extractor,
            transform=transform,
            calc_mean=False,
            mode=mode,
            provided_mean_left=train_log_mean_hrtf_left,
            provided_mean_right=train_log_mean_hrtf_right
        )

    def __len__(self):
        """返回单个受试者的总方位数"""
        return self.positions_per_subject

    def __getitem__(self, idx):
        """
        获取数据项
        Args:
            idx (int): 方位索引
        """
        # 读取HRTF数据
        with h5py.File(self.hrtf_files[0], 'r') as data:
            # 获取方位角
            position = torch.tensor(data["theta"][:, idx].T).reshape(1, -1).type(torch.float32)

            # 获取训练集对应的均值
            if self.mode == "left":
                mean_value = torch.tensor(self.log_mean_hrtf_left[idx, :]).type(torch.float32)
                hrtf = torch.tensor(data["F_left"][idx, :]).reshape(1, -1).type(torch.float32)
            elif self.mode == "right":
                mean_value = torch.tensor(self.log_mean_hrtf_right[idx, :]).type(torch.float32)
                hrtf = torch.tensor(data["F_right"][idx, :]).reshape(1, -1).type(torch.float32)
            else:
                mean_left = torch.tensor(self.log_mean_hrtf_left[idx, :]).type(torch.float32)
                mean_right = torch.tensor(self.log_mean_hrtf_right[idx, :]).type(torch.float32)
                mean_value = torch.cat([mean_left, mean_right], dim=0)
                hrtf_left = torch.tensor(data["F_left"][idx, :]).reshape(1, -1).type(torch.float32)
                hrtf_right = torch.tensor(data["F_right"][idx, :]).reshape(1, -1).type(torch.float32)
                hrtf = torch.cat([hrtf_left, hrtf_right], dim=1)

        # 获取预计算的特征
        img_feature = self.image_features[0]

        return {
            "hrtf": hrtf,
            "meanlog": mean_value,
            "position": position,
            "image_feature": img_feature
        }
    
    @staticmethod
    def collate_fn(batch):
        """自定义批处理函数"""
        hrtfs = torch.stack([item["hrtf"] for item in batch])
        positions = torch.stack([item["position"] for item in batch])
        image_features = torch.stack([item["image_feature"] for item in batch])
        meanlog = torch.stack([item["meanlog"] for item in batch])
        
        return {
            "hrtf": hrtfs,
            "position": positions,
            "image_feature": image_features,
            "meanlog": meanlog
        }


class SonicomDataSetLeft(SonicomDataSet):
    """只返回左耳图像的数据集"""
    def __init__(self, hrtf_files, left_images, right_images,
                 transform=None, calc_mean=True, 
                 mode="left", provided_mean_left=None, provided_mean_right=None, status="train"):
        super().__init__(hrtf_files, left_images, right_images,  
                         transform=transform, calc_mean=calc_mean, 
                         mode=mode, provided_mean_left=provided_mean_left, 
                         provided_mean_right=provided_mean_right, status=status)
        
    def __getitem__(self, idx):
        # 调用父类的 __getitem__ 方法获取完整的 batch
        batch = super().__getitem__(idx)
        # 只返回 left_image 部分的数据
        return batch["left_image"]
    
    @staticmethod
    def collate_fn(batch):
        """自定义批处理函数：适配仅返回 left_image 的数据集"""
        # batch 是一个由 left_image 张量组成的列表，例如 [tensor1, tensor2, ...]
        left_images = torch.stack(batch)  # 直接堆叠所有 left_image 张量
        
        return {
            "left_image": left_images
        }
    
class SonicomDataSetHRTF(SonicomDataSet):
    """只返回左耳HRTF和position的数据集"""
    def __init__(self, hrtf_files, left_images, right_images,
                 transform=None, calc_mean=True, 
                 mode="left", provided_mean_left=None, provided_mean_right=None, status="train"):
        super().__init__(hrtf_files, left_images, right_images,  
                         transform=transform, calc_mean=calc_mean, 
                         mode=mode, provided_mean_left=provided_mean_left, 
                         provided_mean_right=provided_mean_right, status=status)
        
    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        position = batch["position"]
        return {  
            "hrtf": batch["hrtf"],
            "sin(azimuth)": position[0],
            "cos(azimuth)": position[1],
            "sin(elevation)": position[2]
        }
    
    @staticmethod
    def collate_fn(batch):
        hrtfs = torch.stack([item["hrtf"] for item in batch])
        sin_azimuths = torch.stack([item["sin(azimuth)"] for item in batch])
        cos_azimuths = torch.stack([item["cos(azimuth)"] for item in batch])
        sin_elevations = torch.stack([item["sin(elevation)"] for item in batch])
        
        return {
            "hrtf": hrtfs,
            "sin(azimuth)": sin_azimuths,
            "cos(azimuth)": cos_azimuths,
            "sin(elevation)": sin_elevations
        }