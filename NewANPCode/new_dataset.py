import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from utils import calculate_hrtf_mean
import h5py
import os
from TestNet import FeatureExtractor

class SonicomDataSet(Dataset):
    """使用预计算特征的数据集"""
    def __init__(self, hrtf_files, left_images, right_images, 
                 feature_extractor, transform=None, calc_mean=True, 
                 mode="both", provided_mean_left=None, provided_mean_right=None):
        """
        Args:
            hrtf_files (list): HRTF文件路径列表
            left_images (list): 左耳图像路径列表
            right_images (list): 右耳图像路径列表
            feature_extractor (FeatureExtractorManager): 特征提取管理器
            transform: 图像转换操作
            calc_mean (bool): 是否计算HRTF均值
            mode (str): 输出模式 - "left"/"right"/"both"
        """
        self.hrtf_files = self._validate_hrtf_files(hrtf_files)
        self.left_images = left_images
        self.right_images = right_images
        self.transform = transform
        self.mode = mode
        self.feature_extractor = feature_extractor
        
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
            
        # 提取并存储所有图像特征
        self.image_features = self.feature_extractor.extract_features(
            self.left_images, self.right_images, self.transform
        )
        
    def __len__(self):
        return len(self.hrtf_files) * self.positions_per_subject

    def __getitem__(self, idx):
        # 计算文件索引和方位索引
        file_idx = idx // self.positions_per_subject
        position_idx = idx % self.positions_per_subject

        # 读取HRTF数据
        with h5py.File(self.hrtf_files[file_idx], 'r') as data:
            # 获取HRTF
            hrtf = self._get_hrtf(data, position_idx)
            # 获取方位角
            position = torch.tensor(data["theta"][:, position_idx].T).reshape(1, -1).type(torch.float32)

        # 获取预计算的特征
        img_feature = self.image_features[file_idx]

        return {
            "hrtf": hrtf,
            "position": position,
            "image_feature": img_feature
        }
        
    def _get_hrtf(self, data, position_idx):
        """获取指定模式的HRTF数据"""
        if self.mode == "left":
            hrtf_data = data["F_left"][position_idx, :]
            mean_hrtf = self.log_mean_hrtf_left[position_idx, :]
            hrtf = torch.tensor(20 * np.log10(hrtf_data) - mean_hrtf).reshape(1, -1).type(torch.float32)
        elif self.mode == "right":
            hrtf_data = data["F_right"][position_idx, :]
            mean_hrtf = self.log_mean_hrtf_right[position_idx, :]
            hrtf = torch.tensor(20 * np.log10(hrtf_data) - mean_hrtf).reshape(1, -1).type(torch.float32)
        else:  # both
            left_data = 20 * np.log10(data["F_left"][position_idx, :]) - self.log_mean_hrtf_left[position_idx, :]
            right_data = 20 * np.log10(data["F_right"][position_idx, :]) - self.log_mean_hrtf_right[position_idx, :]
            left_hrtf = torch.tensor(left_data).reshape(1, -1).type(torch.float32)
            right_hrtf = torch.tensor(right_data).reshape(1, -1).type(torch.float32)
            hrtf = torch.cat([left_hrtf, right_hrtf], dim=1)
        return hrtf

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
        image_features = torch.stack([item["image_feature"] for item in batch])
        
        return {
            "hrtf": hrtfs,
            "position": positions,
            "image_feature": image_features
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
    
class FeatureExtractorManager:
    """特征提取管理器"""
    def __init__(self, model_path=None, extractor=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = extractor if extractor is not None else FeatureExtractor().to(self.device)
        
        # 用于存储特征的字典
        self.features_cache = {}
        
    def extract_features(self, left_image_paths, right_image_paths, transform):
        """提取并缓存特征"""
        features = []
        
        for i, (left_path, right_path) in enumerate(zip(left_image_paths, right_image_paths)):
            # 创建特征的唯一标识
            feature_key = f"{left_path}_{right_path}"
            
            # 检查是否已经缓存
            if feature_key in self.features_cache:
                features.append(self.features_cache[feature_key])
                continue
            
            # 加载和处理图像
            left_img = Image.open(left_path).convert('L')
            right_img = Image.open(right_path).convert('L')
            left_tensor = transform(left_img).unsqueeze(0).to(self.device)
            right_tensor = transform(right_img).unsqueeze(0).to(self.device)
            
            # 提取特征
            feature = self.extractor(left_tensor, right_tensor)
                
            # 缓存特征
            features.append(feature)
            
        return features
    
    def clear_cache(self):
        """清除特征缓存"""
        self.features_cache = {}