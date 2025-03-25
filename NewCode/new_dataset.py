import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from utils import calculate_hrtf_mean

class SonicomDataSet(Dataset):
    """人耳图像数据集"""
    def __init__(self, hrtf_files: list, left_images: list, right_images: list, transform=None, mode="both"):
        """
        Args:
            hrtf_files (list): HRTF文件路径列表
            left_images (list): 左耳图像路径列表
            right_images (list): 右耳图像路径列表
            transform: 图像转换操作
            mode (str): 输出模式 - "left"/"right"/"both"
        """
        self.hrtf_files = self._validate_hrtf_files(hrtf_files)
        self.left_images = left_images
        self.right_images = right_images
        self.mode = mode
        
        # 初始化转换器
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ]) if transform is None else transform

        # 计算HRTF均值
        self.log_mean_hrtf_left = 20 * np.log10(calculate_hrtf_mean(self.hrtf_files, whichear='left'))
        self.log_mean_hrtf_right = 20 * np.log10(calculate_hrtf_mean(self.hrtf_files, whichear='right'))
        
        # 获取方位数
        with h5py.File(self.hrtf_files[0], 'r') as f:
            self.positions_per_subject = f["F_left"].shape[0]

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

        # 读取并处理图像
        left_img = Image.open(self.left_images[file_idx])
        right_img = Image.open(self.right_images[file_idx])
        left_img = self.transform(left_img)
        right_img = self.transform(right_img)

        return {
            "hrtf": hrtf,
            "position": position,
            "left_image": left_img,
            "right_image": right_img
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
        left_images = torch.stack([item["left_image"] for item in batch])
        right_images = torch.stack([item["right_image"] for item in batch])
        
        return {
            "hrtf": hrtfs,
            "position": positions,
            "left_image": left_images,
            "right_image": right_images
        }