from turtle import st
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from utils import calculate_hrtf_mean

class SonicomDataSet(Dataset):
    """人耳图像数据集"""
    def __init__(self, hrtf_files: list, left_images: list, right_images: list, device,
                 status="train",
                 transform=None,  calc_mean=True, use_diff=False,
                 mode="both", provided_mean_left=None, 
                 provided_mean_right=None):
        """
        Args:
            hrtf_files (list): HRTF文件路径列表
            left_images (list): 左耳图像路径列表
            right_images (list): 右耳图像路径列表
            transform: 图像转换操作
            calc_mean (bool): 是否计算HRTF均值
            mode (str): 输出模式 - "left"/"right"/"both"
        """
        super().__init__()
        self.hrtf_files = self._validate_hrtf_files(hrtf_files)
        self.left_images = left_images
        self.right_images = right_images
        self.device = device
        self.mode = mode
        self.status = status
        self.positions_chosen_num = 793  # 训练集每个文件选择的方位数
        # 初始化转换器
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ]) if transform is None else transform
        
        self.left_tensor, self.right_tensor = self._get_image_tensor(left_images, right_images)

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
        self.use_diff = use_diff  # 是否使用当前HRTF和平均HRTF之间的差值作为预测目标

    def __len__(self):
        if self.status =="train":
            return len(self.hrtf_files)
        else: 
            return len(self.hrtf_files) * self.positions_per_subject
        

    def __getitem__(self, idx):
        # 计算文件索引和方位索引
        if self.status == "train":
            file_idx = idx
            position_idx = sorted(np.random.choice(self.positions_per_subject, self.positions_chosen_num, replace=False))

            # 读取HRTF数据
            with h5py.File(self.hrtf_files[file_idx], 'r') as data:
                # 获取HRTF
                hrtf = self._get_hrtf(data, position_idx)
                # 获取方位角
                original_position_rad = torch.deg2rad(torch.tensor(data["theta"][:, position_idx].T).type(torch.float32))
                position = torch.stack([
                    torch.sin(original_position_rad[:, 0]), # sin(azimuth)
                    torch.cos(original_position_rad[:, 0]), # cos(azimuth)
                    torch.sin(original_position_rad[:, 1])  # sin(elevation)
                ], dim=1)

            # left_image = self.left_tensor[file_idx, :, :, :]
            right_image = self.right_tensor[file_idx, :, :, :]
        else:
            file_idx = idx // self.positions_per_subject
            position_idx = idx % self.positions_per_subject

            # 读取HRTF数据
            with h5py.File(self.hrtf_files[file_idx], 'r') as data:
                # 获取HRTF
                hrtf = self._get_hrtf(data, position_idx)
                # 获取方位角
                original_position_rad = torch.deg2rad(torch.tensor(data["theta"][:, position_idx]).type(torch.float32))
                position = torch.stack([
                    torch.sin(original_position_rad[0]), # sin(azimuth)
                    torch.cos(original_position_rad[0]), # cos(azimuth)
                    torch.sin(original_position_rad[1])  # sin(elevation)
                ])

            # left_image = self.left_tensor[file_idx, :, :, :]
            right_image = self.right_tensor[file_idx, :, :, :]

        return {
            "hrtf": hrtf,
            "position": position,
            "left_image": [],
            "right_image": right_image,
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
    
    def _get_image_tensor(self, left_image_path, right_image_path):
        """获取图像张量"""
        left_tensors = []
        right_tensors = []
        for right_path in right_image_path:
            # left_image = Image.open(left_path).convert('L')
            right_image = Image.open(right_path).convert('L').transpose(Image.FLIP_LEFT_RIGHT)
            # left_image_tensor = self.transform(left_image).unsqueeze(0)
            right_image_tensor = self.transform(right_image).unsqueeze(0)
            # left_tensors.append(left_image_tensor)
            right_tensors.append(right_image_tensor)
        # left_tensors = torch.cat(left_tensors, dim=0)
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
        # left_images = torch.stack([item["left_image"] for item in batch]).unsqueeze(1)
        right_images = torch.stack([item["right_image"] for item in batch]).unsqueeze(1)
        
        return {
            "hrtf": hrtfs,
            "position": positions,
            "left_image": [],
            "right_image": right_images
        }
   
class SingleSubjectDataSet(SonicomDataSet):
    """单个受试者的数据集"""
    def __init__(
            self, 
            hrtf_files: list,
            device,
            left_images: list, 
            right_images: list,
            train_log_mean_hrtf_left: np.ndarray,
            train_log_mean_hrtf_right: np.ndarray,
            subject_id: int,
            transform=None,
            mode="both",
            
            
    ):
        """
        Args:
            hrtf_files (list): HRTF文件路径列表
            left_images (list): 左耳图像路径列表
            right_images (list): 右耳图像路径列表
            subject_id (int): 目标受试者ID（从1开始）
            transform: 图像转换操作
            mode (str): 输出模式 - "left"/"right"/"both"
        """
        # 确保subject_id有效
        if not (1 <= subject_id <= len(hrtf_files)):
            raise ValueError(f"Invalid subject_id: {subject_id}")
            
        # 只保留目标受试者的数据
        target_idx = subject_id - 1
        single_hrtf = [hrtf_files[target_idx]]
        # single_left = [left_images[target_idx]]
        single_right = [right_images[target_idx]]

        # 调用父类初始化
        super().__init__(
            hrtf_files=single_hrtf,
            left_images=[],
            right_images=single_right,
            transform=transform,
            calc_mean=False,
            mode=mode,
            device=device,
            provided_mean_left=train_log_mean_hrtf_left,
            provided_mean_right=train_log_mean_hrtf_right
        )
        # self.log_mean_hrtf_left = train_log_mean_hrtf_left # 直接使用训练集的均值
        # self.log_mean_hrtf_right = train_log_mean_hrtf_right # 直接使用训练集的均值

    def __len__(self):
        """返回单个受试者的总方位数"""
        return len(self.hrtf_files) # 还不知道是哪个
        # return self.positions_per_subject

    def __getitem__(self, position_idx):
        """
        获取数据项
        Args:
            idx (int): 方位索引
        """
        # 读取HRTF数据
        position_idx = np.arange(self.positions_per_subject)  # 测试集使用所有方位
        with h5py.File(self.hrtf_files[0], 'r') as data:
            # 获取方位角
            # 获取方位角
            original_position_rad = torch.deg2rad(torch.tensor(data["theta"][:, position_idx].T).type(torch.float32))
            position = torch.stack([
                torch.sin(original_position_rad[:, 0]), # sin(azimuth)
                torch.cos(original_position_rad[:, 0]), # cos(azimuth)
                torch.sin(original_position_rad[:, 1])  # sin(elevation)
            ], dim=1)

            # left_img = self.left_tensor[0, :, :, :]
            right_img = self.right_tensor[0, :, :, :]
            # 获取训练集对应的均值，注意是训练集！
            # 同时获取原始HRTF
            if self.mode == "left":
                mean_value = torch.tensor(self.log_mean_hrtf_left[position_idx, :]).type(torch.float32)
                hrtf = torch.tensor(data["F_left"][position_idx, :]).type(torch.float32) # 删掉了.reshape(1, -1)
            elif self.mode == "right":
                mean_value = torch.tensor(self.log_mean_hrtf_right[position_idx, :]).type(torch.float32)
                hrtf = torch.tensor(data["F_right"][position_idx, :]).type(torch.float32)
            else:
                mean_left = torch.tensor(self.log_mean_hrtf_left[position_idx, :]).type(torch.float32)
                mean_right = torch.tensor(self.log_mean_hrtf_right[position_idx, :]).type(torch.float32)
                mean_value = torch.cat([mean_left, mean_right], dim=0)
                hrtf_left = torch.tensor(data["F_left"][position_idx, :]).type(torch.float32)
                hrtf_right = torch.tensor(data["F_right"][position_idx, :]).type(torch.float32)
                hrtf = torch.cat([hrtf_left, hrtf_right], dim=1)

        # # 读取并处理图像
        # left_img = Image.open(self.left_images[0]).convert('RGB')
        # right_img = Image.open(self.right_images[0]).convert('RGB')
        # left_img = self.transform(left_img)
        # right_img = self.transform(right_img)

        return {
            "hrtf": hrtf,
            "meanlog": mean_value,
            "position": position,
            "left_image": [],
            "right_image": right_img
        }
    
    @staticmethod
    def collate_fn(batch):
        """自定义批处理函数"""
        hrtfs = torch.stack([item["hrtf"] for item in batch])
        positions = torch.stack([item["position"] for item in batch])
        # left_images = torch.stack([item["left_image"] for item in batch])
        right_images = torch.stack([item["right_image"] for item in batch])
        meanlog = torch.stack([item["meanlog"] for item in batch])
        
        return {
            "hrtf": hrtfs,
            "position": positions,
            "left_image": [],
            "right_image": right_images,
            "meanlog": meanlog
        }