import torch
import numpy as np
from torch.utils.data import Dataset
from utils import calculate_hrtf_mean
from PIL import Image
from torchvision import transforms
import h5py

class SonicomDataSet(Dataset):
    """使用预计算特征的数据集"""
    def __init__(self, hrtf_files, left_voxels, right_voxels, 
                 status="train",
                 calc_mean=True, use_diff=True, inputform="voxel",
                 mode="both", provided_mean_left=None, provided_mean_right=None, provided_feature=None):
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
        self.hrtf_files = hrtf_files
        self.mode = mode
        self.status = status
        # self.model = model
        if inputform == "image":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
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
        
        self.feature = provided_feature

    def __len__(self):
        return len(self.hrtf_files)

    def __getitem__(self, idx):
        # 计算文件索引和方位索引
        file_idx = idx
        if self.status == "train":
            position_idx = sorted(np.random.choice(self.positions_per_subject, self.positions_per_subject, replace=False))
        else:
            position_idx = np.arange(self.positions_per_subject)  # 测试集使用所有方位

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

        left_voxel = self.left_tensor[file_idx, :, :, :] if self.left_tensor is not None else None
        right_voxel = self.right_tensor[file_idx, :, :, :] if self.right_tensor is not None else None

        feature = self.feature[file_idx, :] if self.feature is not None else None
        return {
            "hrtf": hrtf,
            "position": position,
            "left_voxel": left_voxel,
            "right_voxel": right_voxel,
            "feature": feature,
        }
        
    def _get_hrtf(self, data, position_idx):
        """获取指定模式的HRTF数据"""
        if self.mode == "left":
            hrtf_data = data["F_left"][position_idx, :]
            mean_hrtf = self.log_mean_hrtf_left[position_idx, :] if self.use_diff else None
            hrtf = torch.tensor(20 * np.log10(hrtf_data) - mean_hrtf).type(torch.float32) if self.use_diff else torch.tensor(20 * np.log10(hrtf_data)).type(torch.float32)
        elif self.mode == "right":
            hrtf_data = data["F_right"][position_idx, :]
            mean_hrtf = self.log_mean_hrtf_right[position_idx, :] if self.use_diff else None
            hrtf = torch.tensor(20 * np.log10(hrtf_data) - mean_hrtf).type(torch.float32) if self.use_diff else torch.tensor(20 * np.log10(hrtf_data)).type(torch.float32)
        else:  # both
            left_data = 20 * np.log10(data["F_left"][position_idx, :]) - self.log_mean_hrtf_left[position_idx, :]
            right_data = 20 * np.log10(data["F_right"][position_idx, :]) - self.log_mean_hrtf_right[position_idx, :]
            left_hrtf = torch.tensor(left_data).type(torch.float32) if self.use_diff else torch.tensor(left_data).type(torch.float32)
            right_hrtf = torch.tensor(right_data).type(torch.float32) if self.use_diff else torch.tensor(right_data).type(torch.float32)
            hrtf = torch.cat([left_hrtf, right_hrtf], dim=1)
        return hrtf
    
    def _get_voxel_tensor(self, left_voxel_path, right_voxel_path):
        """获取体素张量，当某个路径列表为空时，仅对应的张量为 None"""
        left_tensors = None
        right_tensors = None

        if left_voxel_path:  # 仅当 left_voxel_path 不为空时才处理
            left_tensors = []
            for left_path in left_voxel_path:
                left_voxel = np.load(left_path)
                left_voxel_tensor = torch.tensor(left_voxel, dtype=torch.float32).unsqueeze(0)
                left_tensors.append(left_voxel_tensor)
            left_tensors = torch.cat(left_tensors, dim=0).unsqueeze(1)

        if right_voxel_path:  # 仅当 right_voxel_path 不为空时才处理
            right_tensors = []
            for right_path in right_voxel_path:
                right_voxel = np.load(right_path)
                right_voxel = np.flip(right_voxel, axis=1).copy()
                right_voxel_tensor = torch.tensor(right_voxel, dtype=torch.float32).unsqueeze(0)
                right_tensors.append(right_voxel_tensor)
            right_tensors = torch.cat(right_tensors, dim=0).unsqueeze(1)

        return left_tensors, right_tensors
    
    def _get_image_tensor(self, left_image_path, right_image_path):
        """获取图像张量，当某个路径列表为空时，仅对应的张量为 None"""
        left_tensors = None
        right_tensors = None

        if left_image_path:  # 仅当 left_image_path 不为空时才处理
            left_tensors = []
            for left_path in left_image_path:
                left_image = Image.open(left_path).convert('L')
                left_image_tensor = self.transform(left_image).unsqueeze(0)
                left_tensors.append(left_image_tensor)
            left_tensors = torch.cat(left_tensors, dim=0)

        if right_image_path:  # 仅当 right_image_path 不为空时才处理
            right_tensors = []
            for right_path in right_image_path:
                right_image = Image.open(right_path).convert('L').transpose(Image.FLIP_LEFT_RIGHT)
                right_image_tensor = self.transform(right_image).unsqueeze(0)
                right_tensors.append(right_image_tensor)
            right_tensors = torch.cat(right_tensors, dim=0)

        return left_tensors, right_tensors
        

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
        left_voxels = torch.stack([item["left_voxel"] for item in batch]) if batch[0]["left_voxel"] is not None else None
        right_voxels = torch.stack([item["right_voxel"] for item in batch]) if batch[0]["right_voxel"] is not None else None
        features = torch.stack([item["feature"] for item in batch]) if batch[0]["feature"] is not None else None
        
        return {
            "hrtf": hrtfs,
            "position": positions,
            "left_voxel": left_voxels,
            "right_voxel": right_voxels,
            "feature": features
        }
    
class OnlyHRTFDataSet(SonicomDataSet):
    '''仅输出HRTF和方位角的数据集'''
    def __init__(self, hrtf_files, 
                 status="train",
                 calc_mean=True, use_diff=True,
                 mode="both", provided_mean_left=None, provided_mean_right=None):
        self.hrtf_files = hrtf_files
        self.mode = mode
        self.status = status
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
        
    def __getitem__(self, idx):
        # 计算文件索引和方位索引
        file_idx = idx
        if self.status == "train":
            position_idx = sorted(np.random.choice(self.positions_per_subject, self.positions_per_subject, replace=False))
        else:
            position_idx = np.arange(self.positions_per_subject)  # 测试集使用所有方位

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

        return {
            "hrtf": hrtf,
            "position": position
        }
    @staticmethod
    def collate_fn(batch):
        """自定义批处理函数"""
        hrtfs = torch.stack([item["hrtf"] for item in batch])
        positions = torch.stack([item["position"] for item in batch])
        return {
            "hrtf": hrtfs,
            "position": positions,
        }


class SingleSubjectDataSet(SonicomDataSet):
    """单个受试者的特征数据集"""
    def __init__(
            self, 
            hrtf_files,
            left_voxels, 
            right_voxels,
            train_log_mean_hrtf_left,
            train_log_mean_hrtf_right,
            subject_id,
            mode="both",
            inputform="voxel",
    ):
        # 确保subject_id有效
        if not (1 <= subject_id <= len(hrtf_files)):
            raise ValueError(f"Invalid subject_id: {subject_id}")
            
        # 只保留目标受试者的数据
        target_idx = subject_id - 1
        single_hrtf = [hrtf_files[target_idx]]
        single_left = [left_voxels[target_idx]] if left_voxels else None
        single_right = [right_voxels[target_idx]] if right_voxels else None
        
        # 调用父类初始化
        super().__init__(
            hrtf_files=single_hrtf,
            left_voxels=single_left,
            right_voxels=single_right,
            status="test",
            calc_mean=False,
            mode=mode,
            provided_mean_left=train_log_mean_hrtf_left,
            provided_mean_right=train_log_mean_hrtf_right,
            inputform=inputform,
        )

    def __getitem__(self, position_idx):
        """
        获取数据项
        Args:
            idx (int): 索引
        """
        position_idx = np.arange(self.positions_per_subject)  # 测试集使用所有方位
        # 读取HRTF数据
        with h5py.File(self.hrtf_files[0], 'r') as data:
            # 获取方位角
            original_position_rad = torch.deg2rad(torch.tensor(data["theta"][:, position_idx].T).type(torch.float32))
            position = torch.stack([
                torch.sin(original_position_rad[:, 0]), # sin(azimuth)
                torch.cos(original_position_rad[:, 0]), # cos(azimuth)
                torch.sin(original_position_rad[:, 1])  # sin(elevation)
            ], dim=1)


            # 获取训练集对应的均值
            if self.mode == "left":
                mean_value = torch.tensor(self.log_mean_hrtf_left[position_idx, :]).type(torch.float32)
                hrtf = torch.tensor(data["F_left"][position_idx, :]).type(torch.float32)
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

        left_voxel = self.left_tensor[0, :, :, :] if self.left_tensor is not None else None
        right_voxel = self.right_tensor[0, :, :, :] if self.right_tensor is not None else None

        return {
            "hrtf": hrtf,
            "meanlog": mean_value,
            "position": position,
            "left_voxel": left_voxel,
            "right_voxel": right_voxel,
        }
    
    @staticmethod
    def collate_fn(batch):
        """自定义批处理函数"""
        hrtfs = torch.stack([item["hrtf"] for item in batch])
        positions = torch.stack([item["position"] for item in batch])
        left_voxel = torch.stack([item["left_voxel"] for item in batch]) if batch[0]["left_voxel"] is not None else None
        right_voxel = torch.stack([item["right_voxel"] for item in batch]) if batch[0]["right_voxel"] is not None else None
        meanlog = torch.stack([item["meanlog"] for item in batch])
        
        return {
            "hrtf": hrtfs,
            "position": positions,
            "left_voxel": left_voxel,
            "right_voxel": right_voxel,
            "meanlog": meanlog
        }
