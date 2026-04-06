import torch
import numpy as np
import h5py
import torch.nn.functional as F
from .hrtf import SonicomDataSet


class HRTFDataSet(SonicomDataSet):
    """使用预计算特征的数据集，继承自SonicomDataSet"""
    def __init__(self, hrtf_files, left_voxels, right_voxels, 
                 status="train",
                 calc_mean=True, use_diff=True, inputform="voxel",
                 mode="both", provided_mean_left=None, provided_mean_right=None, pos_num_per_batch=7):
        super().__init__(
            hrtf_files=hrtf_files,
            left_voxels=left_voxels,
            right_voxels=right_voxels,
            status=status,
            calc_mean=calc_mean,
            use_diff=use_diff,
            inputform=inputform,
            mode=mode,
            provided_mean_left=provided_mean_left,
            provided_mean_right=provided_mean_right,
        )
        self.pos_num_per_batch = pos_num_per_batch
        self.shuffled_indices = {}

    def __len__(self):
        self.max_batch_num_per_file = self.positions_per_subject // self.pos_num_per_batch \
            + (1 if self.positions_per_subject % self.pos_num_per_batch != 0 else 0)
        return len(self.hrtf_files) * self.max_batch_num_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.max_batch_num_per_file
        batch_idx = idx % self.max_batch_num_per_file

        start = batch_idx * self.pos_num_per_batch
        end = start + self.pos_num_per_batch
        # 获取当前文件打乱后的索引
        shuffled_file_indices = self.shuffled_indices[file_idx]
        position_idx = shuffled_file_indices[start:end]

        # 载入 HRTF
        with h5py.File(self.hrtf_files[file_idx], 'r') as data:
            hrtf = self._get_hrtf(data, position_idx)

        one_hot = F.one_hot(torch.LongTensor(position_idx), num_classes=self.positions_per_subject)  # shape: [N, num_classes]

        # 载入图像或体素
        left_voxel = self._load_data(self.left_voxel_paths[file_idx], is_right=False) if self.left_voxel_paths else None
        right_voxel = self._load_data(self.right_voxel_paths[file_idx], is_right=True) if self.right_voxel_paths else None

        return {
            "hrtf": hrtf,
            "one_hot": one_hot,
            "left_voxel": left_voxel,
            "right_voxel": right_voxel
        }

    def _get_hrtf(self, data, position_idx):
        if self.mode == "left":
            hrtf_data = data["F_left"][:, :][position_idx, :]
            mean_hrtf = self.log_mean_hrtf_left[position_idx, :] if self.use_diff else 0
            return torch.tensor(20 * np.log10(hrtf_data) - mean_hrtf).type(torch.float32)
        elif self.mode == "right":
            hrtf_data = data["F_right"][:, :][position_idx, :]
            mean_hrtf = self.log_mean_hrtf_right[position_idx, :] if self.use_diff else 0
            return torch.tensor(20 * np.log10(hrtf_data) - mean_hrtf).type(torch.float32)

    def on_epoch_end(self):
        """在每个epoch开始时调用，为每个文件生成随机不重复的索引序列"""
        if self.status == "train":
            for i in range(len(self.hrtf_files)):
                indices = np.arange(self.positions_per_subject)
                np.random.shuffle(indices)
                self.shuffled_indices[i] = indices
        else:
            for i in range(len(self.hrtf_files)):
                self.shuffled_indices[i] = np.arange(self.positions_per_subject)

    @staticmethod
    def collate_fn(batch):
        hrtfs = torch.stack([item["hrtf"] for item in batch])
        one_hots = torch.stack([item["one_hot"] for item in batch])
        left_voxels = torch.stack([item["left_voxel"] for item in batch]) if batch[0]["left_voxel"] is not None else None
        right_voxels = torch.stack([item["right_voxel"] for item in batch]) if batch[0]["right_voxel"] is not None else None
        return {
            "hrtf": hrtfs,
            "one_hot": one_hots,
            "left_voxel": left_voxels,
            "right_voxel": right_voxels,
        }

class SingleSubjectDataSet(HRTFDataSet):
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

        left_voxel = self._load_data(self.left_voxel_paths[0], is_right=False) if self.left_voxel_paths else None
        right_voxel = self._load_data(self.right_voxel_paths[0], is_right=True) if self.right_voxel_paths else None

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
