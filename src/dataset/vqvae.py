import torch
import numpy as np
from tqdm import tqdm
import h5py
from .hrtf import SonicomDataSet

from ..utils import calculate_hrtf_mean


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


class CNNDataSet(SonicomDataSet):
    """
    CNN 分类器训练数据集
    预计算 VQVAE 的 VQ 索引作为分类目标
    支持 ResNet3DClassifier 和 ResNet2DClassifier
    """
    def __init__(self, hrtf_files, left_voxels, right_voxels,
                 vqvae_model=None, device=None,
                 status="train", calc_mean=True, use_diff=False, mode="both",
                 inputform="image",
                 provided_mean_left=None, provided_mean_right=None):
        # 调用父类初始化
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
            provided_mean_right=provided_mean_right
        )

        # 保存模型引用
        self.vqvae_model = vqvae_model
        self.device = device

        # 预计算所有 VQVAE 特征
        print(f"预计算 CNNDataSet VQVAE 特征 ({len(self.hrtf_files)} 个样本)...")
        self._precompute_vqvae_features()

    def _precompute_vqvae_features(self):
        """批量预计算所有样本的 VQVAE VQ 索引"""
        self.vq_indices_data = []
        self.vqvae_model.eval()

        for hrtf_file in tqdm(self.hrtf_files, desc="预计算 VQVAE 特征"):
            # 读取 HRTF 数据
            with h5py.File(hrtf_file, 'r') as data:
                position_idx = np.arange(self.positions_per_subject)  # 使用所有方位
                hrtf = self._get_hrtf(data, position_idx).unsqueeze(0).to(self.device)
                original_position_rad = torch.deg2rad(torch.tensor(data["theta"][:, position_idx].T).type(torch.float32))
                position = torch.stack([
                    torch.sin(original_position_rad[:, 0]),
                    torch.cos(original_position_rad[:, 0]),
                    torch.sin(original_position_rad[:, 1])
                ], dim=1).unsqueeze(0).to(self.device)

            # VQVAE 前向获取 VQ 索引
            with torch.no_grad():
                _, _, indices = self.vqvae_model(hrtf, position)

            self.vq_indices_data.append(indices.cpu())

        # 合并所有数据
        self.feature = torch.cat(self.vq_indices_data, dim=0)  # [total_positions, encoder_out_vec_num]

    def __getitem__(self, idx):
        file_idx = idx

        # 载入图像或体素
        left_voxel = self._load_data(self.left_voxel_paths[file_idx], is_right=False) if self.left_voxel_paths else None
        right_voxel = self._load_data(self.right_voxel_paths[file_idx], is_right=True) if self.right_voxel_paths else None
        feature = self.feature[file_idx, :] if self.feature is not None else None

        return {
            "left_voxel": left_voxel,
            "right_voxel": right_voxel,
            "feature": feature,
        }
    
    @staticmethod
    def collate_fn(batch):
        left_voxels = torch.stack([item["left_voxel"] for item in batch]) if batch[0]["left_voxel"] is not None else None
        right_voxels = torch.stack([item["right_voxel"] for item in batch]) if batch[0]["right_voxel"] is not None else None
        features = torch.stack([item["feature"] for item in batch]) if batch[0]["feature"] is not None else None
        return {
            "left_voxel": left_voxels,
            "right_voxel": right_voxels,
            "vq_indices": features,  # [batch, encoder_out_vec_num]
        }
