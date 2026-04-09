import torch
import numpy as np
from tqdm import tqdm
import h5py
from .hrtf import SonicomDataSet


class VAEDataSet(SonicomDataSet):
    """
    专用于 VAE 训练的数据集
    每个样本返回双耳图像
    """
    def __init__(self, hrtf_files, left_voxels, right_voxels, 
                 status="train",
                 calc_mean=True, use_diff=True,
                 mode="left", provided_mean_left=None, provided_mean_right=None):
        super().__init__(
            hrtf_files=hrtf_files,
            left_voxels=left_voxels,
            right_voxels=right_voxels,
            status=status,
            calc_mean=calc_mean,
            use_diff=use_diff,
            inputform="image",
            mode=mode,
            provided_mean_left=provided_mean_left,
            provided_mean_right=provided_mean_right,
        )

    def __getitem__(self, idx):
        # 载入图像
        left_voxel = self._load_data(self.left_voxel_paths[idx], is_right=False) if self.left_voxel_paths else None
        right_voxel = self._load_data(self.right_voxel_paths[idx], is_right=True) if self.right_voxel_paths else None
        return {
            "left_voxel": left_voxel,
            "right_voxel": right_voxel,
        }

    @staticmethod
    def collate_fn(batch):
        """自定义批处理函数"""
        left_voxels = torch.stack([item["left_voxel"] for item in batch]) if batch[0]["left_voxel"] is not None else None
        right_voxels = torch.stack([item["right_voxel"] for item in batch]) if batch[0]["right_voxel"] is not None else None
        return {
            "left_voxel": left_voxels,
            "right_voxel": right_voxels,
        }


class DNNDataSet(SonicomDataSet):
    """
    DNN 训练数据集
    预计算 VAE 的 z_ears（左右耳分别计算）和 CVAE 的 z_hrtf（均值）
    返回: z_ears_left(64) + z_ears_right(64) + az + el -> z_hrtf(32)
    """
    def __init__(self, hrtf_files, left_images, right_images,
                 vae_model=None, cvae_model=None, device=None,
                 status="train", calc_mean=True, use_diff=False, mode="both",
                 provided_mean_left=None, provided_mean_right=None):
        # 调用父类初始化
        super().__init__(
            hrtf_files=hrtf_files,
            left_voxels=left_images,
            right_voxels=right_images,
            status=status,
            calc_mean=calc_mean,
            use_diff=use_diff,
            inputform="image",
            mode=mode,
            provided_mean_left=provided_mean_left,
            provided_mean_right=provided_mean_right
        )

        # 保存模型引用
        self.vae_model = vae_model
        self.cvae_model = cvae_model
        self.device = device

        # 预计算所有潜在变量
        self.total_positions = self.positions_per_subject * len(self.hrtf_files)
        print(f"预计算 DNNDataSet 数据 ({self.total_positions} 个位置)...")
        self._precompute_latent_variables()

    def _precompute_latent_variables(self):
        """批量预计算所有样本的 z_ears 和 z_hrtf"""
        self.z_ears_left_data = []
        self.z_ears_right_data = []
        self.z_hrtf_data = []
        self.position_data = []  # [az, el] 原始方位角和俯仰角

        self.vae_model.eval()
        self.cvae_model.eval()

        for file_idx, hrtf_file in enumerate(tqdm(self.hrtf_files, desc="预计算潜在变量")):
            # 获取耳朵图像
            left_image = self._load_data(self.left_voxel_paths[file_idx], is_right=False).to(self.device) if self.left_voxel_paths else None
            right_image = self._load_data(self.right_voxel_paths[file_idx], is_right=True).to(self.device) if self.right_voxel_paths else None

            # 读取 HRTF 数据
            with h5py.File(hrtf_file, 'r') as data:
                hrtf = self._get_hrtf(data, np.arange(self.positions_per_subject)).to(self.device)
                position = torch.tensor(data["theta"][:].T, dtype=torch.float32).to(self.device)

            # VAE 前向获取 z_ears_left
            with torch.no_grad():
                if left_image is not None:
                    h_vae_left = self.vae_model.encoder(left_image.unsqueeze(0))
                    z_ears_left = self.vae_model.fc_mu(h_vae_left)
                    del h_vae_left
                else:
                    z_ears_left = torch.zeros(1, 1, device=self.device)

            # VAE 前向获取 z_ears_right
            with torch.no_grad():
                if right_image is not None:
                    h_vae_right = self.vae_model.encoder(right_image.unsqueeze(0))
                    z_ears_right = self.vae_model.fc_mu(h_vae_right)
                    del h_vae_right
                else:
                    z_ears_right = torch.zeros(1, 1, device=self.device)

            # CVAE encoder 获取 z_hrtf
            with torch.no_grad():
                z_hrtf, _ = self.cvae_model.enc(hrtf, position)

            self.z_ears_left_data.append(z_ears_left.cpu())
            self.z_ears_right_data.append(z_ears_right.cpu())
            self.z_hrtf_data.append(z_hrtf.cpu())
            self.position_data.append(position.cpu())

        # 合并所有数据
        self.z_ears_left_data = torch.cat(self.z_ears_left_data, dim=0)
        self.z_ears_right_data = torch.cat(self.z_ears_right_data, dim=0)
        self.z_hrtf_data = torch.cat(self.z_hrtf_data, dim=0)
        self.position_data = torch.cat(self.position_data, dim=0)

    def __len__(self):
        return self.total_positions

    def __getitem__(self, idx):
        subject_idx = idx // self.positions_per_subject
        return {
            "z_ears_left": self.z_ears_left_data[subject_idx] if self.left_voxel_paths else None,
            "z_ears_right": self.z_ears_right_data[subject_idx] if self.right_voxel_paths else None,
            "z_hrtf": self.z_hrtf_data[idx],
            "position": self.position_data[idx],  # [2]: [az, el]
        }

    @staticmethod
    def collate_fn(batch):
        """自定义批处理函数"""
        z_ears_left = torch.stack([item["z_ears_left"] for item in batch]) if batch[0]["z_ears_left"] is not None else None
        z_ears_right = torch.stack([item["z_ears_right"] for item in batch]) if batch[0]["z_ears_right"] is not None else None
        z_hrtf = torch.stack([item["z_hrtf"] for item in batch])
        positions = torch.stack([item["position"] for item in batch])
        return {
            "z_ears_left": z_ears_left,
            "z_ears_right": z_ears_right,
            "z_hrtf": z_hrtf,
            "position": positions,  # [batch, 2]
        }


class CVAEDataSet(SonicomDataSet):
    """
    专用于 CVAE 训练的数据集
    每个样本返回单个位置的 HRTF 数据和对应的方位角标签
    预加载所有数据到内存以提高加载速度
    """
    def __init__(self, hrtf_files,
                 status="train",
                 calc_mean=True, use_diff=True,
                 mode="left", provided_mean_left=None, provided_mean_right=None):
        super().__init__(
            hrtf_files=hrtf_files,
            left_voxels=None,
            right_voxels=None,
            status=status,
            calc_mean=calc_mean,
            use_diff=use_diff,
            mode=mode,
            provided_mean_left=provided_mean_left,
            provided_mean_right=provided_mean_right,
        )
        # 计算总位置数
        self.total_positions = self.positions_per_subject * len(self.hrtf_files)

        # 预加载所有数据到内存
        print(f"预加载 CVAEDataSet 数据到内存 ({self.total_positions} 个位置)...")
        self.hrtf_data = []
        self.theta_data = []
        for hrtf_file in tqdm(self.hrtf_files, desc="加载 HRTF 数据"):
            with h5py.File(hrtf_file, 'r') as data:
                # 预加载 HRTF（复用父类逻辑，但加载全部数据）
                hrtf = self._get_hrtf(data, np.arange(self.positions_per_subject))
                self.hrtf_data.append(hrtf)  # [positions, nfft]

                # 预加载 theta（直接转为 torch.tensor）
                self.theta_data.append(torch.tensor(data["theta"][:].T, dtype=torch.float32))  # [positions, 2]

    def __len__(self):
        return self.total_positions

    def __getitem__(self, idx):
        # 计算受试者索引和位置索引
        subject_idx = idx // self.positions_per_subject
        position_idx = idx % self.positions_per_subject

        # 从预加载的内存中获取数据
        hrtf = self.hrtf_data[subject_idx][position_idx]
        position = self.theta_data[subject_idx][position_idx]  # [2]: [az, el]

        return {
            "hrtf": hrtf,
            "position": position,
        }

    @staticmethod
    def collate_fn(batch):
        """自定义批处理函数"""
        hrtfs = torch.stack([item["hrtf"] for item in batch])
        positions = torch.stack([item["position"] for item in batch])
        return {
            "hrtf": hrtfs,
            "position": positions,  # shape: [batch, 2]
        }


class FullPipelineDataSet(SonicomDataSet):
    """
    同时输出 HRTF、position、左耳图像、右耳图像
    用于 VAE-DNN-CVAE 端到端评估
    """
    def __init__(self, hrtf_files, left_voxels, right_voxels,
                 status="test",
                 use_diff=False, mode="left",
                 provided_mean_left=None, provided_mean_right=None):
        super().__init__(
            hrtf_files=hrtf_files,
            left_voxels=left_voxels,
            right_voxels=right_voxels,
            status=status,
            calc_mean=False,
            use_diff=use_diff,
            inputform="image",
            mode=mode,
            provided_mean_left=provided_mean_left,
            provided_mean_right=provided_mean_right,
        )

    def __getitem__(self, idx):
        file_idx = idx
        position_idx = np.arange(self.positions_per_subject)

        with h5py.File(self.hrtf_files[file_idx], 'r') as data:
            hrtf = self._get_hrtf(data, position_idx)
            # 直接返回原始方位角和俯仰角 (度), [positions, 2]
            position = torch.tensor(data["theta"][:, position_idx].T, dtype=torch.float32)

        left_image = self._load_data(self.left_voxel_paths[file_idx], is_right=False) if self.left_voxel_paths else None
        right_image = self._load_data(self.right_voxel_paths[file_idx], is_right=True) if self.right_voxel_paths else None

        return {
            "hrtf": hrtf,
            "position": position,
            "left_voxel": left_image,
            "right_voxel": right_image,
        }

    @staticmethod
    def collate_fn(batch):
        hrtfs = torch.stack([item["hrtf"] for item in batch])
        positions = torch.stack([item["position"] for item in batch])
        left_voxels = torch.stack([item["left_voxel"] for item in batch]) if batch[0]["left_voxel"] is not None else None
        right_voxels = torch.stack([item["right_voxel"] for item in batch]) if batch[0]["right_voxel"] is not None else None
        return {
            "hrtf": hrtfs,
            "position": positions,
            "left_voxel": left_voxels,
            "right_voxel": right_voxels,
        }
    