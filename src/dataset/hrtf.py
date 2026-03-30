import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py

from ..utils import calculate_hrtf_mean
from PIL import Image
from torchvision import transforms

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
        self.left_voxel_paths = left_voxels
        self.right_voxel_paths = right_voxels
        self.status = status
        self.mode = mode
        self.inputform = inputform
        self.use_diff = use_diff
        self.feature = provided_feature

        # HRTF 均值
        if calc_mean:
            self.log_mean_hrtf_left = 20 * np.log10(calculate_hrtf_mean(self.hrtf_files, whichear='left'))
            self.log_mean_hrtf_right = 20 * np.log10(calculate_hrtf_mean(self.hrtf_files, whichear='right'))
        else:
            self.log_mean_hrtf_left = provided_mean_left
            self.log_mean_hrtf_right = provided_mean_right

        # 获取每个样本的方位数
        with h5py.File(self.hrtf_files[0], 'r') as f:
            self.positions_per_subject = f["F_left"].shape[0]

        # 设置 transforms
        self.image_transform_train = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.Resize((256, 256)),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.image_transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.hrtf_files)

    def __getitem__(self, idx):
        file_idx = idx
        position_idx = (sorted(np.random.choice(self.positions_per_subject, self.positions_per_subject, replace=False))
                        if self.status == "train" else np.arange(self.positions_per_subject))

        # 载入 HRTF
        with h5py.File(self.hrtf_files[file_idx], 'r') as data:
            hrtf = self._get_hrtf(data, position_idx)
            original_position_rad = torch.deg2rad(torch.tensor(data["theta"][:, position_idx].T).type(torch.float32))
            position = torch.stack([
                torch.sin(original_position_rad[:, 0]),
                torch.cos(original_position_rad[:, 0]),
                torch.sin(original_position_rad[:, 1])
            ], dim=1)

        # 载入图像或体素
        left_voxel = self._load_data(self.left_voxel_paths[file_idx], is_right=False) if self.left_voxel_paths else None
        right_voxel = self._load_data(self.right_voxel_paths[file_idx], is_right=True) if self.right_voxel_paths else None
        feature = self.feature[file_idx, :] if self.feature is not None else None

        return {
            "hrtf": hrtf,
            "position": position,
            "left_voxel": left_voxel,
            "right_voxel": right_voxel,
            "feature": feature,
        }

    def _get_hrtf(self, data, position_idx):
        if self.mode == "left":
            hrtf_data = data["F_left"][position_idx, :]
            mean_hrtf = self.log_mean_hrtf_left[position_idx, :] if self.use_diff else 0
            return torch.tensor(20 * np.log10(hrtf_data) - mean_hrtf).type(torch.float32)
        elif self.mode == "right":
            hrtf_data = data["F_right"][position_idx, :]
            mean_hrtf = self.log_mean_hrtf_right[position_idx, :] if self.use_diff else 0
            return torch.tensor(20 * np.log10(hrtf_data) - mean_hrtf).type(torch.float32)
        else:  # both
            left = 20 * np.log10(data["F_left"][position_idx, :]) - self.log_mean_hrtf_left[position_idx, :]
            right = 20 * np.log10(data["F_right"][position_idx, :]) - self.log_mean_hrtf_right[position_idx, :]
            return torch.tensor(np.concatenate([left, right], axis=1)).type(torch.float32)

    def _load_data(self, path, is_right=False):
        if self.inputform == "image":
            image = Image.open(path).convert('L')
            if is_right:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            transform = self.image_transform_train if self.status == "train" else self.image_transform_test
            return transform(image)
        else:
            voxel = np.load(path)
            if is_right:
                voxel = np.flip(voxel, axis=1).copy()

            # if self.status == "train":
            #     # 数据增强：翻转、旋转、加噪声
            #     if random.random() < 0.5:
            #         voxel = np.flip(voxel, axis=0).copy()
            #     if random.random() < 0.5:
            #         voxel = np.flip(voxel, axis=2).copy()
            #     k = random.randint(0, 3)
            #     voxel = np.rot90(voxel, k, axes=(0, 1)).copy()
            #     # if random.random() < 0.3:
            #     #     voxel += np.random.normal(0, 0.02, voxel.shape)
            #     #     voxel = np.clip(voxel, 0, 1)
            return torch.tensor(voxel, dtype=torch.float32).unsqueeze(0)

    @staticmethod
    def collate_fn(batch):
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
