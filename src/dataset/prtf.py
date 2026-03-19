import torch
import numpy as np
from torch.utils.data import Dataset
from ..utils import calculate_hrtf_mean
from PIL import Image
from torchvision import transforms
import h5py
import torch.nn.functional as F

class HRTFDataSet(Dataset):
    """使用预计算特征的数据集"""
    def __init__(self, hrtf_files, left_voxels, right_voxels, 
                 status="train",
                 calc_mean=True, use_diff=True, inputform="voxel",
                 mode="both", provided_mean_left=None, provided_mean_right=None, pos_num_per_batch=7):
        """
        Args:
            hrtf_files (list): HRTF文件路径列表
            left_voxels (list): 左耳体素路径列表
            right_voxels (list): 右耳体素路径列表
            device (str): 设备类型 - "cpu"/"cuda"
            status (str): 输出数据集模式 - "train"/"test"
            calc_mean (bool): 是否计算HRTF均值
            mode (str): 输出模式 - "left"/"right"/"both"
            provided_mean_left/provided_mean_right: 预计算的HRTF均值（如果calc_mean=False）
            pos_num_per_batch: 每个batch中包含的位置数量，最好是 2*3*7*61=2562 的约数
        """
        self.hrtf_files = hrtf_files
        self.left_voxel_paths = left_voxels
        self.right_voxel_paths = right_voxels
        self.status = status
        self.mode = mode
        self.inputform = inputform
        self.use_diff = use_diff
        self.pos_num_per_batch = pos_num_per_batch
        self.shuffled_indices = {}

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
