import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils import *
from TestNet import TestNet
import matplotlib.pyplot as plt
from sklearn import preprocessing

# 全局配置
TARGET_FREQ_IDX = 10  # 要评估的频率点索引（0-38）
BATCH_SIZE = 128


# 1. 数据集（仅加载目标频率）
class OneImageDataset(Dataset):
    def __init__(
            self, imglist, hrtflist, hrtfid, target_freq_idx=0,
            transform=transforms.ToTensor(), run_validation=True
    ):
        super().__init__()
        self.transform = transform
        self.image_file_names = imglist
        self.imageNumber = len(imglist) // 2
        self.hrtf_file_names = hrtflist
        self.hrtfid = hrtfid
        self.target_freq_idx = target_freq_idx

        # 验证HRTF文件
        if run_validation:
            valid_hrtf_files = []
            for f in self.hrtf_file_names:
                with h5py.File(f, 'r') as data:
                    hrtf = torch.tensor(data["abspHRTF"][:])
                    if not torch.isnan(hrtf).any():
                        valid_hrtf_files.append(f)
            self.hrtf_file_names = valid_hrtf_files

        # 计算目标频率的全局均值
        full_mean = 20 * np.log10(calculate_hrtf_mean(self.hrtf_file_names))  # [num_positions, 39]
        self.mean_log_hrtf = full_mean[:, self.target_freq_idx]  # [num_positions]

        # 获取方位数
        with h5py.File(self.hrtf_file_names[0], 'r') as data:
            self.positionNumber = data["abspHRTF"].shape[0]

    def __getitem__(self, index: int):
        idx1 = self.hrtfid - 1  # 个体索引
        idx2 = index % self.positionNumber  # 方位索引

        # 读取目标频率的HRTF和均值
        with h5py.File(self.hrtf_file_names[idx1], 'r') as data:
            hrtf_value = 20 * np.log10(data["abspHRTF"][idx2, self.target_freq_idx])  # 标量
            mean_value = self.mean_log_hrtf[idx2]  # 标量

            position = torch.tensor(data["theta"][:, idx2].T).flatten().float()

            # 图像索引修正
            if idx2 > self.positionNumber // 2 - 1:
                img_idx = idx1 + self.imageNumber
            else:
                img_idx = idx1

            image = Image.open(self.image_file_names[img_idx]).convert("L")
            image = self.transform(image)  # [1, H, W]

        return {
            "hrtf": torch.tensor([hrtf_value]).float(),  # [1]
            "meanlog": torch.tensor([mean_value]).float(),  # [1]
            "position": position,  # [position_dim]
            "image": image  # [1, H, W]
        }

    def __len__(self):
        return self.positionNumber




# 3. 评估函数
def evaluate_single_freq(model, test_loader, device):
    model.eval()
    total_se = 0.0  # 平方误差累加
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            # 数据迁移到设备
            image = batch["image"].to(device)
            pos = batch["position"].to(device)
            hrtf = batch["hrtf"].flatten().to(device)  # [batch]
            mean_log = batch["meanlog"].flatten().to(device)  # [batch]

            # 前向传播
            pred_diff = model(image, pos).squeeze()  # [batch]

            # 还原为绝对HRTF（对数域）
            pred_log = pred_diff + mean_log  # 模型预测的是差值
            true_log = hrtf  # 真实值已经是20*log10后的结果

            # 累加误差
            total_se += torch.sum((pred_log - true_log) ** 2)
            total_samples += image.size(0)

    # 计算LSD
    return torch.sqrt(total_se / total_samples).item()


# 4. 主流程
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    net = TestNet().to(device)
    model_path = "model2.pth"
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")

    # 评估5个HRTF个体
    res = []
    for hrtf_id in range(1, 6):
        dataset = OneImageDataset(
            full_test_image_path_list,
            full_test_hrtf_path_list,
            hrtfid=hrtf_id,
            target_freq_idx=TARGET_FREQ_IDX
        )
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        lsd = evaluate_single_freq(net, loader, device)
        res.append(lsd)
        print(f"HRTF{hrtf_id} 频率{TARGET_FREQ_IDX} LSD: {lsd:.4f} dB")

    print(f"平均 LSD: {np.mean(res):.4f} ± {np.std(res):.4f} dB")
