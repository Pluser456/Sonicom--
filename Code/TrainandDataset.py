import h5py
import torch
import numpy as np
import os
from TestNet import TestNet
from torch.utils.data import Dataset, DataLoader
from torch import nn
from utils import *
from PIL import Image
from torchvision import transforms

freq_point = 15

# 1. 数据准备
class TrainDataset(Dataset):
    def __init__(self, transform=None, run_validation=True, mode= "both") -> None:
        '''
        - transform: 数据预处理
        - run_validation: 是否验证HRTF文件
        - mode: 数据集输出模式，不同的模式将输出左耳或右耳的HRTF。
        可选值为 "left" 、 "right" 或 "both"，默认为 "both"
        '''
        super(TrainDataset, self).__init__()
        self.mode = mode
        self.left_image_path = left_train
        self.right_image_path = right_train
        self.train_people_num = train_people_num
        self.hrtf_file_names = train_hrtf_list 

        # 初始化预处理
        self.transform = transforms.Compose([
            transforms.ToTensor()  # 灰度图转单通道Tensor
        ]) if transform is None else transform

        # 验证HRTF文件
        if run_validation:
            valid_hrtf_file_names = []
            invalid_hrtf_file_names = []
            for hrtf_file_name in self.hrtf_file_names:
                data = h5py.File(hrtf_file_name)
                lefthrtf = torch.tensor(data["F_left"][:])
                righthrtf = torch.tensor(data["F_right"][:])
                if not np.isnan(np.sum(lefthrtf.cpu().data.numpy())+np.sum(righthrtf.cpu().data.numpy())):
                    valid_hrtf_file_names.append(hrtf_file_name)
                else:
                    invalid_hrtf_file_names.append(hrtf_file_name)
            self.hrtf_file_names = valid_hrtf_file_names

            if len(invalid_hrtf_file_names) > 0:
                print(f"Invalid HRTF files: {invalid_hrtf_file_names}")

        # HRTF均值计算
        self.log_mean_hrtf_left = 20 * np.log10(calculate_hrtf_mean(self.hrtf_file_names,whichear='left')) # 用左耳HRTF计算均值
        self.log_mean_hrtf_right = 20 * np.log10(calculate_hrtf_mean(self.hrtf_file_names,whichear='right'))
        self.positionNumber = h5py.File(self.hrtf_file_names[0])["F_left"].shape[0]  # 方位数

    def __getitem__(self, batch_index: int):
        # 计算索引
        idx1 = batch_index // self.positionNumber  # 个体索引
        # print(len(self.hrtf_file_names))
        # print((self.hrtf_file_names))
        idx2 = batch_index % self.positionNumber  # 方位索引

        # 读取HRTF和方位角
        data = h5py.File(self.hrtf_file_names[idx1])
        if self.mode == "left":
            hrtf = (torch.tensor(20 * np.log10(data["F_left"][idx2, :]) - self.log_mean_hrtf_left[idx2, :]).reshape(1, -1).type(torch.float32))
        elif self.mode == "right":
            hrtf = (torch.tensor(20 * np.log10(data["F_right"][idx2, :]) - self.log_mean_hrtf_right[idx2, :]).reshape(1, -1).type(torch.float32))
        else:
            hrtf_left = (torch.tensor(20 * np.log10(data["F_left"][idx2, :]) - self.log_mean_hrtf_left[idx2, :]).reshape(1, -1).type(torch.float32))
            hrtf_right = (torch.tensor(20 * np.log10(data["F_right"][idx2, :]) - self.log_mean_hrtf_right[idx2, :]).reshape(1, -1).type(torch.float32))
            hrtf = torch.cat([hrtf_left, hrtf_right], dim=1)
        position = (
            torch.tensor(data["theta"][:, idx2].T).reshape(1, -1).type(torch.float32)
        )  # position * f
        # 读取图像
        image_idx = idx1
        image_left = Image.open(self.left_image_path[image_idx])
        image_right = Image.open(self.right_image_path[image_idx])
        image_left = self.transform(image_left)
        image_right = self.transform(image_right)
        return {
            "diffloghrtf": hrtf,
            "position": position,
            "imageleft": image_left,
            "imageright": image_right
        }

    def __len__(self) -> int:
        return self.train_people_num * self.positionNumber


# 2. 模型和训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = TrainDataset(mode="left")
dataloader = DataLoader(train_dataset, batch_size=28, shuffle=True)

# 初始化模型
net = TestNet().to(device)

# 加载预训练模型（可选）
model_path = "model15.pth"  # 定义路径
if os.path.exists(model_path):
    try:
        net.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded pretrained model.")
    except:
        print("Failed to load pretrained model. Training from scratch.")
else:
    print("Training from scratch.")

# 定义损失和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)


# 3. 训练函数
def train(epoch):
    net.train()
    for batch_idx, sample_batch in enumerate(dataloader):
        # 数据迁移到设备
        imageleft = sample_batch["imageleft"].to(device)
        imageright = sample_batch["imageright"].to(device)
        pos = sample_batch["position"].to(device)
        target = sample_batch["diffloghrtf"].squeeze(1)[:, :].to(device)

        # 前向传播
        optimizer.zero_grad()
        output = net(imageleft,imageright, pos)
        loss = criterion(output, target)

        # 反向传播
        loss.backward()

        # +++ 新增梯度裁剪（添加在此处）+++
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)  # 限制梯度范数为5
        optimizer.step()

        # 打印日志
        if batch_idx % 10 == 0:
            # print(f"Epoch: {epoch} | Batch: {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
            print(f"Train Epoch: {epoch} [{batch_idx * len(target)}/{len(dataloader.dataset)}({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.4f}")

    # 保存模型
    if epoch % 1 == 0:
        torch.save(net.state_dict(), model_path)
        print(f"Model saved at epoch {epoch}")


torch.save(net.state_dict(), model_path)
print(f"Model saved at epoch 0")
# 4. 启动训练
for epoch in range(1, 3):

    train(epoch)
