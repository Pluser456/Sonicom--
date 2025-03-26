import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils import *
from vit_model import vit_base_patch16_224_in21k as create_model
import matplotlib.pyplot as plt

# 全局配置
TARGET_FREQ_IDX = 1 # 要评估的频率点索引（0-38）
BATCH_SIZE = 128

model_path = "model15.pth"

# 1. 数据集（仅加载目标频率）
class OneImageDataset(Dataset):
    def __init__(
            self, hrtfid, target_freq_idx=0,
            transform=None, run_validation=True, mode= "left"
    ):
        super(OneImageDataset,self).__init__()
        self.mode = mode
        self.left_image_path = left_test
        self.right_image_path = right_test
        self.test_people_num = test_people_num
        self.hrtf_file_names = test_hrtf_list

        # self.left_image_path = left_train
        # self.right_image_path = right_train
        # self.test_people_num = train_people_num
        # self.hrtf_file_names = train_hrtf_list

        self.transform = transform
        # self.image_file_names = imglist
        # self.imageNumber = len(imglist) // 2
        # self.hrtf_file_names = hrtflist
        self.hrtfid = hrtfid
        self.target_freq_idx = target_freq_idx

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
        full_log_mean_hrtf_left = 20 * np.log10(calculate_hrtf_mean(self.hrtf_file_names,whichear='left')) # 用左耳HRTF计算均值
        full_log_mean_hrtf_right = 20 * np.log10(calculate_hrtf_mean(self.hrtf_file_names,whichear='right'))

        self.positionNumber = h5py.File(self.hrtf_file_names[0])["F_left"].shape[0]  # 方位数
        # self.log_mean_hrtf_left = full_log_mean_hrtf_left[:, self.target_freq_idx]  # [num_positions]
        # self.log_mean_hrtf_right = full_log_mean_hrtf_right[:, self.target_freq_idx]  # [num_positions]
        self.log_mean_hrtf_left = full_log_mean_hrtf_left
        self.log_mean_hrtf_right = full_log_mean_hrtf_right

    def __getitem__(self, index: int):
        # 这里不确定
        idx1 = self.hrtfid - 1  # 个体索引
        idx2 = index % self.positionNumber  # 方位索引

        # 读取目标频率的HRTF和均值
        with h5py.File(self.hrtf_file_names[idx1], 'r') as data:
            if self.mode == "left":
                hrtf = (torch.tensor(data["F_left"][idx2, :]).reshape(1,-1).type(torch.float32))
                mean_value = (torch.tensor(self.log_mean_hrtf_left[idx2,:] )).type(torch.float32)  # 标量
                # print(mean_value.shape)

            elif self.mode == "right":
                hrtf = (torch.tensor(data["F_right"][idx2, :]).reshape(1,-1).type(torch.float32))
                mean_value = (torch.tensor(self.log_mean_hrtf_right[idx2,:] )).type(torch.float32)  # 标量
            else:
                hrtf_left = (torch.tensor(data["F_left"][idx2, :] ).reshape(1,-1).type(torch.float32))
                hrtf_right = (torch.tensor(data["F_right"][idx2, :] ).reshape(1,-1).type(torch.float32))
                mean_value_left = (torch.tensor(self.log_mean_hrtf_left[idx2,:] )).type(torch.float32)
                mean_value_right = (torch.tensor(self.log_mean_hrtf_right[idx2,:] )).type(torch.float32)  # 标量
                hrtf = torch.cat([hrtf_left, hrtf_right], dim=1)
                mean_value = torch.cat([mean_value_left, mean_value_right], dim=1)

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
            "hrtf": hrtf,  # [1]
            "meanlog": mean_value,  # [1]
            "position": position,  # [position_dim]
            "imageleft": image_left,
            "imageright": image_right
        }

    def __len__(self) -> int:
        return self.positionNumber


# 2. 模型和训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = create_model().to(device)  # 使用之前定义的网络结构

# 如果需要加载预训练模型
if os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print("Load model from", model_path)


# 3. 评估函数
def evaluate_one_hrtf(model, test_loader):
    model.eval()

    all_preds = []
    all_targets = []


    with torch.no_grad():
        for batch in test_loader:
            # 数据迁移到设备
            imageleft = batch["imageleft"].to(device)
            imageright = batch["imageright"].to(device)
            position = batch["position"].to(device)
            hrtf = batch["hrtf"].to(device)  # [batch]
            meanloghrtf = batch["meanlog"].to(device)  # [batch]

            # 前向传播
            outputs = model(imageleft,imageright, position)  # [batch]
            targets = hrtf.squeeze(1)
            # 添加epsilon防止log(0)
            targets = targets + 1e-8

            # 转换到对数域 (dB)
            log_target = 20 * torch.log10(targets)
            pred = torch.abs(outputs + meanloghrtf)
            log_target = torch.abs(log_target)

            # 将当前batch的结果添加到列表
            all_preds.append(pred)
            all_targets.append(log_target)
            # print(outputs.shape)
            # print(meanloghrtf.shape)


    # 将所有batch的结果拼接成两个大矩阵
    final_preds = torch.cat(all_preds, dim=0)  # [total_samples, n_frequencies]
    final_targets = torch.cat(all_targets, dim=0)  # [total_samples, n_frequencies]

    return final_preds, final_targets


res_list = []
pred_list = []
true_list = []
for hrtfid in range(1, 13):  # 选择计算第几个HRTF的LSD
    dataloader = DataLoader(
        OneImageDataset(hrtfid),
        batch_size=128,
        shuffle=False,
    )
    pred_log_hrtf, true_log_hrtf = evaluate_one_hrtf(net, dataloader)
    pred_list.append(pred_log_hrtf)
    true_list.append(true_log_hrtf)
    lsd = torch.sqrt(torch.mean((pred_log_hrtf - true_log_hrtf) ** 2)).item()
    res_list.append(lsd)
    print(f"LSD of HRTF {hrtfid}:", lsd)
print(f"Mean LSD: {np.mean(res_list)}")
pred_tensor = torch.stack(pred_list, dim=0)
true_tensor = torch.stack(true_list, dim=0)

freq_list = np.linspace(0,107,108)  # 获取频率列表
# 存储每个频率点的平均LSD
avg_lsd_per_freq = np.zeros(len(freq_list))
for freq_idx in range(len(freq_list)):
    # 计算平均LSD
    LSDvec = torch.sqrt(torch.mean((pred_tensor[:, :, freq_idx] - true_tensor[:, :, freq_idx]) ** 2, dim=1))
    avg_lsd_per_freq[freq_idx] = torch.mean(LSDvec).item()
    print(f"Avg LSD of freq point {freq_idx}:{avg_lsd_per_freq[freq_idx]}")


# 绘制频率-LSD图
plt.figure(figsize=(10, 6))
plt.semilogx(freq_list, avg_lsd_per_freq, 'b-o')
plt.title('Frequency vs LSD')
plt.xlabel('Frequency')
plt.ylabel('LSD (dB)')
plt.grid(True, which="both", ls="--")
plt.show()