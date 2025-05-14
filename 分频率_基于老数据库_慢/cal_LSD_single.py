from cgi import test
import numpy as np
import h5py
import torch
from torchvision import transforms
from utils import *
from TestNet import TestNet as create_model
import matplotlib.pyplot as plt
import os
import argparse
from torchvision import transforms
from new_dataset import SonicomDataSet, SingleSubjectDataSet

# from utils import read_split_data, train_one_epoch, evaluate

model_path = "D:\大学\大三下\大创项目\Sonicom--2d\weights\model-1.pth"
# model_path = "123"

target_index = 50

def evaluate_one_hrtf(model, test_loader, target_index = 50):
    model.eval()

    all_preds = []
    all_targets = []


    with torch.no_grad():
        for batch in test_loader:
            # 数据迁移到设备
            imageleft = batch["left_image"].to(device) #[batch, 3, 224, 224]
            imageright = batch["right_image"].to(device)
            position = batch["position"].squeeze(1).to(device) #[batch, 2]
            # mark
            targets = batch["hrtf"].squeeze(1)[:, target_index].unsqueeze(-1).to(device)  # [batch]
            meanloghrtf = batch["meanlog"].unsqueeze(-1)[:, target_index].to(device)  # [batch]
            #
            # targets = batch["hrtf"].squeeze(1)[:, :].to(device)  # [batch]
            # meanloghrtf = batch["meanlog"][:, :].to(device)  # [batch]

            # 前向传播
            outputs = model(imageleft, imageright, position)  # [batch]

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
    # return all_preds, all_targets



if __name__ == '__main__':
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符 jx_vit_base_patch16_224_in21k-e5005f0a.pth
    parser.add_argument('--weights', type=str, default='./jx_vit_base_patch16_224_in21k-e5005f0a.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    args = parser.parse_args()

    # 2. 模型和训练配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)  # 使用之前定义的网络结构

    # 如果需要加载预训练模型
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Load model from", model_path)
    res_list = []
    pred_list = []
    true_list = []

    image_dir = "Ear_image_gray"
    hrtf_dir = "FFT_HRTF"

    dataset_paths = split_dataset(image_dir, hrtf_dir)
    # 获取各个数据集
    train_hrtf_list = dataset_paths['train_hrtf_list']
    test_hrtf_list = dataset_paths['test_hrtf_list']
    left_train = dataset_paths['left_train']
    right_train = dataset_paths['right_train']
    left_test = dataset_paths['left_test']
    right_test = dataset_paths['right_test']
    data_transform = {
    "train": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 强制转换为单通道
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 单通道标准化
    ]),
    "val": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 强制转换为单通道
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 单通道标准化
    ])
    }

    # 实例化训练数据集
    train_dataset = SonicomDataSet(hrtf_files=train_hrtf_list,
                            left_images=left_train,
                            right_images=right_train,
                            transform=data_transform["train"],
                            mode="left")

    # 实例化验证数据集
    log_mean_hrtf_left = train_dataset.log_mean_hrtf_left
    log_mean_hrtf_right = train_dataset.log_mean_hrtf_right

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4]) # 加载数据所用进程数，若大于1需要时间准备进程，不是越大越好

    for hrtfid in range(1, len(left_test)+1):  # 选择计算第几个HRTF的LSD
        val_dataset = SingleSubjectDataSet(hrtf_files=test_hrtf_list,
                                           left_images=left_test,
                                           right_images=right_test,
                                           transform=data_transform["val"],
                                           mode="left",
                                           train_log_mean_hrtf_left=log_mean_hrtf_left,
                                           train_log_mean_hrtf_right=log_mean_hrtf_right,
                                           subject_id = hrtfid
                                           )
        dataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers= 0,
                                             collate_fn=val_dataset.collate_fn
                                                 )
        pred_log_hrtf, true_log_hrtf = evaluate_one_hrtf(model, dataloader,target_index = target_index)
        pred_list.append(pred_log_hrtf)
        true_list.append(true_log_hrtf.squeeze(1)) # mark
        lsd = torch.sqrt(torch.mean((pred_log_hrtf - true_log_hrtf) ** 2)).item()
        res_list.append(lsd)
        print(f"LSD of HRTF {hrtfid}:", lsd)

    print(f"Mean LSD: {np.mean(res_list)}")
    pred_tensor = torch.stack(pred_list, dim=0)
    true_tensor = torch.stack(true_list, dim=0)

    freq_list = np.linspace(0, 1, 1)  # 获取频率列表
    freq_list = 48000 /256 * freq_list  # 计算频率值
    # 存储每个频率点的平均LSD mark
    # avg_lsd_per_freq = np.zeros(len(freq_list))
    # for freq_idx in range(len(freq_list)):
    #     # 计算平均LSD
    #     LSDvec = torch.sqrt(torch.mean((pred_tensor[:, freq_idx] - true_tensor[:, freq_idx]) ** 2, dim=1))
    #     avg_lsd_per_freq[freq_idx] = torch.mean(LSDvec).item()
    #     print(f"Avg LSD of freq point {freq_idx}:{avg_lsd_per_freq[freq_idx]}")
    print("\n-----------------contrast with mean HRTF-----------------\n")
    res_list_mean = []
    # 将均值转为tensor
    log_mean_hrtf_left = torch.tensor(np.abs(log_mean_hrtf_left[:,target_index]), dtype=torch.float32).to(device)
    log_mean_hrtf_left = log_mean_hrtf_left.unsqueeze(0)  # 添加batch维度
for hrtfid in range(1, len(left_test)+1):
    # 之前已经计算预测HRTF和真实HRTF之间LSD，
    # 现在计算平均HRTF和真实HRTF之间LSD
    lsd_of_mean = torch.sqrt(torch.mean((log_mean_hrtf_left - true_tensor[hrtfid-1,:]) ** 2)).item()
    res_list_mean.append(lsd_of_mean)
    print(f"LSD between mean HRTF and HRTF {hrtfid}:", lsd_of_mean)

print(f"Mean LSD of mean HRTF: {np.mean(res_list_mean)}")

avg_lsd_per_freq_of_mean = np.zeros(len(freq_list))
# mark
# for freq_idx in range(len(freq_list)):
#     # 计算平均LSD
#     LSDvec = torch.sqrt(torch.mean((log_mean_hrtf_left[:,freq_idx] - true_tensor[:, freq_idx]) ** 2, dim=0))
#     avg_lsd_per_freq_of_mean[freq_idx] = torch.mean(LSDvec).item()
#     # print(f"Avg LSD of freq point {freq_idx}:{avg_lsd_per_freq_of_mean[freq_idx]}")


# # 绘制频率-LSD图
# plt.figure(figsize=(10, 6))
# plt.semilogx(freq_list, avg_lsd_per_freq, 'b-o')
# plt.semilogx(freq_list, avg_lsd_per_freq_of_mean, 'r-o')
# plt.title('Frequency vs LSD')
# plt.xlabel('Frequency')
# plt.ylabel('LSD (dB)')
# plt.grid(True, which="both", ls="--")
# plt.legend(['LSD of predicted HRTF', 'LSD of mean HRTF'])
# # 保存频率-LSD图片
# plt.savefig("LSD_per_frequency.png")  # 保存频率-LSD图片

# print(res_list)
#绘制LSD对比图
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(res_list)+1), res_list, 'b-o', label='LSD of predicted HRTF')
plt.plot(range(1, len(res_list_mean)+1), res_list_mean, 'r-o', label='LSD of mean HRTF')
plt.title('LSD Comparison')
plt.xlabel('HRTF ID')
plt.ylabel('LSD (dB)')
plt.legend()
plt.grid(True, which="both", ls="--")
# 保存LSD对比图
# plt.savefig("LSD_comparison.png")  # 保存LSD对比图
plt.show(block=True)  # 显示图像，阻止脚本结束时关闭图像窗口

# 保存LSD结果
np.savetxt("LSD_results.txt", res_list, fmt='%.6f')