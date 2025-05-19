import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from TestNet import TestNet as threeDResnetANP
from TestNet import ResNet3D as threeDResnet
from TestNet import ResNet2DClassifier as twoDResnet
from new_dataset import SonicomDataSet, SingleSubjectDataSet
from utils import split_dataset
import numpy as np
import matplotlib.pyplot as plt
from AE import HRTF_VQVAE
from AEconfig import pos_dim_for_each_row, \
    num_hrtf_rows, width_per_hrtf_row, transformer_encoder_settings, decoder_mlp_layers, encoder_out_vec_num, \
    num_codebook_embeddings, commitment_cost_beta

# 设备配置
# current_model = "3DResNet" # ["3DResNetANP", "3DResNet", "2DResNetANP", "2DResNet"]
weightname = "best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
usediff = False  # 是否使用差分数据

weightdir = "./CNNweights"
ear_dir = "Ear_image_gray_Wi"
isANP = False
if os.path.exists(weightdir) is False:
    os.makedirs(weightdir)
modelpath = f"{weightdir}/{weightname}"
# positions_chosen_num = 793
model = twoDResnet().to(device)
inputform = "image"

model.load_state_dict(torch.load(modelpath, map_location=device, weights_only=True))
print("Load model from", modelpath)
hrtf_encoder = HRTF_VQVAE(
    hrtf_row_width=width_per_hrtf_row,
    hrtf_num_rows=num_hrtf_rows,
    encoder_out_vec_num=encoder_out_vec_num, # 编码器输出序列长度
    encoder_transformer_config=transformer_encoder_settings,
    num_embeddings=num_codebook_embeddings,
    commitment_cost=commitment_cost_beta,
    pos_dim_per_row=pos_dim_for_each_row,
    decoder_mlp_hidden_dims=decoder_mlp_layers
).to(device)
hrtf_encoder.load_state_dict(torch.load("HRTFAEweights/model-vqvae-30oyy.pth", map_location=device, weights_only=True))
print("Load hrtf_encoder from", "HRTFAEweights/model-vqvae-30oyy.pth")
def evaluate_one_hrtf(model, hrtf_encoder, test_loader):
    model.eval()
    hrtf_encoder.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            # 数据迁移到设备
            targets = batch["hrtf"]
            meanloghrtf = batch["meanlog"].to(device)  # [batch]
            pos = batch["position"].to(device)
            right_picture = batch["right_voxel"].to(device)
            pred, _ = model(right_picture, device=device) # [batch_size, 18]
            pred = pred.reshape(-1, 2, 3, 3)
            pred = pred.permute(1, 0, 2, 3) # [2, batch_size, 3, 3]
            # pred =torch.randint_like(pred, low=0, high=num_codebook_embeddings) # 随机生成索引以测试
            zq = hrtf_encoder.vq_layer.get_output_from_indices(pred)
            outputs = hrtf_encoder.decoder(zq, pos).squeeze(1)  # [batch_size, 90]
            # 添加epsilon防止log(0)
            targets = targets + 1e-8

            # 转换到对数域 (dB)
            log_target = 20 * torch.log10(targets)
            if usediff:
                pred = torch.abs(outputs + meanloghrtf)
            else:
                pred = torch.abs(outputs)
            log_target = torch.abs(log_target)

            # 将当前batch的结果添加到列表
            all_preds.append(pred)
            all_targets.append(log_target)

    # 将所有batch的结果拼接成两个大矩阵
    final_preds = torch.cat(all_preds, dim=0)  # [total_samples, n_frequencies]
    final_targets = torch.cat(all_targets, dim=0).to(device)  # [total_samples, n_frequencies]

    return final_preds, final_targets

res_list = []
pred_list = []
true_list = []


hrtf_dir = "FFT_HRTF_Wi"

dataset_paths = split_dataset(ear_dir, "FFT_HRTF_Wi",inputform=inputform)
# 获取各个数据集
right_test = dataset_paths['right_test']


# 实例化训练数据集
train_dataset = SonicomDataSet(
    dataset_paths["train_hrtf_list"],
    dataset_paths["left_train"],
    dataset_paths["right_train"],
    use_diff=usediff,
    calc_mean=True,
    status="test", # 因为这里希望坐标是按顺序输入的
    inputform=inputform,
    mode="left"
)


# 实例化验证数据集
log_mean_hrtf_left = train_dataset.log_mean_hrtf_left
log_mean_hrtf_right = train_dataset.log_mean_hrtf_right


for hrtfid in range(1, len(right_test)+1):  # 选择计算第几个HRTF的LSD
    val_dataset = SingleSubjectDataSet( dataset_paths["test_hrtf_list"],
                                        dataset_paths["left_test"],
                                        dataset_paths["right_test"],
                                        mode="right",
                                        train_log_mean_hrtf_left=log_mean_hrtf_left,
                                        train_log_mean_hrtf_right=log_mean_hrtf_right,
                                        subject_id=hrtfid,
                                        inputform=inputform
                                        )
    dataloader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            collate_fn=val_dataset.collate_fn
                            )
    pred_log_hrtf, true_log_hrtf = evaluate_one_hrtf(model, hrtf_encoder, dataloader)
    pred_list.append(pred_log_hrtf)
    true_list.append(true_log_hrtf)
    lsd = torch.sqrt(torch.mean((pred_log_hrtf - true_log_hrtf) ** 2)).item()
    res_list.append(lsd)
    print(f"LSD of HRTF {hrtfid}:", lsd)

print(f"Mean LSD: {np.mean(res_list)}")
pred_tensor = torch.cat(pred_list, dim=0)
true_tensor = torch.cat(true_list, dim=0)

freq_list = np.linspace(0, 89, 90)  # 获取频率列表
freq_list = 48000 /256 * freq_list  # 计算频率值
# 存储每个频率点的平均LSD
avg_lsd_per_freq = np.zeros(len(freq_list))
for freq_idx in range(len(freq_list)):
    # 计算平均LSD
    LSDvec = torch.sqrt(torch.mean((pred_tensor[:, :, freq_idx] - true_tensor[:, :, freq_idx]) ** 2, dim=1))
    avg_lsd_per_freq[freq_idx] = torch.mean(LSDvec).item()
    # print(f"Avg LSD of freq point {freq_idx}:{avg_lsd_per_freq[freq_idx]}")
print("\n-----------------contrast with mean HRTF-----------------\n")
res_list_mean = []
# 将均值转为tensor
log_mean_hrtf_left = torch.tensor(np.abs(log_mean_hrtf_left), dtype=torch.float32).to(device)
log_mean_hrtf_left = log_mean_hrtf_left.unsqueeze(0)  # 添加batch维度

for hrtfid in range(1, len(right_test)+1):  # 选择计算第几个HRTF的LSD
    # 之前已经计算预测HRTF和真实HRTF之间LSD，
    # 现在计算平均HRTF和真实HRTF之间LSD
    lsd_of_mean = torch.sqrt(torch.mean((log_mean_hrtf_left - true_tensor[hrtfid-1, :, :]) ** 2)).item()
    res_list_mean.append(lsd_of_mean)
    print(f"LSD between mean HRTF and HRTF {hrtfid}:", lsd_of_mean)

print(f"Mean LSD of mean HRTF: {np.mean(res_list_mean)}")

avg_lsd_per_freq_of_mean = np.zeros(len(freq_list))
for freq_idx in range(len(freq_list)):
    # 计算平均LSD
    LSDvec = torch.sqrt(torch.mean((log_mean_hrtf_left[:,:,freq_idx] - true_tensor[:, :, freq_idx]) ** 2, dim=1))
    avg_lsd_per_freq_of_mean[freq_idx] = torch.mean(LSDvec).item()
    # print(f"Avg LSD of freq point {freq_idx}:{avg_lsd_per_freq_of_mean[freq_idx]}")


# 绘制频率-LSD图
plt.figure(figsize=(10, 6))
plt.semilogx(freq_list, avg_lsd_per_freq, 'b-o')
plt.semilogx(freq_list, avg_lsd_per_freq_of_mean, 'r-o')
plt.title('Frequency vs LSD')
plt.xlabel('Frequency')
plt.ylabel('LSD (dB)')
plt.grid(True, which="both", ls="--")
plt.legend(['LSD of predicted HRTF', 'LSD of mean HRTF'])
# 保存频率-LSD图片
plt.savefig("LSD_per_frequency.png")  # 保存频率-LSD图片

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
plt.savefig("LSD_comparison.png")  # 保存LSD对比图
plt.show(block=True)  # 显示图像，阻止脚本结束时关闭图像窗口

# 保存LSD结果
np.savetxt("LSD_results.txt", res_list, fmt='%.6f')