import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.TestNet import TestNet as threeDResnetANP
from src.models.TestNet import ResNet3DClassifier as threeDResnet
from src.models.TestNet import ResNet2DClassifier as twoDResnet
from src.dataset.hrtf import SonicomDataSet, SingleSubjectDataSet
from src.utils.data import split_dataset
import numpy as np
import matplotlib.pyplot as plt
from src.models.AE import HRTF_VQVAE
from src.models.AEconfig import pos_dim_for_each_row, \
    num_hrtf_rows, hrtf_row_len, transformer_encoder_settings, decoder_mlp_layers, encoder_out_vec_num, \
    num_codebook_embeddings, commitment_cost_beta, num_quantizers
from new_cal_LSD import evaluate_one_hrtf

print(f"当前码本大小为：{num_codebook_embeddings}")
# 设备配置
batch_size = 32
usediff = False  # 是否使用差分数据

current_model = "2DResNet" # ["3DResNetANP", "3DResNet", "2DResNetANP", "2DResNet"]
if current_model == "3DResNet":
    weightname = f"best_model_codebook_size_{num_codebook_embeddings}_3D.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weightdir = "./CNN3Dweights"
    ear_dir = "Ear_voxel_Wi"
    isANP = False
    model = threeDResnet(num_classes=num_codebook_embeddings).to(device)
    inputform = "voxel"
elif current_model == "2DResNet":
    weightname = f"best_model_codebook_size_{num_codebook_embeddings}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weightdir = "./CNNweights"
    ear_dir = "Ear_image_gray_Wi"
    isANP = False
    model = twoDResnet(num_classes=num_codebook_embeddings).to(device)
    inputform = "image"

if os.path.exists(weightdir) is False:
    os.makedirs(weightdir)
modelpath = f"{weightdir}/{weightname}"
# positions_chosen_num = 793

model.load_state_dict(torch.load(modelpath, map_location=device, weights_only=True))
print("Load model from", modelpath)
hrtf_encoder = HRTF_VQVAE(
    hrtf_row_len=hrtf_row_len,
    hrtf_num_rows=num_hrtf_rows,
    encoder_out_vec_num=encoder_out_vec_num, # 编码器输出序列长度
    encoder_transformer_config=transformer_encoder_settings,
    num_embeddings=num_codebook_embeddings,
    commitment_cost=commitment_cost_beta,
    pos_dim_per_row=pos_dim_for_each_row,
    num_quantizers=num_quantizers
).to(device)
hrtf_encoder.load_state_dict(torch.load(f"HRTFAEweights\diff_False_enc_n_1_enc_num_heads-6_num_encoder_layers-4_num_decoder_layers-15_dim_feedforward-512_dropout-0.05_codebook_size_{num_codebook_embeddings}_quan_n_3_120.pth", map_location=device, weights_only=True))
print("Load hrtf_encoder")

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
    mode="right"
)


# 实例化验证数据集
log_mean_hrtf_left = train_dataset.log_mean_hrtf_left
log_mean_hrtf_right = train_dataset.log_mean_hrtf_right

# 只取第一个测试集
hrtfid = 6
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
pred_log_hrtf = pred_log_hrtf.squeeze(0)
true_log_hrtf = true_log_hrtf.squeeze(0)

# 确保 pred_log_hrtf 和 true_log_hrtf 是 CPU tensor
pred_log_hrtf = pred_log_hrtf.cpu().numpy()  # 转换为 NumPy 数组
true_log_hrtf = true_log_hrtf.cpu().numpy()  # 转换为 NumPy 数组

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 12})  # 全局基础字号

# 1. 读取P00013_results.txt文件
data = np.loadtxt('P00013_results.txt', skiprows=1)  # 跳过表头行
indices = data[:, 0].astype(int) - 1  # 索引修正
angles = data[:, 1]                   # 角度(度)

# 2. 加载频率列表
freq_list = np.loadtxt(os.path.join('HRTF可视化', 'freq_data.txt'))
freq_list_kHz = freq_list / 1000  # 转换为kHz单位

# 筛选有效角度
valid_mask = (angles >= -50) & (angles <= 180)
valid_angles = angles[valid_mask]
valid_indices = indices[valid_mask]

# 按角度降序排序（最大角度排在最上面）
sort_order = np.argsort(valid_angles)[::1]  # 获取降序索引
sorted_angles = valid_angles[sort_order]      # 应用排序
sorted_indices = valid_indices[sort_order]    # 保持对应的索引

# 创建热图矩阵（使用排序后的数据）
hrtf_matrix = pred_log_hrtf[sorted_indices]    # 形状为 (n_angles, n_frequencies)

# 创建热图（注意Y轴范围使用排序后的角度值）
im = plt.imshow(hrtf_matrix,
                extent=[freq_list_kHz.min(), freq_list_kHz.max(), 
                        sorted_angles.min(), sorted_angles.max()],
                aspect='auto', 
                origin='lower',  # 原点在左下角
                cmap='viridis')

# 设置坐标轴标签（单独设置字号）
plt.xlabel('Frequency (kHz)', fontsize=18, fontfamily='Times New Roman')
plt.ylabel('Angle φ (degrees)', fontsize=18, fontfamily='Times New Roman')

# 添加颜色条并设置标签
cbar = plt.colorbar(im)
cbar.set_label('dB', fontsize=16, fontfamily='Times New Roman')
# 设置颜色条刻度的字号
cbar.ax.tick_params(labelsize=16)

# 设置坐标轴范围
plt.xlim(freq_list_kHz.min(), freq_list_kHz.max())
plt.ylim(valid_angles.min(), valid_angles.max())

# 设置刻度标签字号
plt.xticks(fontsize=16, fontfamily='Times New Roman')
plt.yticks(fontsize=16, fontfamily='Times New Roman')

# # 添加标题
# plt.title('HRTF Magnitude (dB) vs Frequency and Angle', 
#           fontsize=16, fontfamily='Times New Roman')

plt.tight_layout()

# 关键修改：先保存再显示！！！
plt.savefig("HRTF_contrast_pred.pdf", bbox_inches='tight', dpi=300)
plt.show()

