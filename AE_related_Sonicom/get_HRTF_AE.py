import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from TestNet import TestNet as threeDResnetANP
from TestNet import ResNet3DClassifier as threeDResnet
from TestNet import ResNet2DClassifier as twoDResnet
from new_dataset import SonicomDataSet, SingleSubjectDataSet
from utils import split_dataset
import numpy as np
import matplotlib.pyplot as plt
from AE import HRTF_VQVAE
from AEconfig import pos_dim_for_each_row, \
    num_hrtf_rows, width_per_hrtf_row, transformer_encoder_settings, decoder_mlp_layers, encoder_out_vec_num, \
    num_codebook_embeddings, commitment_cost_beta, num_quantizers
from new_cal_LSD import evaluate_one_hrtf

print(f"当前码本大小为：{num_codebook_embeddings}")
# 设备配置
batch_size = 32
usediff = False  # 是否使用差分数据

current_model = "3DResNet" # ["3DResNetANP", "3DResNet", "2DResNetANP", "2DResNet"]
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
    hrtf_row_width=width_per_hrtf_row,
    hrtf_num_rows=num_hrtf_rows,
    encoder_out_vec_num=encoder_out_vec_num, # 编码器输出序列长度
    encoder_transformer_config=transformer_encoder_settings,
    num_embeddings=num_codebook_embeddings,
    commitment_cost=commitment_cost_beta,
    pos_dim_per_row=pos_dim_for_each_row,
    num_quantizers=num_quantizers
).to(device)
hrtf_encoder.load_state_dict(torch.load(f"HRTFAEweights_So\diff_False_enc_n_1_enc_num_heads-6_num_encoder_layers-4_num_decoder_layers-15_dim_feedforward-512_dropout-0.05_codebook_size_{num_codebook_embeddings}_quan_n_3_120.pth", map_location=device, weights_only=True),strict=False)
print("Load hrtf_encoder")

res_list = []
pred_list = []
true_list = []


hrtf_dir = "FFT_HRTF"

dataset_paths = split_dataset(ear_dir, hrtf_dir, inputform=inputform)
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
hrtfid = 1
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


idx_0_0 = 1957
idx_0_90 = 12
idx_0_80 = 415
path = f'HRTF可视化'
input_type = '2D' if current_model in ['2DResNet', '2DResNetANP'] else '3D'
np.savetxt(f'{path}\\hrtf_AE_0_0_{input_type}_So.txt', pred_log_hrtf[idx_0_0-1,:], fmt='%.3f', header='Magnitude (dB)')
np.savetxt(f'{path}\\hrtf_true_0_0_{input_type}_So.txt', true_log_hrtf[idx_0_0-1,:], fmt='%.3f', header='Magnitude (dB)')

