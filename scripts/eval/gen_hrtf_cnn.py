import os
import torch
from torch.utils.data import DataLoader
from TestNet import ResNet3D as threeDResnet
from TestNet import ResNet2D as twoDResnet
from new_dataset import SonicomDataSet, SingleSubjectDataSet
from utils import split_dataset
import numpy as np
from new_cal_LSD import evaluate_one_hrtf

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
usediff = False  # 是否使用差分数据
weightname = "best_model_Wi.pth"

current_model = "3DResNet" # ["3DResNetANP", "3DResNet", "2DResNetANP", "2DResNet"]
# 模型和训练配置
if current_model == "3DResNet":
    weightdir = "./CNN3Dweights"
    ear_dir = "Ear_voxel_Wi"
    isANP = False
    model = threeDResnet().to(device)
    inputform = "voxel"
    model_path = f"{weightdir}/{weightname}"
elif current_model == "2DResNet":
    weightdir = "./CNNweights"
    ear_dir = "Ear_image_gray_Wi"
    isANP = False
    model_path = f"{weightdir}/{weightname}"
    positions_chosen_num = 793
    model = twoDResnet().to(device)
    inputform = "image"
    model_path = f"{weightdir}/{weightname}"

if os.path.exists(weightdir) is False:
    os.makedirs(weightdir)
modelpath = f"{weightdir}/{weightname}"
# positions_chosen_num = 793

model.load_state_dict(torch.load(modelpath, map_location=device, weights_only=True))
print("Load model from", modelpath)


res_list = []
pred_list = []
true_list = []


hrtf_dir = "FFT_HRTF_Wi"

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
pred_log_hrtf, true_log_hrtf = evaluate_one_hrtf(model, dataloader, usediff)
pred_log_hrtf = pred_log_hrtf.squeeze(0)
true_log_hrtf = true_log_hrtf.squeeze(0)

# 确保 pred_log_hrtf 和 true_log_hrtf 是 CPU tensor
pred_log_hrtf = pred_log_hrtf.cpu().numpy()  # 转换为 NumPy 数组
true_log_hrtf = true_log_hrtf.cpu().numpy()  # 转换为 NumPy 数组

idx_dict = {"0_0": 1957, "0_90": 12, "90_0": 305, "20_54": 501}
path = f'HRTF可视化'
input_type = '2D' if current_model in ['2DResNet', '2DResNetANP'] else '3D'
for idx_key, idx in idx_dict.items():
    np.savetxt(f'{path}\\hrtf_CNN_{idx_key}_{input_type}_Wi.txt', pred_log_hrtf[idx-1,:], fmt='%.3f', header='Magnitude (dB)')
    # np.savetxt(f'{path}\\hrtf_true_{idx_key}_Wi.txt', true_log_hrtf[idx-1,:], fmt='%.3f', header='Magnitude (dB)')

