import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # <--- 添加导入
import tqdm
import sys
from AE import HRTFAutoencoder
from new_dataset import OnlyHRTFDataSet
from utils import split_dataset
from transformers import get_cosine_schedule_with_warmup
from matplotlib import pyplot as plt
import numpy as np
import time

weightname = ".pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_dir = "./runs/HRTF_VQVAE_TransformerDecoder_compact" # <--- TensorBoard 日志目录
usediff = False  # 是否使用差值HRTF数据

weightdir = "./HRTFAEweights"
ear_dir = "Ear_image_gray"
if os.path.exists(weightdir) is False:
    os.makedirs(weightdir)
modelpath = f"{weightdir}/{weightname}"
positions_chosen_num = 793
inputform = "image"

dataset_paths = split_dataset(ear_dir, "FFT_HRTF",inputform=inputform)

train_dataset = OnlyHRTFDataSet(
    dataset_paths["train_hrtf_list"],
    use_diff=usediff,
    calc_mean=True,
    status="test", # 因为这里希望坐标是按顺序输入的
    mode="left"
)
test_dataset = OnlyHRTFDataSet(
    dataset_paths["test_hrtf_list"],
    calc_mean=False,
    status="test",
    mode="left",
    use_diff=usediff,
    provided_mean_left=train_dataset.log_mean_hrtf_left,
    provided_mean_right=train_dataset.log_mean_hrtf_right
)

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=6,
    shuffle=True,
    collate_fn=train_dataset.collate_fn
)


test_loader = DataLoader(
    test_dataset,
    batch_size=12,
    shuffle=False,
    collate_fn=test_dataset.collate_fn
)

# --- 模型实例化和优化器 ---
from AEconfig import pos_dim_for_each_row,\
      num_hrtf_rows, width_per_hrtf_row, transformer_encoder_settings, decoder_mlp_layers, encoder_out_vec_num


model = HRTFAutoencoder(
    pos_dim_per_row=pos_dim_for_each_row,
    encoder_out_vec_num=encoder_out_vec_num,
    hrtf_num_rows=num_hrtf_rows,
    hrtf_row_width=width_per_hrtf_row,
    decoder_mlp_hidden_dims=decoder_mlp_layers,
    encoder_transformer_config=transformer_encoder_settings
).to(device)

if os.path.exists(modelpath):
    print("Load model from", modelpath)
    model.load_state_dict(torch.load(modelpath, map_location=device), strict=False)

print(f"Model Name: {model.model_name}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
# print(model) # 取消注释以查看详细模型结构

optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
# optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5) # VQVAE可能需要不同的学习率
loss_function = nn.MSELoss()
num_epochs = 1000 # 示例 epoch 数
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.85, min_lr=1e-6, patience=10)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=num_epochs)
transformer_settings_str = "_".join([f"{key}-{value}" for key, value in transformer_encoder_settings.items()])
# writer = SummaryWriter(log_dir=f"{log_dir}/diff_{str(usediff)}_enc_n_{str(encoder_out_vec_num)}_enc_{str(transformer_settings_str)}_dec_{str(decoder_mlp_layers)}_{time.strftime('%m%d-%H%M')}") # <--- TensorBoard 日志目录
writer = SummaryWriter(log_dir=f"{log_dir}/diff_{str(usediff)}_enc_n_{str(encoder_out_vec_num)}_enc_{str(transformer_settings_str)}_dec_transformer_{time.strftime('%m%d-%H%M')}") # <--- TensorBoard 日志目录
# --- 训练循环 ---

def visualize_hrtf(model, test_loader, device, save_path, max_samples=16):
    model.eval()
    hrtf_true_list = []
    hrtf_pred_list = []
    with torch.no_grad():
        count = 0
        for batch in test_loader:
            pos = batch["position"].to(device)
            hrtf_true = batch["hrtf"].to(device).unsqueeze(1)
            hrtf_pred, _ = model(hrtf_true,pos)
            # 去除多余维度
            hrtf_true = hrtf_true.squeeze()
            hrtf_pred = hrtf_pred.squeeze()
            # 若数据是三维（形如 (batch, num_rows, features)），取第一个样本第一行的数据
            if hrtf_true.ndim == 3:
                sample_true = hrtf_true[0, 0, :].cpu().numpy()
                sample_pred = hrtf_pred[0, 0, :].cpu().numpy()
            # 若数据是二维（形如 (num_rows, features)），同样取第一行
            elif hrtf_true.ndim == 2:
                sample_true = hrtf_true[0, :].cpu().numpy()
                sample_pred = hrtf_pred[0, :].cpu().numpy()
            else:
                continue
            hrtf_true_list.append(sample_true)
            hrtf_pred_list.append(sample_pred)
            count += 1
            if count >= max_samples:
                break

    if len(hrtf_true_list) == 0:
        print("未获取到有效样本用于可视化")
        return

    # 确定网格行列数（尽量构成正方形）
    n_samples = len(hrtf_true_list)
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    # 将 axs 展平便于遍历
    axs = np.array(axs).reshape(-1)
    for i in range(n_rows * n_cols):
        ax = axs[i]
        if i < n_samples:
            ax.plot(hrtf_true_list[i], label="True HRTF", linewidth=2)
            ax.plot(hrtf_pred_list[i], label="Predicted HRTF", linestyle="--", linewidth=2)
            ax.set_title(f"Sample {i+1}", fontsize=10)
            ax.tick_params(labelsize=8)
            ax.legend(fontsize=8)
        else:
            ax.axis('off')

    fig.suptitle("HRTF 对比网格", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"可视化图已保存至 {save_path}")

best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    # 使用tqdm包装train_loader
    train_progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", file=sys.stdout)
    train_size = 0
    for i, batch in enumerate(train_progress_bar):
        # 从batch中获取数据，确保key与您的collate_fn一致
        hrtf = batch["hrtf"].to(device).unsqueeze(1) # 假设形状是 (batch, 1, 793, 108)
        pos = batch["position"].to(device)   # 假设形状是 (batch, 793, 3)

        optimizer.zero_grad()
        
        # 前向传播
        reconstructed_hrtf, _ = model(hrtf, pos) # AE 通常只关心重构
        
        # 计算损失
        loss = loss_function(reconstructed_hrtf, hrtf)
        
        # 反向传播和优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        train_size += hrtf.shape[0]
        epoch_loss += loss.item() * hrtf.shape[0]
        train_progress_bar.desc = "[train epoch {}] loss: {:.3f} lr: {:.3e}".format(epoch + 1, epoch_loss / train_size, optimizer.param_groups[0]['lr'])

    writer.add_scalar('train_loss', epoch_loss / train_size, epoch) # 记录训练损失
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch) # 记录学习率
    # --- 验证循环 (可选) ---
    model.eval()
    val_loss = 0
    val_size = 0
    with torch.no_grad():
        val_progress_bar = tqdm.tqdm(test_loader, file=sys.stdout)
        for step, batch in enumerate(val_progress_bar):
            hrtf = batch["hrtf"].to(device).unsqueeze(1)
            pos = batch["position"].to(device)
            
            reconstructed_hrtf, _ = model(hrtf, pos)
            loss = loss_function(reconstructed_hrtf, hrtf)
            val_size += hrtf.shape[0]
            val_loss += loss.item() * hrtf.shape[0]
            # val_progress_bar.set_postfix(loss=loss.item())
            val_progress_bar.desc = "[valid epoch {}] loss: {:.3f}".format(epoch + 1, val_loss / val_size)
        if val_loss / val_size < best_val_loss:
            best_val_loss = val_loss / val_size
            visualize_hrtf(model, test_loader, device, f"{weightdir}/hrtf_visualization.png", max_samples=16)
            
    writer.add_scalar('val_loss', val_loss / val_size, epoch) # 记录验证损失
    scheduler.step()  # 更新学习率
    # 保存模型
    if (epoch + 1) % 100 == 0: # 每100个epoch保存一次
        torch.save(model.state_dict(), f"{weightdir}/model-{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}")

print("Training finished.")
writer.close() # <--- 关闭 writer