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
from new_dataset import SonicomDataSet
from utils import split_dataset
from transformers import get_cosine_schedule_with_warmup

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
test_dataset = SonicomDataSet(
    dataset_paths["test_hrtf_list"],
    dataset_paths["left_test"],
    dataset_paths["right_test"],
    calc_mean=False,
    status="test",
    inputform=inputform,
    mode="left",
    use_diff=usediff,
    provided_mean_left=train_dataset.log_mean_hrtf_left,
    provided_mean_right=train_dataset.log_mean_hrtf_right
)

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=12,
    shuffle=True,
    collate_fn=train_dataset.collate_fn
)


test_loader = DataLoader(
    test_dataset,
    batch_size=24,
    shuffle=False,
    collate_fn=test_dataset.collate_fn
)

# --- 模型实例化和优化器 ---
from AEconfig import latent_dim, pos_dim_for_each_row, num_hrtf_rows, width_per_hrtf_row, transformer_encoder_settings, decoder_mlp_layers


model = HRTFAutoencoder(
    latent_feature_dim=latent_dim,
    pos_dim_per_row=pos_dim_for_each_row,
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
writer = SummaryWriter(log_dir=f"{log_dir}/diff_{str(usediff)}_lat_d_{str(latent_dim)}_enc_{str(transformer_settings_str)}_dec_{str(decoder_mlp_layers)}") # <--- TensorBoard 日志目录
# --- 训练循环 ---

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    # 使用tqdm包装train_loader
    train_progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", file=sys.stdout)
    
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
        
        epoch_loss += loss.item()
        train_progress_bar.desc = "[train epoch {}] loss: {:.3f} lr: {:.3e}".format(epoch + 1, epoch_loss / (i + 1), optimizer.param_groups[0]['lr'])

    writer.add_scalar('train_loss', epoch_loss / len(train_loader), epoch) # 记录训练损失
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch) # 记录学习率
    # --- 验证循环 (可选) ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        val_progress_bar = tqdm.tqdm(test_loader, file=sys.stdout)
        for step, batch in enumerate(val_progress_bar):
            hrtf = batch["hrtf"].to(device).unsqueeze(1)
            pos = batch["position"].to(device)
            
            reconstructed_hrtf, _ = model(hrtf, pos)
            loss = loss_function(reconstructed_hrtf, hrtf)
            val_loss += loss.item()
            # val_progress_bar.set_postfix(loss=loss.item())
            val_progress_bar.desc = "[valid epoch {}] loss: {:.3f}".format(epoch + 1, val_loss / (step + 1))
            
    writer.add_scalar('val_loss', val_loss / len(test_loader), epoch) # 记录验证损失
    scheduler.step()  # 更新学习率
    # 保存模型
    if (epoch + 1) % 100 == 0: # 每100个epoch保存一次
        torch.save(model.state_dict(), f"{weightdir}/model-{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}")

print("Training finished.")
writer.close() # <--- 关闭 writer