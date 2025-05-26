import os
import time
from numpy import indices
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import sys
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from new_dataset import OnlyHRTFDataSet
from utils import split_dataset
from AE import HRTF_VQVAE

weightname = "jlj"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_dir = "./runs/HRTF_VQVAE" # <--- TensorBoard 日志目录
usediff = False  # 是否使用差值HRTF数据

weightdir = "./HRTFAEweights"
ear_dir = "Ear_image_gray_Wi"
if os.path.exists(weightdir) is False:
    os.makedirs(weightdir)
modelpath = f"{weightdir}/{weightname}"
inputform = "image"

dataset_paths = split_dataset(ear_dir, "FFT_HRTF_Wi",inputform=inputform)
train_hrtf_list = dataset_paths['train_hrtf_list']
test_hrtf_list = dataset_paths['test_hrtf_list']
left_train = dataset_paths['left_train']
right_train = dataset_paths['right_train']
left_test = dataset_paths['left_test']
right_test = dataset_paths['right_test']


train_dataset = OnlyHRTFDataSet(
    dataset_paths["train_hrtf_list"],
    use_diff=usediff,
    calc_mean=usediff,
    status="test", # 因为这里希望坐标是按顺序输入的
    mode="right"
)
test_dataset = OnlyHRTFDataSet(
    dataset_paths["test_hrtf_list"],
    calc_mean=False,
    status="test",
    mode="right",
    use_diff=usediff,
    provided_mean_left=train_dataset.log_mean_hrtf_left,
    provided_mean_right=train_dataset.log_mean_hrtf_right
)

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=train_dataset.collate_fn
)


test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=test_dataset.collate_fn
)


# --- 模型实例化和优化器 ---


from AEconfig import transformer_encoder_settings, encoder_out_vec_num, \
    pos_dim_for_each_row, num_hrtf_rows, width_per_hrtf_row, num_codebook_embeddings, commitment_cost_beta,num_quantizers

model = HRTF_VQVAE(
    hrtf_row_width=width_per_hrtf_row,
    hrtf_num_rows=num_hrtf_rows,
    encoder_out_vec_num=encoder_out_vec_num, # 编码器输出序列长度
    encoder_transformer_config=transformer_encoder_settings,
    num_embeddings=num_codebook_embeddings,
    commitment_cost=commitment_cost_beta,
    pos_dim_per_row=pos_dim_for_each_row,
    num_quantizers=num_quantizers
).to(device)

if os.path.exists(modelpath):
    model.load_state_dict(torch.load(modelpath, map_location=device, weights_only=True))
    print("Load model from", modelpath)
print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5) # VQVAE可能需要不同的学习率
reconstruction_loss_fn = nn.MSELoss()
num_epochs = 120
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=6, num_training_steps=num_epochs)
transformer_settings_str = "_".join([f"{key}-{value}" for key, value in transformer_encoder_settings.items()])
writer = SummaryWriter(log_dir=f"{log_dir}/diff_{str(usediff)}_enc_n_{str(encoder_out_vec_num)}_enc_{str(transformer_settings_str)}_codebook_size_{str(num_codebook_embeddings)}_quan_n_{str(num_quantizers)}_{time.strftime('%m%d-%H%M')}") # <--- TensorBoard 日志目录
# --- 训练循环 ---
for epoch in range(num_epochs):
    model.train()
    epoch_loss_recon = 0
    epoch_loss_vq = 0
    
    train_progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", file=sys.stdout)
    indices_list = []
    for i, batch in enumerate(train_progress_bar):
        hrtf = batch["hrtf"].to(device) # 假设形状是 (batch, 793, 108)
        if hrtf.dim() == 3:
            hrtf = hrtf.unsqueeze(1) # -> (batch, 1, 793, 108)
        pos = batch["position"].to(device)   # (batch, 793, 3)

        optimizer.zero_grad()
        
        reconstructed_hrtf, vq_loss, indices = model(hrtf, pos)
        indices_list.append(indices)
        recon_loss = reconstruction_loss_fn(reconstructed_hrtf, hrtf)
        total_loss = recon_loss + vq_loss # vq_loss 内部已包含 commitment_cost * e_latent_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        epoch_loss_recon += recon_loss.item()
        epoch_loss_vq += vq_loss.item()
        
        train_progress_bar.desc = (f"[train epoch {epoch+1}] loss_recon: {epoch_loss_recon/(i+1):.2f} "
                                   f"loss_vq: {epoch_loss_vq/(i+1):.2f} "
                                   f"lr: {optimizer.param_groups[0]['lr']:.2e} ")
    indicies = torch.cat(indices_list, dim=1)
    activity = torch.unique(indicies).numel() / num_codebook_embeddings * 100

    avg_recon_loss_train = epoch_loss_recon / len(train_loader)
    avg_vq_loss_train = epoch_loss_vq / len(train_loader)
    writer.add_scalar("train_loss_recon", avg_recon_loss_train, epoch)
    writer.add_scalar("train_loss_vq", avg_vq_loss_train, epoch)
    writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar("activity", activity, epoch)
    print(f"Epoch {epoch+1} Train: Recon Loss: {avg_recon_loss_train:.4f}, VQ Loss: {avg_vq_loss_train:.4f}")

    # --- 验证循环 (可选) ---
    model.eval()
    val_loss_recon = 0
    val_loss_vq = 0
    indices_list = []
    with torch.no_grad():
        val_progress_bar = tqdm.tqdm(test_loader, file=sys.stdout)
        for step, batch in enumerate(val_progress_bar):
            hrtf_val = batch["hrtf"].to(device)
            if hrtf_val.dim() == 3:
                hrtf_val = hrtf_val.unsqueeze(1)
            pos_val = batch["position"].to(device)
            
            reconstructed_hrtf_val, vq_loss_val, indices = model(hrtf_val, pos_val)
            recon_loss_val = reconstruction_loss_fn(reconstructed_hrtf_val, hrtf_val)
            indices_list.append(indices)
            val_loss_recon += recon_loss_val.item()
            val_loss_vq += vq_loss_val.item()
            val_progress_bar.desc = (f"[valid epoch {epoch+1}] loss_recon: {val_loss_recon/(step+1):.3f} "
                                     f"loss_vq: {val_loss_vq/(step+1):.3f}")
    
    avg_recon_loss_val = val_loss_recon / len(test_loader)
    avg_vq_loss_val = val_loss_vq / len(test_loader)
    activity_val = torch.unique(torch.cat(indices_list, dim=1)).numel() / num_codebook_embeddings * 100
    # print(f"Epoch {epoch+1} Valid: Recon Loss: {avg_recon_loss_val:.4f}, VQ Loss: {avg_vq_loss_val:.4f}")
    
    writer.add_scalar("val_loss_recon", avg_recon_loss_val, epoch)
    writer.add_scalar("val_loss_vq", avg_vq_loss_val, epoch)
    writer.add_scalar("val_activity", activity_val, epoch)
    scheduler.step()
    # 保存模型
    if (epoch + 1) % 30 == 0:
        torch.save(model.state_dict(), f"{weightdir}/diff_{str(usediff)}_enc_n_{str(encoder_out_vec_num)}_enc_{str(transformer_settings_str)}_codebook_size_{str(num_codebook_embeddings)}_quan_n_{str(num_quantizers)}_{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}")

print("Training finished.")