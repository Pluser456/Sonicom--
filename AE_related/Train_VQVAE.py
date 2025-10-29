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

weightname = "nopretrain"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_dir = "AE_related/HRTF_VQVAE" # <--- TensorBoard 日志目录
usediff = False  # 是否使用差值HRTF数据
batch_size = 3

weightdir = log_dir
ear_dir = "Ear_image_gray_Wi"
hrtf_dir = "FFT_HRTF_Wi"
if os.path.exists(weightdir) is False:
    os.makedirs(weightdir)
modelpath = f"{weightdir}/{weightname}"
inputform = "image"

dataset_paths = split_dataset(ear_dir, hrtf_dir, inputform=inputform)
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
    batch_size=batch_size,
    shuffle=True,
    collate_fn=train_dataset.collate_fn
)


test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=test_dataset.collate_fn
)


# --- 模型实例化和优化器 ---
from AEconfig import transformer_encoder_settings, transformer_decoder_settings, encoder_out_vec_num, \
    hrtf_row_len, num_codebook_embeddings, commitment_cost_beta, embed_dim, use_VQ, input_pos_as_seq, \
        tolerance_for_calc_threshold, decay

model = HRTF_VQVAE(
    hrtf_row_len=hrtf_row_len,
    encoder_out_vec_num=encoder_out_vec_num, # 编码器输出序列长度
    embed_dim=embed_dim,
    encoder_transformer_config=transformer_encoder_settings,
    decoder_transformer_config=transformer_decoder_settings,
    num_embeddings=num_codebook_embeddings,
    use_VQ=use_VQ,
    input_pos_as_seq=input_pos_as_seq,
    tolerance_for_calc_threshold=tolerance_for_calc_threshold,
    decay=decay
).to(device)

print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5) # VQVAE可能需要不同的学习率
reconstruction_loss_fn = nn.MSELoss()
num_epochs = 120
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=6, num_training_steps=num_epochs)
if os.path.exists(modelpath):
    model.load_state_dict(torch.load(modelpath, map_location=device, weights_only=True)['model_state_dict'])
    optimizer.load_state_dict(torch.load(modelpath, map_location=device, weights_only=False)['optimizer_state_dict'])
    scheduler.load_state_dict(torch.load(modelpath, map_location=device, weights_only=False)['scheduler_state_dict'])
    start_epoch = torch.load(modelpath, map_location=device, weights_only=False)['epoch'] + 1
    print("Load model from", modelpath)
else:
    start_epoch = 0

transformer_settings_str = "_".join([f"{key}-{value}" for key, value in transformer_encoder_settings.items()])
writer = SummaryWriter(log_dir=f"{log_dir}/test_{time.strftime('%m-%d_%H-%M')}")
# Write model configuration to TensorBoard
config_params = {
    "use_diff": usediff,
    "batch_size": batch_size,
    "width_per_hrtf_row": hrtf_row_len,
    "encoder_out_vec_num": encoder_out_vec_num,
    "embed_dim": embed_dim,
    "num_codebook_embeddings": num_codebook_embeddings,
    "commitment_cost_beta": commitment_cost_beta,
    **{f"encoder_transformer/{k}": v for k, v in transformer_encoder_settings.items()}, # 平铺字典
    **{f"decoder_transformer/{k}": v for k, v in transformer_decoder_settings.items()}, # 平铺字典
    "use_VQ": use_VQ,
    "input_pos_as_seq": input_pos_as_seq,
    "tolerance_for_calc_threshold": tolerance_for_calc_threshold,
    "decay": decay,
}
config_text = "## Model Architecture Configuration\n\n"
config_text += "| Parameter | Value |\n"
config_text += "|:---|:---|\n"
for key, value in config_params.items():
    config_text += f"| {key} | {value} |\n"
writer.add_text("model_config", config_text)
# --- 训练循环 ---
for epoch in range(start_epoch, num_epochs):
    model.train()
    epoch_loss_recon = 0
    epoch_loss_vq = 0
    
    train_progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", file=sys.stdout)
    indexes_list = []
    for i, batch in enumerate(train_progress_bar):
        hrtf = batch["hrtf"].to(device) # 假设形状是 (batch, 793, 108)
        pos = batch["position"].to(device)   # (batch, 793, 3)

        optimizer.zero_grad()
        
        reconstructed_hrtf, vq_loss, indices = model(hrtf, pos)
        indexes_list.append(indices)
        recon_loss = reconstruction_loss_fn(reconstructed_hrtf, hrtf)
        total_loss = recon_loss + vq_loss * commitment_cost_beta # vq_loss 内部已包含 commitment_cost * e_latent_loss
        
        total_loss.backward()
        optimizer.step()
        
        epoch_loss_recon += recon_loss.item()
        epoch_loss_vq += vq_loss.item()
        
        train_progress_bar.desc = (f"[train epoch {epoch+1}] loss_recon: {epoch_loss_recon/(i+1):.2f} "
                                   f"loss_vq: {epoch_loss_vq/(i+1):.2f} "
                                   f"lr: {optimizer.param_groups[0]['lr']:.2e} ")
    indexes = torch.cat(indexes_list, dim=0)
    activity = torch.unique(indexes).numel() / num_codebook_embeddings * 100

    avg_recon_loss_train = epoch_loss_recon / len(train_loader)
    avg_vq_loss_train = epoch_loss_vq / len(train_loader)
    writer.add_scalar("train_loss_recon", avg_recon_loss_train, epoch)
    writer.add_scalar("train_loss_vq", avg_vq_loss_train, epoch)
    writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar("activity", activity, epoch)
    # --- 验证循环 ---
    model.eval()
    val_loss_recon = 0
    val_loss_vq = 0
    indexes_list = []
    with torch.no_grad():
        val_progress_bar = tqdm.tqdm(test_loader, file=sys.stdout)
        for step, batch in enumerate(val_progress_bar):
            hrtf_val = batch["hrtf"].to(device)
            pos_val = batch["position"].to(device)
            
            reconstructed_hrtf_val, vq_loss_val, indices = model(hrtf_val, pos_val)
            recon_loss_val = reconstruction_loss_fn(reconstructed_hrtf_val, hrtf_val)
            indexes_list.append(indices)
            val_loss_recon += recon_loss_val.item()
            val_loss_vq += vq_loss_val.item()
            val_progress_bar.desc = (f"[valid epoch {epoch+1}] loss_recon: {val_loss_recon/(step+1):.3f} "
                                     f"loss_vq: {val_loss_vq/(step+1):.3f}")
    
    avg_recon_loss_val = val_loss_recon / len(test_loader)
    avg_vq_loss_val = val_loss_vq / len(test_loader)
    activity_val = torch.unique(torch.cat(indexes_list, dim=1)).numel() / num_codebook_embeddings * 100
    # print(f"Epoch {epoch+1} Valid: Recon Loss: {avg_recon_loss_val:.4f}, VQ Loss: {avg_vq_loss_val:.4f}")
    
    writer.add_scalar("val_loss_recon", avg_recon_loss_val, epoch)
    writer.add_scalar("val_loss_vq", avg_vq_loss_val, epoch)
    writer.add_scalar("val_activity", activity_val, epoch)
    scheduler.step()
    print("\n")

    # 保存模型
    if (epoch + 1) % 30 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, f"{weightdir}/savetime_{time.strftime('%m-%d_%H-%M')}.pt")
        print(f"Checkpoint saved at epoch {epoch+1}")

print("Training finished.")