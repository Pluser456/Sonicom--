import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import sys
from new_dataset import SonicomDataSet
from utils import split_dataset
weightname = "model-0.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    positions_chosen_num=positions_chosen_num,
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
    batch_size=24,
    shuffle=True,
    collate_fn=train_dataset.collate_fn
)


test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=test_dataset.collate_fn
)

# --- AE 模型定义 ---

class HrtfEncoderFC(nn.Module): # FC for Fully Connected
    def __init__(self, latent_feature_dim=128, hrtf_num_rows=793, hrtf_row_width=108, encoder_fc_hidden_dims=None):
        """
        基于全连接层的HRTF编码器。
        参数:
            latent_feature_dim: 编码器输出的潜在特征维度。
            hrtf_num_rows: HRTF的行数 (例如，793)。
            hrtf_row_width: HRTF每行的宽度 (例如，108)。
            encoder_fc_hidden_dims: 全连接编码器的隐藏层维度列表。
        """
        super().__init__()
        self.latent_feature_dim = latent_feature_dim
        self.input_dim = hrtf_num_rows * hrtf_row_width # HRTF扁平化后的总维度

        if encoder_fc_hidden_dims is None:
            # 示例：逐渐减小维度。你需要根据实际情况调整这些值。
            # 考虑到输入维度可能非常大，这里的隐藏层维度也需要相应调整。
            encoder_fc_hidden_dims = [self.input_dim // 8, self.input_dim // 16, self.input_dim // 64]
            # 确保隐藏层维度不会太小，至少比 latent_feature_dim 大
            encoder_fc_hidden_dims = [max(dim, latent_feature_dim * 2) for dim in encoder_fc_hidden_dims if dim > latent_feature_dim]
            if not encoder_fc_hidden_dims: # 如果所有计算出的隐藏层都太小
                 encoder_fc_hidden_dims = [max(self.input_dim // 10, latent_feature_dim * 4)]


        fc_layers = []
        current_dim = self.input_dim
        for hidden_dim in encoder_fc_hidden_dims:
            fc_layers.append(nn.Linear(current_dim, hidden_dim))
            fc_layers.append(nn.ReLU()) # 可以考虑 BatchNorm1d
            # fc_layers.append(nn.BatchNorm1d(hidden_dim)) # 可选
            current_dim = hidden_dim
        
        fc_layers.append(nn.Linear(current_dim, self.latent_feature_dim))
        # 编码器的最后一层通常不加激活函数，或者根据需要加，例如 tanh
        
        self.encoder_fc = nn.Sequential(*fc_layers)
        self.flatten = nn.Flatten(start_dim=1) # 从通道维度之后开始扁平化

    def forward(self, hrtf):
        """
        前向传播。
        参数:
            hrtf: 输入的HRTF数据，形状 (batch_size, 1, hrtf_num_rows, hrtf_row_width)。
                  例如, (batch_size, 1, 793, 108)。
        返回:
            feature: 编码器输出的全局HRTF特征，形状 (batch_size, latent_feature_dim)。
        """
        # 扁平化输入 HRTF: (batch_size, 1, num_rows, row_width) -> (batch_size, num_rows * row_width)
        # 注意：如果输入已经是 (batch_size, num_rows, row_width)，则 flatten(start_dim=1)
        # 如果输入是 (batch_size, 1, num_rows, row_width)，则需要先 squeeze(1) 或 view
        if hrtf.dim() == 4 and hrtf.shape[1] == 1: # (B, 1, H, W)
            hrtf_flat = hrtf.view(hrtf.shape[0], -1) # -> (B, H*W)
        elif hrtf.dim() == 3: # (B, H, W)
            hrtf_flat = self.flatten(hrtf) # -> (B, H*W)
        else:
            raise ValueError(f"Unsupported HRTF input shape: {hrtf.shape}")

        feature = self.encoder_fc(hrtf_flat)
        return feature

class HrtfDecoder(nn.Module):
    def __init__(self, latent_feature_dim=128, pos_dim_per_row=3, hrtf_row_output_width=108, decoder_mlp_hidden_dims=None):
        """
        解码器，逐行重构HRTF。
        参数:
            latent_feature_dim: 全局HRTF特征的维度。
            pos_dim_per_row: HRTF每一行对应位置/上下文特征的维度 (例如，3D坐标则为3)。
            hrtf_row_output_width: HRTF每一行需要重构的宽度 (频率点数，例如108)。
            decoder_mlp_hidden_dims: 用于解码每一行的MLP的隐藏层维度列表。
        """
        super().__init__()
        self.latent_feature_dim = latent_feature_dim
        self.pos_dim_per_row = pos_dim_per_row
        self.hrtf_row_output_width = hrtf_row_output_width

        if decoder_mlp_hidden_dims is None:
            decoder_mlp_hidden_dims = [256, 256]  # 示例MLP隐藏层大小

        mlp_layers = []
        current_dim = self.latent_feature_dim + self.pos_dim_per_row
        for hidden_dim in decoder_mlp_hidden_dims:
            mlp_layers.append(nn.Linear(current_dim, hidden_dim))
            mlp_layers.append(nn.ReLU()) # 可以考虑使用其他激活函数或添加归一化层
            current_dim = hidden_dim
        mlp_layers.append(nn.Linear(current_dim, self.hrtf_row_output_width))
        
        self.row_decoder_mlp = nn.Sequential(*mlp_layers)

    def forward(self, feature, pos):
        """
        前向传播。
        参数:
            feature: 编码器输出的全局HRTF特征，形状 (batch_size, latent_feature_dim)。
            pos: HRTF每一行对应的位置/上下文信息，形状 (batch_size, num_rows, pos_dim_per_row)。
                 例如, (batch_size, 793, 3)。
        返回:
            reconstructed_hrtf: 重构的HRTF，形状 (batch_size, 1, num_rows, hrtf_row_output_width)。
                                例如, (batch_size, 1, 793, 108)。
        """
        batch_size = feature.shape[0]
        num_rows = pos.shape[1] # HRTF的行数，例如793

        # 将全局特征扩展，使其可以与每一行的位置信息拼接
        # feature_expanded 形状: (batch_size, num_rows, latent_feature_dim)
        feature_expanded = feature.unsqueeze(1).expand(-1, num_rows, -1)
        
        # 拼接扩展后的全局特征和每行的位置信息
        # decoder_input_per_row 形状: (batch_size, num_rows, latent_feature_dim + pos_dim_per_row)
        decoder_input_per_row = torch.cat([feature_expanded, pos], dim=2)
        
        # 为了MLP处理，将输入扁平化：MLP将独立处理每一行的特征向量
        # decoder_input_flat 形状: (batch_size * num_rows, latent_feature_dim + pos_dim_per_row)
        decoder_input_flat = decoder_input_per_row.view(batch_size * num_rows, -1)
        
        # 应用MLP解码每一行
        # decoded_rows_flat 形状: (batch_size * num_rows, hrtf_row_output_width)
        decoded_rows_flat = self.row_decoder_mlp(decoder_input_flat)
        
        # 将扁平化的行重构回 (batch_size, num_rows, hrtf_row_output_width)
        # 例如, (batch_size, 793, 108)
        reconstructed_hrtf_rows = decoded_rows_flat.view(batch_size, num_rows, self.hrtf_row_output_width)
        
        # 添加通道维度以匹配目标HRTF形状 (batch_size, 1, num_rows, hrtf_row_output_width)
        # 例如, (batch_size, 1, 793, 108)
        reconstructed_hrtf = reconstructed_hrtf_rows.unsqueeze(1)
            
        return reconstructed_hrtf

class HRTFAutoencoder(nn.Module):
    def __init__(self, latent_feature_dim=128, pos_dim_per_row=3, hrtf_num_rows=793, hrtf_row_width=108, 
                 decoder_mlp_hidden_dims=None, 
                 encoder_fc_hidden_dims=None): 
        super().__init__()
        
        self.hrtf_num_rows = hrtf_num_rows
        self.hrtf_row_width = hrtf_row_width

        
        self.encoder = HrtfEncoderFC(
            latent_feature_dim=latent_feature_dim,
            hrtf_num_rows=self.hrtf_num_rows,
            hrtf_row_width=self.hrtf_row_width,
            encoder_fc_hidden_dims=encoder_fc_hidden_dims
        )
        self.model_name = "HRTF_AE_FCEncoder_RowWiseDecoder"
        self.decoder = HrtfDecoder(
            latent_feature_dim=latent_feature_dim,
            pos_dim_per_row=pos_dim_per_row,
            hrtf_row_output_width=self.hrtf_row_width,
            decoder_mlp_hidden_dims=decoder_mlp_hidden_dims
        )
        
        self.latent_feature_dim = latent_feature_dim
        self.pos_dim_per_row = pos_dim_per_row


    def forward(self, hrtf_data, pos_data):
        feature = self.encoder(hrtf_data) 
        reconstructed_hrtf = self.decoder(feature, pos_data)
        return reconstructed_hrtf, feature


# --- 模型实例化和优化器 ---
latent_dim = 256 # 您可以调整这个潜在特征的维度
position_dim = 3   # 假设位置是三维的 (azimuth, elevation, distance) 或其他
model = HRTFAutoencoder(latent_feature_dim=latent_dim, pos_dim_per_row=position_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5) # 学习率可能需要调整
loss_function = nn.MSELoss()

# --- 训练循环 ---
num_epochs = 50 # 示例 epoch 数
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
        optimizer.step()
        
        epoch_loss += loss.item()
        # 更新tqdm的后缀信息
        train_progress_bar.set_postfix(loss=loss.item())

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")

    # --- 验证循环 (可选) ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        val_progress_bar = tqdm.tqdm(test_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", file=sys.stdout)
        for batch in val_progress_bar:
            hrtf = batch["hrtf"].to(device).unsqueeze(1)
            pos = batch["position"].to(device)
            
            reconstructed_hrtf, _ = model(hrtf, pos)
            loss = loss_function(reconstructed_hrtf, hrtf)
            val_loss += loss.item()
            val_progress_bar.set_postfix(loss=loss.item())
            
    avg_val_loss = val_loss / len(test_loader)
    print(f"Validation Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_val_loss:.4f}")

    # 保存模型
    if (epoch + 1) % 10 == 0: # 每10个epoch保存一次
        torch.save(model.state_dict(), f"{weightdir}/hrtf_ae_model_epoch_{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}")

print("Training finished.")