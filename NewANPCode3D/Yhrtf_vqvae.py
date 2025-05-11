import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import sys
import math
from torch.utils.tensorboard import SummaryWriter
from new_dataset import SonicomDataSet
from utils import split_dataset

weightname = "model-100.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
usediff = True  # 是否使用差值HRTF数据

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
    batch_size=32,
    shuffle=True,
    collate_fn=train_dataset.collate_fn
)


test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=test_dataset.collate_fn
)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1./self._num_embeddings, 1./self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # inputs: (B, C, H, W) or (B, L, C) from Transformer
        # C is embedding_dim

        # 保留原始形状
        input_shape = inputs.shape
        
        # 将输入扁平化: (B, C, H, W) -> (B*H*W, C) or (B, L, C) -> (B*L, C)
        if inputs.dim() == 4: # From CNN-like encoder
            flat_input = inputs.permute(0, 2, 3, 1).contiguous() # (B, H, W, C)
            flat_input = flat_input.view(-1, self._embedding_dim)
        elif inputs.dim() == 3: # From Transformer-like encoder (B, L, C)
            flat_input = inputs.reshape(-1, self._embedding_dim)
        else:
            raise ValueError(f"Input tensor to VectorQuantizer has unsupported dimensions: {inputs.dim()}")

        # 计算与码本向量的距离
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # 编码: 找到最近的码本向量的索引
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 量化: 从码本中获取量化后的向量
        quantized = torch.matmul(encodings, self._embedding.weight).view_as(flat_input) # (B*H*W, C) or (B*L,C)
        
        # 计算损失
        e_latent_loss = F.mse_loss(quantized.detach(), flat_input) # Codebook loss
        q_latent_loss = F.mse_loss(quantized, flat_input.detach()) # Commitment loss
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator (STE)
        quantized_sg = flat_input + (quantized - flat_input).detach()
        
        # 将量化后的向量重塑回原始输入的形状 (除了通道维度可能在最后)
        if inputs.dim() == 4:
            quantized_out = quantized_sg.view(input_shape[0], input_shape[2], input_shape[3], input_shape[1])
            quantized_out = quantized_out.permute(0, 3, 1, 2).contiguous() # (B, C, H, W)
        elif inputs.dim() == 3:
            quantized_out = quantized_sg.view(input_shape) # (B, L, C)
            
        return quantized_out, loss, encoding_indices.view(input_shape[0], -1) # 返回量化输出，VQ损失，和编码索引


# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model) # batch_first=True, so shape (1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # pe shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x shape: (batch_size, seq_len, d_model)
        # self.pe shape: (1, max_len, d_model)
        # We need to slice pe to match seq_len: self.pe[:, :x.size(1)]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# --- Transformer Encoder for HRTF ---
class HrtfTransformerEncoder(nn.Module):
    def __init__(self, hrtf_row_width=108, num_heads=4, num_encoder_layers=3,
                 dim_feedforward=1024, dropout=0.1,
                 hrtf_num_rows=793, 
                 target_seq_len_for_vq=108): # feature_num 在这里可能不再需要
        super().__init__()
        self.d_model = hrtf_row_width
        self.hrtf_num_rows = hrtf_num_rows
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=dropout, max_len=hrtf_num_rows + 10)
        self.target_seq_len_for_vq = target_seq_len_for_vq
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.seq_len_projector = nn.Linear(self.hrtf_num_rows, self.target_seq_len_for_vq)

    def forward(self, hrtf: torch.Tensor) -> torch.Tensor:
        if hrtf.dim() == 4 and hrtf.shape[1] == 1:
            x = hrtf.squeeze(1) 
        elif hrtf.dim() == 3:
            x = hrtf
        else:
            raise ValueError(f"Unsupported HRTF input shape for Transformer: {hrtf.shape}")
        # x = self.pos_encoder(x)
        transformer_output = self.transformer_encoder(x)  # Output shape (batch, hrtf_num_rows, d_model)
        # projected_output = transformer_output.permute(0, 2, 1)  # (B, d_model, hrtf_num_rows)
        # projected_output = self.seq_len_projector(projected_output)  # (B, d_model, target_seq_len_for_vq)
        # projected_output = projected_output.permute(0, 2, 1)  # (B, target_seq_len_for_vq, d_model)
        # return projected_output # 直接返回序列输出
        # return transformer_output[:, :self.target_seq_len_for_vq, :]  # 直接返回前 target_seq_len_for_vq 行的输出
        return transformer_output

# --- HrtfDecoder (逐行MLP解码器，修改以包含BatchNorm) ---
class HrtfDecoder(nn.Module):
    def __init__(self, latent_feature_dim, pos_dim_per_row=3, hrtf_row_output_width=108, decoder_mlp_hidden_dims=None):
        super().__init__()
        # latent_feature_dim 现在是展平后的 VQ 输出维度 (例如, 108*108 = 11664)
        self.latent_feature_dim = latent_feature_dim
        self.pos_dim_per_row = pos_dim_per_row
        self.hrtf_row_output_width = hrtf_row_output_width
        
        mlp_layers = []
        # 逐行 MLP 的输入维度
        current_input_dim_for_row_mlp = self.latent_feature_dim + self.pos_dim_per_row 
        
        num_hidden_fc_layers = len(decoder_mlp_hidden_dims)

        if num_hidden_fc_layers == 0:
            # 没有隐藏层，直接线性连接
            mlp_layers.append(nn.Linear(current_input_dim_for_row_mlp, self.hrtf_row_output_width))
        else:
            # 第一个隐藏层: Linear -> ReLU
            first_hidden_dim = decoder_mlp_hidden_dims[0]
            mlp_layers.append(nn.Linear(current_input_dim_for_row_mlp, first_hidden_dim))
            mlp_layers.append(nn.ReLU())
            current_input_dim_for_row_mlp = first_hidden_dim

            # 后续的隐藏层 (第二个, 第三个, ...): Linear -> BatchNorm1d -> ReLU
            for i in range(1, num_hidden_fc_layers):
                current_hidden_dim = decoder_mlp_hidden_dims[i]
                mlp_layers.append(nn.Linear(current_input_dim_for_row_mlp, current_hidden_dim))
                mlp_layers.append(nn.BatchNorm1d(current_hidden_dim)) # BatchNorm 在 Linear 之后
                mlp_layers.append(nn.ReLU())
                current_input_dim_for_row_mlp = current_hidden_dim
            
            # MLP 的输出层
            mlp_layers.append(nn.Linear(current_input_dim_for_row_mlp, self.hrtf_row_output_width))
            
        self.row_decoder_mlp = nn.Sequential(*mlp_layers)

    def forward(self, global_flattened_feature, pos_sequence):
        # global_flattened_feature: (batch_size, latent_feature_dim) - 这是展平后的 zq
        # pos_sequence: (batch_size, num_rows_to_reconstruct, pos_dim_per_row) - 例如 (B, 793, 3)
        
        batch_size = global_flattened_feature.shape[0]
        num_rows_to_reconstruct = pos_sequence.shape[1] # 应该是 793
        
        # 将全局展平特征扩展以匹配要重构的行数
        # global_flattened_feature_expanded: (B, num_rows_to_reconstruct, latent_feature_dim)
        feature_expanded = global_flattened_feature.unsqueeze(1).expand(-1, num_rows_to_reconstruct, -1)
        
        # 拼接扩展后的特征和位置信息
        # decoder_input_per_row: (B, num_rows_to_reconstruct, latent_feature_dim + pos_dim_per_row)
        decoder_input_per_row = torch.cat([feature_expanded, pos_sequence], dim=2) 
        
        # 为逐行 MLP 展平: (B * num_rows_to_reconstruct, latent_feature_dim + pos_dim_per_row)
        decoder_input_flat = decoder_input_per_row.reshape(batch_size * num_rows_to_reconstruct, -1)
        
        # 由 MLP 处理 (内部包含 BatchNorm)
        decoded_rows_flat = self.row_decoder_mlp(decoder_input_flat) 
        
        # 重塑回 (B, num_rows_to_reconstruct, hrtf_row_output_width)
        reconstructed_hrtf_rows = decoded_rows_flat.view(batch_size, num_rows_to_reconstruct, self.hrtf_row_output_width)
        
        # 添加通道维度: (B, 1, num_rows_to_reconstruct, hrtf_row_output_width)
        reconstructed_hrtf = reconstructed_hrtf_rows.unsqueeze(1) 
        return reconstructed_hrtf


class HRTF_VQVAE(nn.Module):
    def __init__(self, 
                 hrtf_row_width, # Transformer 的 d_model, 也是 VQ 的 embedding_dim
                 hrtf_num_rows,  # Transformer 输入的原始序列长度
                 target_seq_len_for_vq, # VQ 输入的目标序列长度 (例如 108)
                 encoder_transformer_config,
                 num_embeddings, 
                 commitment_cost,
                 pos_dim_per_row,
                 decoder_mlp_hidden_dims
                 ): 
        super().__init__()
        
        self.d_model = hrtf_row_width
        self.target_seq_len_for_vq = target_seq_len_for_vq # 例如 108

        self.encoder = HrtfTransformerEncoder(
            hrtf_row_width=self.d_model, 
            hrtf_num_rows=hrtf_num_rows, # 原始输入序列长度 (例如 793)
            target_seq_len_for_vq=self.target_seq_len_for_vq, # 编码器输出此长度的序列 (例如 108)
            num_heads=encoder_transformer_config.get("num_heads", 4),
            num_encoder_layers=encoder_transformer_config.get("num_encoder_layers", 3),
            dim_feedforward=encoder_transformer_config.get("dim_feedforward", 512),
            dropout=encoder_transformer_config.get("dropout", 0.1)
        )
        self.model_name = "HRTF_VQVAE_FlattenedVQ_Decoder"

        self.vq_layer = VectorQuantizer(num_embeddings, self.d_model, commitment_cost)

        self.projector = nn.Linear(self.d_model, 1)

        # 解码器的 latent_feature_dim 是展平后的 VQ 输出维度
        # decoder_latent_dim = self.target_seq_len_for_vq * self.d_model # 例如 108 * 108
        decoder_latent_dim = 793
        self.decoder = HrtfDecoder(
            latent_feature_dim=decoder_latent_dim, 
            pos_dim_per_row=pos_dim_per_row,
            hrtf_row_output_width=self.d_model, # 假设输出的每行 HRTF 宽度仍为 d_model
            decoder_mlp_hidden_dims=decoder_mlp_hidden_dims
        )

    def forward(self, hrtf_data, pos_data):
        # hrtf_data: (B, 1, hrtf_num_rows, hrtf_row_width)
        # pos_data: (B, hrtf_num_rows, pos_dim_per_row)

        # ze: (B, target_seq_len_for_vq, d_model) 例如 (B, 108, 108)
        ze = self.encoder(hrtf_data) 
        
        # zq: (B, target_seq_len_for_vq, d_model), vq_loss, indices 例如 (B, 108, 108)
        zq, vq_loss, _ = self.vq_layer(ze) 
        
        # 将 zq 展平
        # zq_flat: (B, target_seq_len_for_vq * d_model) 例如 (B, 108*108)
        # batch_size = zq.shape[0]
        # zq_flat = zq.reshape(batch_size, -1)
        zq = self.projector(zq) # (B, 1, target_seq_len_for_vq)
        zq_flat = zq.reshape(zq.shape[0], -1) # (B, target_seq_len_for_vq * 1)
        
        # 将展平后的 zq 和 pos_data 传递给解码器
        # reconstructed_hrtf: (B, 1, hrtf_num_rows, hrtf_row_width)
        reconstructed_hrtf = self.decoder(zq_flat, pos_data)
        
        return reconstructed_hrtf, vq_loss

# --- 模型实例化和优化器 ---
pos_dim_for_each_row = 3
hrtf_num_rows_original = 793  # Transformer 输入的原始 HRTF 行数
hrtf_d_model = 108           # 每行 HRTF 的维度, 也是 Transformer 的 d_model 和 VQ 的 embedding_dim

# VQ 输入的目标序列长度
target_num_quantized_vectors = 10 # Encoder 输出序列的长度，VQ处理这个长度的序列

# VQ-VAE 特定参数
num_codebook_embeddings = 108 
commitment_cost_beta = 0.25

transformer_encoder_settings = {
    "num_heads": 9, 
    "num_encoder_layers": 18, 
    "dim_feedforward": 512,
    "dropout": 0.02
}
# 解码器 MLP 的隐藏层维度示例
# 输入到 MLP 的维度将是 (108*108 + 3)
decoder_mlp_layers = [512, 512, 256, 256, 128, 128] # 示例，可以根据需要调整

model = HRTF_VQVAE(
    hrtf_row_width=hrtf_d_model,
    hrtf_num_rows=hrtf_num_rows_original,
    target_seq_len_for_vq=target_num_quantized_vectors, # 编码器输出序列长度
    encoder_transformer_config=transformer_encoder_settings,
    num_embeddings=num_codebook_embeddings,
    commitment_cost=commitment_cost_beta,
    pos_dim_per_row=pos_dim_for_each_row,
    decoder_mlp_hidden_dims=decoder_mlp_layers
).to(device)

print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5) # VQVAE可能需要不同的学习率
reconstruction_loss_fn = nn.MSELoss()
num_epochs = 1000
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6) # eta_min可以设小一点

# --- 训练循环 ---
for epoch in range(num_epochs):
    model.train()
    epoch_loss_recon = 0
    epoch_loss_vq = 0
    
    train_progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", file=sys.stdout)
    
    for i, batch in enumerate(train_progress_bar):
        hrtf = batch["hrtf"].to(device) # 假设形状是 (batch, 793, 108)
        if hrtf.dim() == 3:
            hrtf = hrtf.unsqueeze(1) # -> (batch, 1, 793, 108)
        pos = batch["position"].to(device)   # (batch, 793, 3)

        optimizer.zero_grad()
        
        reconstructed_hrtf, vq_loss = model(hrtf, pos)
        
        recon_loss = reconstruction_loss_fn(reconstructed_hrtf, hrtf)
        total_loss = recon_loss + vq_loss # vq_loss 内部已包含 commitment_cost * e_latent_loss
        
        total_loss.backward()
        optimizer.step()
        
        epoch_loss_recon += recon_loss.item()
        epoch_loss_vq += vq_loss.item()
        
        train_progress_bar.desc = (f"[train epoch {epoch+1}] loss_recon: {epoch_loss_recon/(i+1):.3f} "
                                   f"loss_vq: {epoch_loss_vq/(i+1):.3f} "
                                   f"lr: {optimizer.param_groups[0]['lr']:.3e}")

    avg_recon_loss_train = epoch_loss_recon / len(train_loader)
    avg_vq_loss_train = epoch_loss_vq / len(train_loader)
    # print(f"Epoch {epoch+1} Train: Recon Loss: {avg_recon_loss_train:.4f}, VQ Loss: {avg_vq_loss_train:.4f}")

    # --- 验证循环 (可选) ---
    model.eval()
    val_loss_recon = 0
    val_loss_vq = 0
    with torch.no_grad():
        val_progress_bar = tqdm.tqdm(test_loader, file=sys.stdout)
        for step, batch in enumerate(val_progress_bar):
            hrtf_val = batch["hrtf"].to(device)
            if hrtf_val.dim() == 3:
                hrtf_val = hrtf_val.unsqueeze(1)
            pos_val = batch["position"].to(device)
            
            reconstructed_hrtf_val, vq_loss_val = model(hrtf_val, pos_val)
            recon_loss_val = reconstruction_loss_fn(reconstructed_hrtf_val, hrtf_val)
            
            val_loss_recon += recon_loss_val.item()
            val_loss_vq += vq_loss_val.item()
            val_progress_bar.desc = (f"[valid epoch {epoch+1}] loss_recon: {val_loss_recon/(step+1):.3f} "
                                     f"loss_vq: {val_loss_vq/(step+1):.3f}")
    
    avg_recon_loss_val = val_loss_recon / len(test_loader)
    avg_vq_loss_val = val_loss_vq / len(test_loader)
    # print(f"Epoch {epoch+1} Valid: Recon Loss: {avg_recon_loss_val:.4f}, VQ Loss: {avg_vq_loss_val:.4f}")
    
    scheduler.step()
    # 保存模型
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"{weightdir}/model-vqvae-{epoch+1}-usediff.pth")
        print(f"Model saved at epoch {epoch+1}")

print("Training finished.")