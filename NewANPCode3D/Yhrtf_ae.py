import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import sys
import math # 需要导入 math 模块
from new_dataset import SonicomDataSet
from utils import split_dataset

weightname = "model-.pth"
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
    batch_size=2,
    shuffle=True,
    collate_fn=train_dataset.collate_fn
)


test_loader = DataLoader(
    test_dataset,
    batch_size=8,
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
        x = self.pos_encoder(x)
        transformer_output = self.transformer_encoder(x)  # Output shape (batch, hrtf_num_rows, d_model)
        projected_output = transformer_output.permute(0, 2, 1)  # (B, d_model, hrtf_num_rows)
        projected_output = self.seq_len_projector(projected_output)  # (B, d_model, target_seq_len_for_vq)
        projected_output = projected_output.permute(0, 2, 1)  # (B, target_seq_len_for_vq, d_model)
        return projected_output # 直接返回序列输出


# --- HrtfDecoder (逐行MLP解码器，保持不变) ---
class HrtfDecoder(nn.Module):
    def __init__(self, latent_feature_dim=128, pos_dim_per_row=3, hrtf_row_output_width=108, decoder_mlp_hidden_dims=None):
        super().__init__()
        self.latent_feature_dim = latent_feature_dim # This will be d_model (108)
        self.pos_dim_per_row = pos_dim_per_row
        self.hrtf_row_output_width = hrtf_row_output_width
        if decoder_mlp_hidden_dims is None:
            decoder_mlp_hidden_dims = [256, 256] # Example
        
        mlp_layers = []
        # Input to MLP is the expanded global feature + pos info for one row
        current_dim = self.latent_feature_dim + self.pos_dim_per_row 
        for hidden_dim in decoder_mlp_hidden_dims:
            mlp_layers.append(nn.Linear(current_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            current_dim = hidden_dim
        mlp_layers.append(nn.Linear(current_dim, self.hrtf_row_output_width))
        self.row_decoder_mlp = nn.Sequential(*mlp_layers)

    def forward(self, feature, pos):
        # feature: (batch_size, latent_feature_dim) - this is the aggregated quantized feature
        # pos: (batch_size, num_rows_to_reconstruct, pos_dim_per_row) - e.g., (B, 793, 3)
        batch_size = feature.shape[0]
        num_rows_to_reconstruct = pos.shape[1] # Should be 793
        
        # Expand the global feature to match the number of rows to reconstruct
        feature_expanded = feature.unsqueeze(1).expand(-1, num_rows_to_reconstruct, -1) # (B, 793, latent_feature_dim)
        
        decoder_input_per_row = torch.cat([feature_expanded, pos], dim=2) # (B, 793, latent_feature_dim + pos_dim)
        
        decoder_input_flat = decoder_input_per_row.reshape(batch_size * num_rows_to_reconstruct, -1)
        decoded_rows_flat = self.row_decoder_mlp(decoder_input_flat)
        reconstructed_hrtf_rows = decoded_rows_flat.view(batch_size, num_rows_to_reconstruct, self.hrtf_row_output_width)
        reconstructed_hrtf = reconstructed_hrtf_rows.unsqueeze(1) # (B, 1, 793, 108)
        return reconstructed_hrtf


class HRTF_VQVAE(nn.Module):
    def __init__(self, 
                 # Encoder params
                 hrtf_row_width, # d_model for Transformer, also embedding_dim for VQ
                 hrtf_num_rows,  # Original sequence length for Transformer
                 target_seq_len_for_vq, # Target sequence length for VQ input
                 encoder_transformer_config,
                 # VQ params
                 num_embeddings, 
                 commitment_cost,
                 # Decoder params
                 pos_dim_per_row,
                 decoder_mlp_hidden_dims
                 ): 
        super().__init__()
        
        self.d_model = hrtf_row_width # Dimension of each vector in VQ, and Transformer's feature dim
        self.target_seq_len_for_vq = target_seq_len_for_vq

        self.encoder = HrtfTransformerEncoder(
            hrtf_row_width=self.d_model, 
            hrtf_num_rows=hrtf_num_rows,
            target_seq_len_for_vq=self.target_seq_len_for_vq, # Pass this new param
            num_heads=encoder_transformer_config.get("num_heads", 4),
            num_encoder_layers=encoder_transformer_config.get("num_encoder_layers", 3),
            dim_feedforward=encoder_transformer_config.get("dim_feedforward", 512),
            dropout=encoder_transformer_config.get("dropout", 0.1)
        )
        self.model_name = "HRTF_VQVAE_TransformerEncoder_ReducedSeq"

        # Vector Quantizer: embedding_dim is d_model
        self.vq_layer = VectorQuantizer(num_embeddings, self.d_model, commitment_cost)

        # Decoder: its latent_feature_dim is d_model (from aggregated VQ output)
        self.decoder = HrtfDecoder(
            latent_feature_dim=self.d_model, 
            pos_dim_per_row=pos_dim_per_row,
            hrtf_row_output_width=self.d_model, # Assuming output row width is also d_model
            decoder_mlp_hidden_dims=decoder_mlp_hidden_dims
        )

    def forward(self, hrtf_data, pos_data):
        # hrtf_data: (B, 1, hrtf_num_rows, hrtf_row_width)
        # pos_data: (B, hrtf_num_rows, pos_dim_per_row)

        # ze: (B, target_seq_len_for_vq, d_model) e.g., (B, 108, 108)
        ze = self.encoder(hrtf_data) 
        
        # zq: (B, target_seq_len_for_vq, d_model), vq_loss, indices
        zq, vq_loss, _ = self.vq_layer(ze) 
        
        # Aggregate zq to a global feature for the current HrtfDecoder
        # zq_aggregated: (B, d_model) e.g. (B, 108)
        zq_aggregated = zq.mean(dim=1) 
        
        # reconstructed_hrtf: (B, 1, hrtf_num_rows, hrtf_row_width)
        reconstructed_hrtf = self.decoder(zq_aggregated, pos_data)
        
        return reconstructed_hrtf, vq_loss

# --- 模型实例化和优化器 ---
pos_dim_for_each_row = 3
hrtf_num_rows_original = 793  # Original number of HRTF rows (sequence length for Transformer input)
hrtf_d_model = 108           # Dimension of each HRTF row, also Transformer's d_model and VQ embedding_dim

# New: Target sequence length for VQ input
target_num_quantized_vectors = 108 

# VQ-VAE specific
num_codebook_embeddings = 512 
commitment_cost_beta = 0.25

transformer_encoder_settings = {
    "num_heads": 6, # d_model (108) must be divisible by num_heads
    "num_encoder_layers": 4, 
    "dim_feedforward": 512,
    "dropout": 0.1
}
decoder_mlp_layers = [512, 256, 256, 128] # Example decoder MLP structure

model = HRTF_VQVAE(
    hrtf_row_width=hrtf_d_model,
    hrtf_num_rows=hrtf_num_rows_original,
    target_seq_len_for_vq=target_num_quantized_vectors,
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
    print(f"Epoch {epoch+1} Train: Recon Loss: {avg_recon_loss_train:.4f}, VQ Loss: {avg_vq_loss_train:.4f}")

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
    print(f"Epoch {epoch+1} Valid: Recon Loss: {avg_recon_loss_val:.4f}, VQ Loss: {avg_vq_loss_val:.4f}")
    
    scheduler.step()
    # 保存模型
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"{weightdir}/model-vqvae-{epoch+1}-usediff.pth")
        print(f"Model saved at epoch {epoch+1}")

print("Training finished.")