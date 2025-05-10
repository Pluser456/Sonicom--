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
    batch_size=32,
    shuffle=True,
    collate_fn=train_dataset.collate_fn
)


test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=test_dataset.collate_fn
)

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
                 dim_feedforward=1024, latent_feature_dim=256, dropout=0.1, 
                 hrtf_num_rows=793): # hrtf_num_rows for PositionalEncoding max_len
        super().__init__()
        self.d_model = hrtf_row_width  # Feature dimension of each HRTF row

        # Optional: Input projection if hrtf_row_width is not the desired d_model
        # self.input_projection = nn.Linear(hrtf_row_width, self.d_model)
        # For simplicity, we assume hrtf_row_width is d_model

        self.pos_encoder = PositionalEncoding(self.d_model, dropout, max_len=hrtf_num_rows + 10) # +10 for safety margin
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Crucial: input format (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Linear layer to map the aggregated output of Transformer to latent_feature_dim
        self.output_fc = nn.Linear(self.d_model*2, latent_feature_dim)

    def forward(self, hrtf: torch.Tensor) -> torch.Tensor:
        """
        hrtf: Input HRTF data, shape (batch_size, 1, hrtf_num_rows, hrtf_row_width)
              e.g., (batch_size, 1, 793, 108)
        Returns:
            feature: Encoded global HRTF feature, shape (batch_size, latent_feature_dim)
        """
        if hrtf.dim() == 4 and hrtf.shape[1] == 1:
            # Reshape to (batch_size, hrtf_num_rows, hrtf_row_width)
            # hrtf_num_rows becomes sequence length, hrtf_row_width becomes feature dim per token
            x = hrtf.squeeze(1) 
        elif hrtf.dim() == 3: # If input is already (batch_size, num_rows, row_width)
            x = hrtf
        else:
            raise ValueError(f"Unsupported HRTF input shape for Transformer: {hrtf.shape}")
        
        # x shape: (batch_size, seq_len=hrtf_num_rows, features=d_model)
        
        # If using input_projection:
        # x = self.input_projection(x)

        x = self.pos_encoder(x)  # Add positional encoding
        
        transformer_output = self.transformer_encoder(x)  # Output shape (batch, hrtf_num_rows, d_model)
        
        # Aggregate the transformer output. Here, we take the mean across the sequence dimension.
        aggregated_output = transformer_output[:, 0:2,:].reshape(transformer_output.shape[0], -1)  # Shape (batch, 2 * d_model)
        
        feature = self.output_fc(aggregated_output)  # Shape (batch, latent_feature_dim)
        
        return feature


# --- HrtfDecoder (逐行MLP解码器，保持不变) ---
class HrtfDecoder(nn.Module):
    def __init__(self, latent_feature_dim=128, pos_dim_per_row=3, hrtf_row_output_width=108, decoder_mlp_hidden_dims=None):
        super().__init__()
        self.latent_feature_dim = latent_feature_dim
        self.pos_dim_per_row = pos_dim_per_row
        self.hrtf_row_output_width = hrtf_row_output_width
        if decoder_mlp_hidden_dims is None:
            decoder_mlp_hidden_dims = [256, 256]
        mlp_layers = []
        current_dim = self.latent_feature_dim + self.pos_dim_per_row
        for hidden_dim in decoder_mlp_hidden_dims:
            mlp_layers.append(nn.Linear(current_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            current_dim = hidden_dim
        mlp_layers.append(nn.Linear(current_dim, self.hrtf_row_output_width))
        self.row_decoder_mlp = nn.Sequential(*mlp_layers)
    def forward(self, feature, pos):
        batch_size, num_rows = feature.shape[0], pos.shape[1]
        feature_expanded = feature.unsqueeze(1).expand(-1, num_rows, -1)
        decoder_input_per_row = torch.cat([feature_expanded, pos], dim=2)
        decoder_input_flat = decoder_input_per_row.view(batch_size * num_rows, -1)
        decoded_rows_flat = self.row_decoder_mlp(decoder_input_flat)
        reconstructed_hrtf_rows = decoded_rows_flat.view(batch_size, num_rows, self.hrtf_row_output_width)
        reconstructed_hrtf = reconstructed_hrtf_rows.unsqueeze(1)
        return reconstructed_hrtf

# --- HRTFAutoencoder (更新以支持Transformer编码器) ---
class HRTFAutoencoder(nn.Module):
    def __init__(self, latent_feature_dim=128, pos_dim_per_row=3, 
                 hrtf_num_rows=793, hrtf_row_width=108, 
                 encoder_type="transformer", # "fc" or "transformer"
                 decoder_mlp_hidden_dims=None, 
                 encoder_fc_config=None, # For HrtfEncoderFC
                 encoder_transformer_config=None # For HrtfTransformerEncoder
                 ): 
        super().__init__()
        
        self.hrtf_num_rows = hrtf_num_rows
        self.hrtf_row_width = hrtf_row_width
        self.encoder_type = encoder_type

        if encoder_transformer_config is None:
            encoder_transformer_config = {} # Default empty config
        self.encoder = HrtfTransformerEncoder(
            hrtf_row_width=self.hrtf_row_width, # This is d_model for Transformer
            num_heads=encoder_transformer_config.get("num_heads", 4), # Default 4 heads
            num_encoder_layers=encoder_transformer_config.get("num_encoder_layers", 3), # Default 3 layers
            dim_feedforward=encoder_transformer_config.get("dim_feedforward", 512), # Default 512
            latent_feature_dim=latent_feature_dim,
            dropout=encoder_transformer_config.get("dropout", 0.1), # Default 0.1 dropout
            hrtf_num_rows=self.hrtf_num_rows
        )
        self.model_name = "HRTF_AE_TransformerEncoder_RowWiseDecoder"

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
latent_dim = 256 
pos_dim_for_each_row = 3
num_hrtf_rows = 793       # HRTF的行数 (序列长度)
width_per_hrtf_row = 108  # HRTF每行的宽度 (特征维度 d_model)
current_encoder_type = "transformer"

# 为Transformer编码器配置 (如果选择 "transformer")
# d_model (hrtf_row_width=108) 必须能被 num_heads 整除
transformer_encoder_settings = {
    "num_heads": 6,             # 例如 2, 3, 4, 6, 9, 12 (108 % num_heads == 0)
    "num_encoder_layers": 6,
    "dim_feedforward": 512,     # 通常是 d_model 的 2-4 倍
    "dropout": 0.1
}

# 为解码器MLP配置
decoder_mlp_layers = [512, 256, 128] # 可根据需要调整

model = HRTFAutoencoder(
    latent_feature_dim=latent_dim,
    pos_dim_per_row=pos_dim_for_each_row,
    hrtf_num_rows=num_hrtf_rows,
    hrtf_row_width=width_per_hrtf_row,
    encoder_type=current_encoder_type,
    decoder_mlp_hidden_dims=decoder_mlp_layers,
    encoder_transformer_config=transformer_encoder_settings if current_encoder_type == "transformer" else None
).to(device)

print(f"Using {model.encoder_type.upper()} Encoder. Model Name: {model.model_name}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
# print(model) # 取消注释以查看详细模型结构

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
loss_function = nn.MSELoss()
num_epochs = 500 # 示例 epoch 数
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

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
    scheduler.step()  # 更新学习率
    # 保存模型
    if (epoch + 1) % 10 == 0: # 每10个epoch保存一次
        torch.save(model.state_dict(), f"{weightdir}/hrtf_ae_model_epoch_{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}")

print("Training finished.")