import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
                 dim_feedforward=1024, latent_feature_dim=256, dropout=0.1, feature_num=2,
                 hrtf_num_rows=793): # hrtf_num_rows for PositionalEncoding max_len
        super().__init__()
        self.d_model = hrtf_row_width  # Feature dimension of each HRTF row

        # Optional: Input projection if hrtf_row_width is not the desired d_model
        # self.input_projection = nn.Linear(hrtf_row_width, self.d_model)
        # For simplicity, we assume hrtf_row_width is d_model
        self.feature_num = feature_num # 使用transformer输出的前几列作为特征输出
        self.pos_encoder = PositionalEncoding(self.d_model, dropout, max_len=hrtf_num_rows + 10) # +10 for safety margin
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Crucial: input format (batch, seq, feature)
            norm_first=True # 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Linear layer to map the aggregated output of Transformer to latent_feature_dim
        self.output_fc = nn.Linear(self.d_model*feature_num, latent_feature_dim)

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

        x = self.pos_encoder(x)  # Add positional encoding
        
        transformer_output = self.transformer_encoder(x)  # Output shape (batch, hrtf_num_rows, d_model)
        
        # Aggregate the transformer output. Here, we take the mean across the sequence dimension.
        aggregated_output = transformer_output[:, 0:self.feature_num, :].reshape(transformer_output.shape[0], -1)  # Shape (batch, feature_num * d_model)
        
        feature = self.output_fc(aggregated_output)  # Shape (batch, latent_feature_dim)
        
        return feature


# --- HrtfDecoder (逐行MLP解码器，保持不变) ---
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, use_batchnorm=True):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        
        self.linear = nn.Linear(input_dim, output_dim)
        if self.use_batchnorm:
            self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        
        if input_dim == output_dim:
            self.shortcut = nn.Identity()
        else:
            # 如果维度不匹配，使用线性层进行投影以匹配残差连接
            self.shortcut = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.linear(x)
        if self.use_batchnorm:
            out = self.bn(out)
        out = self.relu(out)
        
        out = out + residual # 添加残差
        return out

class HrtfDecoder(nn.Module):
    def __init__(self, latent_feature_dim=128, pos_dim_per_row=3, hrtf_row_output_width=108, decoder_mlp_hidden_dims=None):
        super().__init__()
        self.latent_feature_dim = latent_feature_dim
        self.pos_dim_per_row = pos_dim_per_row
        self.hrtf_row_output_width = hrtf_row_output_width

        mlp_layers = []
        current_dim = self.latent_feature_dim + self.pos_dim_per_row
        
        if decoder_mlp_hidden_dims is None or len(decoder_mlp_hidden_dims) == 0:
            # 没有隐藏层，直接线性连接到输出
            mlp_layers.append(nn.Linear(current_dim, self.hrtf_row_output_width))
        else:
            # 第一个隐藏层: 使用 ResidualBlock (不带 BatchNorm，因为通常第一层后不加)
            first_hidden_dim = decoder_mlp_hidden_dims[0]
            mlp_layers.append(ResidualBlock(current_dim, first_hidden_dim, use_batchnorm=False))
            current_dim = first_hidden_dim

            # 后续的隐藏层: 使用 ResidualBlock (带 BatchNorm)
            for i in range(1, len(decoder_mlp_hidden_dims)):
                current_hidden_dim = decoder_mlp_hidden_dims[i]
                mlp_layers.append(ResidualBlock(current_dim, current_hidden_dim, use_batchnorm=True))
                current_dim = current_hidden_dim
            
            # MLP 的输出层 (不使用残差块，直接线性输出)
            mlp_layers.append(nn.Linear(current_dim, self.hrtf_row_output_width))
            
        self.row_decoder_mlp = nn.Sequential(*mlp_layers)

    def forward(self, feature, pos):
        batch_size, num_rows = feature.shape[0], pos.shape[1]
        feature_expanded = feature.unsqueeze(1).expand(-1, num_rows, -1)
        decoder_input_per_row = torch.cat([feature_expanded, pos], dim=2)
        decoder_input_flat = decoder_input_per_row.view(batch_size * num_rows, -1)
        
        # 注意：BatchNorm1d期望输入是 (N, C) 或 (N, C, L)，这里是 (N, C)
        # 如果Sequential中包含BatchNorm1d，输入必须是2D的
        decoded_rows_flat = self.row_decoder_mlp(decoder_input_flat)
        
        reconstructed_hrtf_rows = decoded_rows_flat.view(batch_size, num_rows, self.hrtf_row_output_width)
        reconstructed_hrtf = reconstructed_hrtf_rows.unsqueeze(1)
        return reconstructed_hrtf

# --- HRTFAutoencoder (更新以支持Transformer编码器) ---
class HRTFAutoencoder(nn.Module):
    def __init__(self, latent_feature_dim=128, pos_dim_per_row=3, 
                 hrtf_num_rows=793, hrtf_row_width=108, 
                 decoder_mlp_hidden_dims=None, 
                 encoder_transformer_config=None # For HrtfTransformerEncoder
                 ): 
        super().__init__()
        
        self.hrtf_num_rows = hrtf_num_rows
        self.hrtf_row_width = hrtf_row_width

        if encoder_transformer_config is None:
            encoder_transformer_config = {} # Default empty config
        self.encoder = HrtfTransformerEncoder(
            hrtf_row_width=self.hrtf_row_width, # This is d_model for Transformer
            num_heads=encoder_transformer_config.get("num_heads", 4), # Default 4 heads
            num_encoder_layers=encoder_transformer_config.get("num_encoder_layers", 3), # Default 3 layers
            dim_feedforward=encoder_transformer_config.get("dim_feedforward", 512), # Default 512
            latent_feature_dim=latent_feature_dim,
            dropout=encoder_transformer_config.get("dropout", 0.1), # Default 0.1 dropout
            feature_num=3,
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