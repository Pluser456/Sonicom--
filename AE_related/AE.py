import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from vector_quantize_pytorch import VectorQuantize, FSQ
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
                 dim_feedforward=1024, dropout=0.1, feature_num=2,
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
        aggregated_output = transformer_output[:, 0:self.feature_num, :]  # Shape (batch, feature_num, d_model)
        return aggregated_output # (batch_size, feature_num, d_model)


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
    def __init__(self, decoder_input_dim=128, pos_dim_per_row=3, decoder_output_dim=108, decoder_mlp_hidden_dims=None):
        super().__init__()
        self.latent_feature_dim = decoder_input_dim
        self.pos_dim_per_row = pos_dim_per_row
        self.hrtf_row_output_width = decoder_output_dim

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
    
class HrtfTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout,
                 pos_dim_input=3, hrtf_row_output_width=108, max_output_seq_len=793 + 10):
        super().__init__()
        self.pos_embed = nn.Linear(pos_dim_input, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_output_seq_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_projector = nn.Linear(d_model, hrtf_row_output_width)

    def forward(self, memory, target_pos_sequence):
        tgt_embedded = self.pos_embed(target_pos_sequence)
        # tgt_embedded = self.pos_encoder(tgt_embedded)
        memory = memory.unsqueeze(1).repeat(1, target_pos_sequence.shape[1], 1, 1).flatten(0, 1)
        tgt_embedded_flatten = tgt_embedded.reshape(-1, tgt_embedded.shape[2]).unsqueeze(1)
        decoder_output = self.transformer_decoder(memory=memory, tgt=tgt_embedded_flatten)
        decoder_output = decoder_output.reshape(tgt_embedded.shape)
        reconstructed_hrtf_rows = self.output_projector(decoder_output)
        reconstructed_hrtf = reconstructed_hrtf_rows.unsqueeze(1)
        return reconstructed_hrtf

# --- HRTFAutoencoder (更新以支持Transformer编码器) ---
class HRTFAutoencoder(nn.Module):
    def __init__(self, pos_dim_per_row=3, 
                 hrtf_num_rows=793, hrtf_row_width=108, 
                 encoder_out_vec_num=3, # Transformer 输出的前几列作为特征输出
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
            dropout=encoder_transformer_config.get("dropout", 0.1), # Default 0.1 dropout
            feature_num=encoder_out_vec_num,
            hrtf_num_rows=self.hrtf_num_rows
        )
        self.model_name = "HRTF_AE_TransformerEncoder_TransformerDecoder"

        # self.decoder = HrtfDecoder(
        #     decoder_input_dim=self.hrtf_row_width*encoder_out_vec_num,
        #     pos_dim_per_row=pos_dim_per_row,
        #     decoder_output_dim=self.hrtf_row_width,
        #     decoder_mlp_hidden_dims=decoder_mlp_hidden_dims
        # )
        # self.pos_dim_per_row = pos_dim_per_row
        # Linear layer to map the aggregated output of Transformer to latent_feature_dim
        # self.encoder_mlp = nn.Linear(, latent_feature_dim)
        self.decoder = HrtfTransformerDecoder(
            d_model=self.hrtf_row_width,
            nhead=encoder_transformer_config.get("num_heads", 4),
            num_decoder_layers=encoder_transformer_config.get("num_encoder_layers", 3),
            dim_feedforward=encoder_transformer_config.get("dim_feedforward", 512),
            dropout=encoder_transformer_config.get("dropout", 0.1),
            pos_dim_input=pos_dim_per_row,
            hrtf_row_output_width=self.hrtf_row_width
        )
            

    def forward(self, hrtf_data, pos_data):
        feature = self.encoder(hrtf_data)
        # feature = feature.reshape(hrtf_data.shape[0], -1)
        # feature = self.encoder_mlp(feature)  # (B, latent_feature_dim)
        reconstructed_hrtf = self.decoder(feature, pos_data)
        return reconstructed_hrtf, feature
    
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
    
class HRTF_VQVAE(nn.Module):
    def __init__(self, 
                 hrtf_row_width, # Transformer 的 d_model, 也是 VQ 的 embedding_dim
                 hrtf_num_rows,  # Transformer 输入的原始序列长度
                 encoder_out_vec_num, # VQ 输入的目标序列长度 (例如 108)
                 encoder_transformer_config,
                 num_embeddings, 
                 commitment_cost,
                 pos_dim_per_row,
                 decoder_mlp_hidden_dims
                 ): 
        super().__init__()
        
        self.hrtf_row_width = hrtf_row_width
        self.encoder_out_vec_num = encoder_out_vec_num # 例如 108
        self.num_embeddings = num_embeddings # 码表大小
        self.encoder = HrtfTransformerEncoder(
            hrtf_row_width=self.hrtf_row_width, 
            hrtf_num_rows=hrtf_num_rows, # 原始输入序列长度 (例如 793)
            feature_num=self.encoder_out_vec_num, # 编码器输出此长度的序列 (例如 108)
            num_heads=encoder_transformer_config.get("num_heads", 4),
            num_encoder_layers=encoder_transformer_config.get("num_encoder_layers", 3),
            dim_feedforward=encoder_transformer_config.get("dim_feedforward", 512),
            dropout=encoder_transformer_config.get("dropout", 0.1)
        )
        self.model_name = "HRTF_VQVAE_FlattenedVQ_Decoder"

        # self.vq_layer = VectorQuantizer(num_embeddings, self.hrtf_row_width, commitment_cost)
        # self.vq_layer = VectorQuantize(dim =hrtf_row_width,codebook_size=num_embeddings, commitment_weight=commitment_cost)

        self.vq_layer = FSQ(levels=[8, 8, 8])

        # self.projector = nn.Linear(self.hrtf_row_width, 1)

        # 解码器的 latent_feature_dim 是展平后的 VQ 输出维度
        decoder_latent_dim = self.hrtf_row_width * self.encoder_out_vec_num
        self.decoder = HrtfDecoder(
            decoder_input_dim=decoder_latent_dim, 
            pos_dim_per_row=pos_dim_per_row,
            decoder_output_dim=self.hrtf_row_width, # 假设输出的每行 HRTF 宽度仍为 d_model
            decoder_mlp_hidden_dims=decoder_mlp_hidden_dims
        )

    def forward(self, hrtf_data, pos_data):
        # hrtf_data: (B, 1, hrtf_num_rows, hrtf_row_width)
        # pos_data: (B, hrtf_num_rows, pos_dim_per_row)

        # ze: (B, target_seq_len_for_vq, d_model) 例如 (B, 108, 108)
        ze = self.encoder(hrtf_data) 
        
        # zq: (B, target_seq_len_for_vq, d_model), vq_loss, indices 例如 (B, 108, 108)
        ze = ze.permute(0, 2, 1)
        zq, indices = self.vq_layer(ze) 
        
        # 将 zq 展平
        zq_flat = zq.reshape(zq.shape[0], -1) # (B, target_seq_len_for_vq * 1)
        
        # 将展平后的 zq 和 pos_data 传递给解码器
        # reconstructed_hrtf: (B, 1, hrtf_num_rows, hrtf_row_width)
        reconstructed_hrtf = self.decoder(zq_flat, pos_data)
        
        return reconstructed_hrtf, torch.zeros(1).cuda(), indices