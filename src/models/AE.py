import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from vector_quantize_pytorch import SimVQ, VectorQuantize, FSQ,GroupedResidualVQ, ResidualVQ
from ema_vq import VectorQuantization
# --- Rotary Positional Encoding ---
pass
    
# --- Transformer Encoder for HRTF ---
class HrtfTransformerEncoder(nn.Module):
    def __init__(self, hrtf_row_len, embed_dim, num_heads, num_layers,
                 dim_feedforward, dropout, feature_num): # hrtf_num_rows for PositionalEncoding max_len
        super().__init__()
        self.embed_dim = embed_dim  # Feature dimension of each HRTF row
        self.enc_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.enc_token, std=0.02)
        self.input_projection = nn.Linear(hrtf_row_len, self.embed_dim)
        self.feature_num = feature_num # 使用transformer输出的前几列作为特征输出
        self.pos_embed_mlp = nn.Sequential(nn.Linear(3, self.embed_dim),
                                            nn.GELU(), nn.Linear(self.embed_dim, self.embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Crucial: input format (batch, seq, feature)
            norm_first=True # 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, hrtf: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        hrtf: Input HRTF data, shape (batch_size, hrtf_num_rows, hrtf_row_width)
              e.g., (batch_size, 793, 108)
        Returns:
            feature: Encoded global HRTF feature, shape (batch_size, latent_feature_dim)
        """
        pos = self.pos_embed_mlp(pos)  # Add positional encoding
        x = self.input_projection(hrtf)  # Project to embed_dim
        x = x + pos  # Add positional encoding to input
        enc_tokens = self.enc_token.expand(x.size(0), -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat([enc_tokens, x], dim=1)  # (batch_size, hrtf_num_rows + 1, embed_dim)
        transformer_output = self.transformer_encoder(x)  # Output shape (batch, hrtf_num_rows, d_model)

        output = transformer_output[:, 0:self.feature_num, :]  # Shape (batch, feature_num, d_model)
        return output # (batch_size, feature_num, d_model)

class HrtfTransformerDecoder(nn.Module):
    def __init__(self, embed_dim, nhead, num_decoder_layers, dim_feedforward, dropout,
                 hrtf_row_len, input_pos_as_seq):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embed_mlp = nn.Sequential(nn.Linear(3, self.embed_dim),
                                    nn.GELU(), nn.Linear(self.embed_dim, self.embed_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_projector = nn.Linear(embed_dim, hrtf_row_len)
        self.input_pos_as_seq = input_pos_as_seq

    def forward(self, memory, target_pos_sequence):
        tgt_embedded = self.pos_embed_mlp(target_pos_sequence)
        if self.input_pos_as_seq:
            # 作为序列输入
            decoder_output = self.transformer_decoder(tgt=tgt_embedded, memory=memory)
        else:
            # 现在是将每个位置独立解码
            memory = memory.unsqueeze(1).repeat(1, target_pos_sequence.shape[1], 1, 1).flatten(0, 1)
            tgt_embedded_flatten = tgt_embedded.reshape(-1, tgt_embedded.shape[2]).unsqueeze(1)
            decoder_output = self.transformer_decoder(memory=memory, tgt=tgt_embedded_flatten)
            decoder_output = decoder_output.reshape(tgt_embedded.shape)
        reconstructed_hrtf = self.output_projector(decoder_output)
        return reconstructed_hrtf
    
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
                 hrtf_row_len, # Transformer 的 d_model, 也是 VQ 的 embedding_dim
                 encoder_out_vec_num,
                 embed_dim,
                 encoder_transformer_config,
                 decoder_transformer_config,
                 num_embeddings, 
                 use_VQ,
                 input_pos_as_seq,
                decay,
                tolerance_for_calc_threshold,
                 ): 
        super().__init__()
        
        self.hrtf_row_len = hrtf_row_len
        self.encoder_out_vec_num = encoder_out_vec_num # 例如 108
        self.num_embeddings = num_embeddings # 码表大小
        self.encoder = HrtfTransformerEncoder(
            hrtf_row_len=self.hrtf_row_len,
            embed_dim=embed_dim,
            num_heads=encoder_transformer_config["num_heads"],
            num_layers=encoder_transformer_config["num_encoder_layers"],
            dim_feedforward=encoder_transformer_config["dim_feedforward"],
            dropout=encoder_transformer_config["dropout"],
            feature_num=self.encoder_out_vec_num, # 编码器输出此长度的序列 (例如 108)
        )

        # self.vq_layer = VectorQuantize(dim = hrtf_row_len, codebook_size=num_embeddings,
        #                             kmeans_init = True,   # set to True
        #                             kmeans_iters = 10,     # number of kmeans iterations to calculate the centroids for the codebook on init
        #                             # threshold_ema_dead_code=2,
        #                             # use_cosine_sim=True, # 使用余弦相似度
        #                             commitment_weight = commitment_cost, # commitment cost
        #                             )

        self.vq_layer = nn.ModuleList([ VectorQuantization(
            dim=embed_dim, # Transformer的嵌入维度
            codebook_size=num_embeddings,
            codebook_dim=embed_dim,
            decay=decay,
            kmeans_init=False,
            tolerance_for_calc_threshold=tolerance_for_calc_threshold, 
        ) for _ in range(self.encoder_out_vec_num)])

        self.decoder = HrtfTransformerDecoder(
            embed_dim=embed_dim,
            nhead=decoder_transformer_config["num_heads"],
            num_decoder_layers=decoder_transformer_config["num_decoder_layers"],
            dim_feedforward=decoder_transformer_config["dim_feedforward"],
            dropout=decoder_transformer_config["dropout"],
            hrtf_row_len=self.hrtf_row_len,
            input_pos_as_seq=input_pos_as_seq # 将位置作为序列输入
        )
        self.use_VQ = use_VQ

    def quantize(self, ze):
        zq_list = []
        indices_list = []
        vq_loss_total = 0
        for i in range(self.encoder_out_vec_num):
            zq, indices, vq_loss = self.vq_layer[i](ze[:, i, :])
            zq_list.append(zq)
            indices_list.append(indices)
            vq_loss_total += vq_loss
        vq_loss = vq_loss_total / self.encoder_out_vec_num
        zq = torch.stack(zq_list, dim=1)
        indices = torch.stack(indices_list, dim=1)
        return zq,indices,vq_loss

    def forward(self, hrtf_data, pos_data):
        ze = self.encoder(hrtf_data, pos_data) # ze: (B, encoder_out_vec_num, d_model)
        if self.use_VQ:
            zq, indices, vq_loss = self.quantize(ze)
            reconstructed_hrtf = self.decoder(zq, pos_data)
            return reconstructed_hrtf, vq_loss, indices
        else:
            reconstructed_hrtf = self.decoder(ze, pos_data)
            return reconstructed_hrtf, torch.zeros(1).cuda(), torch.zeros(1, 1)