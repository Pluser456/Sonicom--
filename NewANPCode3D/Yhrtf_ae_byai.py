import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import sys
import math
# Assuming new_dataset.py and utils.py are in the same directory or accessible in PYTHONPATH
from new_dataset import SonicomDataSet
from utils import split_dataset
from torch.utils.tensorboard import SummaryWriter # <--- 添加导入

# --- Configuration ---
weightname = "model-100.pth" # Base name, epoch will be added
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
usediff = False
weightdir = "./HRTF_VQVAE_TransformerDecoder_weights_compact"
log_dir = "./runs/HRTF_VQVAE_TransformerDecoder_compact" # <--- TensorBoard 日志目录
ear_dir = "Ear_image_gray" # Make sure this path is correct
hrtf_dir_name = "FFT_HRTF" # Make sure this path is correct

if os.path.exists(weightdir) is False:
    os.makedirs(weightdir)

positions_chosen_num = 793
inputform = "image" # or "hrtf" or "anthropometric" depending on your dataset

# --- Dataset and DataLoader ---
dataset_paths = split_dataset(ear_dir, hrtf_dir_name, inputform=inputform)

train_dataset = SonicomDataSet(
    dataset_paths["train_hrtf_list"],
    dataset_paths["left_train"],
    dataset_paths["right_train"],
    use_diff=usediff,
    calc_mean=True,
    status="test", # Use 'train' for shuffling coordinates if desired
    positions_chosen_num=positions_chosen_num,
    inputform=inputform,
    mode="left" # or "right" or "both"
)
test_dataset = SonicomDataSet(
    dataset_paths["test_hrtf_list"],
    dataset_paths["left_test"],
    dataset_paths["right_test"],
    calc_mean=False,
    status="test",
    inputform=inputform,
    mode="left", # or "right" or "both"
    use_diff=usediff,
    provided_mean_left=train_dataset.log_mean_hrtf_left,
    provided_mean_right=train_dataset.log_mean_hrtf_right,
    positions_chosen_num=positions_chosen_num
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16, # Adjust based on GPU memory
    shuffle=True,
    collate_fn=train_dataset.collate_fn,
    # num_workers=4 if device.type == 'cuda' else 0, # Use num_workers for faster loading
    pin_memory=True if device.type == 'cuda' else False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16, # Adjust based on GPU memory
    shuffle=False,
    collate_fn=test_dataset.collate_fn,
    # num_workers=4 if device.type == 'cuda' else 0,
    pin_memory=True if device.type == 'cuda' else False
)

# --- VectorQuantizer ---
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1./self._num_embeddings, 1./self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        input_shape = inputs.shape
        if inputs.dim() == 4:
            flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self._embedding_dim)
        elif inputs.dim() == 3:
            flat_input = inputs.reshape(-1, self._embedding_dim)
        else:
            raise ValueError(f"Input tensor to VQ has unsupported dimensions: {inputs.dim()}")

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self._embedding.weight)
        if inputs.dim() == 4:
             quantized = quantized.view(input_shape[0], input_shape[2], input_shape[3], self._embedding_dim)
        elif inputs.dim() == 3:
             quantized = quantized.view(input_shape[0], input_shape[1], self._embedding_dim)
        
        e_latent_loss = F.mse_loss(quantized.detach(), flat_input.view_as(quantized))
        q_latent_loss = F.mse_loss(quantized, flat_input.view_as(quantized).detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized_sg = flat_input.view_as(quantized) + (quantized - flat_input.view_as(quantized)).detach()
        
        if inputs.dim() == 4:
            quantized_out = quantized_sg.permute(0, 3, 1, 2).contiguous() 
        elif inputs.dim() == 3:
            quantized_out = quantized_sg 
            
        if inputs.dim() == 4:
            indices_reshaped = encoding_indices.view(input_shape[0], input_shape[2]*input_shape[3])
        elif inputs.dim() == 3:
            indices_reshaped = encoding_indices.view(input_shape[0], input_shape[1])
        return quantized_out, loss, indices_reshaped

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[0, :, 1::2] = torch.cos(position * div_term)[:,:d_model // 2]
        else:
            pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# --- Transformer Encoder for HRTF ---
class HrtfTransformerEncoder(nn.Module):
    def __init__(self, hrtf_row_width, num_heads, num_encoder_layers,
                 dim_feedforward, dropout, hrtf_num_rows, target_seq_len_for_vq):
        super().__init__()
        self.d_model = hrtf_row_width
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=dropout, max_len=hrtf_num_rows + 10)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.target_seq_len_for_vq = target_seq_len_for_vq
        self.seq_len_projector = nn.Linear(hrtf_num_rows, target_seq_len_for_vq)

    def forward(self, hrtf: torch.Tensor) -> torch.Tensor:
        if hrtf.dim() == 4 and hrtf.shape[1] == 1: x = hrtf.squeeze(1)
        elif hrtf.dim() == 3: x = hrtf
        else: raise ValueError(f"Unsupported HRTF input shape for Encoder: {hrtf.shape}")
        
        x = self.pos_encoder(x)
        transformer_output = self.transformer_encoder(x)
        # projected_output = transformer_output.permute(0, 2, 1)
        # projected_output = self.seq_len_projector(projected_output)
        # projected_output = projected_output.permute(0, 2, 1)
        projected_output = transformer_output[:, :self.target_seq_len_for_vq, :]
        return projected_output

# --- HrtfTransformerDecoder ---
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

    def forward(self, target_pos_sequence, memory):
        tgt_embedded = self.pos_embed(target_pos_sequence)
        tgt_embedded = self.pos_encoder(tgt_embedded)
        decoder_output = self.transformer_decoder(tgt=tgt_embedded, memory=memory)
        reconstructed_hrtf_rows = self.output_projector(decoder_output)
        reconstructed_hrtf = reconstructed_hrtf_rows.unsqueeze(1)
        return reconstructed_hrtf

# --- HRTF_VQVAE Model ---
class HRTF_VQVAE(nn.Module):
    def __init__(self, hrtf_input_row_width, hrtf_num_rows, encoder_d_model, 
                 encoder_target_seq_len, encoder_transformer_config, num_embeddings, 
                 commitment_cost, decoder_transformer_config, pos_dim_per_row, 
                 hrtf_output_row_width): 
        super().__init__()
        self.model_name = "HRTF_VQVAE_TrDec"
        if hrtf_input_row_width != encoder_d_model:
            self.input_projection = nn.Linear(hrtf_input_row_width, encoder_d_model)
        else:
            self.input_projection = nn.Identity()

        self.encoder = HrtfTransformerEncoder(
            hrtf_row_width=encoder_d_model, hrtf_num_rows=hrtf_num_rows,
            target_seq_len_for_vq=encoder_target_seq_len,
            **encoder_transformer_config
        )
        self.vq_layer = VectorQuantizer(num_embeddings, encoder_d_model, commitment_cost)
        self.decoder = HrtfTransformerDecoder(
            d_model=encoder_d_model, 
            pos_dim_input=pos_dim_per_row,
            hrtf_row_output_width=hrtf_output_row_width, 
            max_output_seq_len=hrtf_num_rows + 10,
            **decoder_transformer_config
        )

    def forward(self, hrtf_data, pos_data):
        hrtf_proc = hrtf_data.squeeze(1) 
        hrtf_projected = self.input_projection(hrtf_proc)
        ze = self.encoder(hrtf_projected)
        zq, vq_loss, _ = self.vq_layer(ze) 
        reconstructed_hrtf = self.decoder(target_pos_sequence=pos_data, memory=zq)
        return reconstructed_hrtf, vq_loss

# --- Model Instantiation and Parameters ---
hrtf_input_row_width_original = 108
hrtf_num_rows_original = positions_chosen_num # Use from global config

# Encoder: d_model * target_seq_len <= 1024
encoder_d_model_config = 108       # Feature dimension for encoder/VQ
encoder_target_seq_len_config = 8 # Sequence length for VQ input (32*32 = 1024)

num_codebook_embeddings = 512 
commitment_cost_beta = 0.25

encoder_transformer_settings = {
    "num_heads": 4, # Divisor of encoder_d_model_config
    "num_encoder_layers": 4, 
    "dim_feedforward": encoder_d_model_config * 4,
    "dropout": 0.1
}
decoder_transformer_settings = {
    "nhead": 4, # Divisor of encoder_d_model_config
    "num_decoder_layers": 4,
    "dim_feedforward": encoder_d_model_config * 4,
    "dropout": 0.1
}
pos_dim_for_each_row = 3
hrtf_output_row_width_target = hrtf_input_row_width_original

model = HRTF_VQVAE(
    hrtf_input_row_width=hrtf_input_row_width_original,
    hrtf_num_rows=hrtf_num_rows_original,
    encoder_d_model=encoder_d_model_config,
    encoder_target_seq_len=encoder_target_seq_len_config,
    encoder_transformer_config=encoder_transformer_settings,
    num_embeddings=num_codebook_embeddings,
    commitment_cost=commitment_cost_beta,
    decoder_transformer_config=decoder_transformer_settings,
    pos_dim_per_row=pos_dim_for_each_row,
    hrtf_output_row_width=hrtf_output_row_width_target
).to(device)

print(f"Using {model.model_name} on {device}")
print(f"Encoder d_model: {encoder_d_model_config}, Enc target_seq_len: {encoder_target_seq_len_config}")
print(f"Total latent dim (d_model*seq_len): {encoder_d_model_config * encoder_target_seq_len_config}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# --- Optimizer, Loss, Scheduler ---
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
reconstruction_loss_fn = nn.MSELoss()
num_epochs = 2000 # Adjust as needed
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
writer = SummaryWriter(log_dir=f"{log_dir}/diff_{str(usediff)}_enc_d{encoder_d_model_config}_seq{encoder_target_seq_len_config}_vq{num_codebook_embeddings}")
# --- Training Loop ---
global_step_train = 0 # <--- 添加全局训练步数计数器
global_step_val = 0   # <--- 添加全局验证步数计数器

for epoch in range(num_epochs):
    model.train()
    epoch_loss_recon = 0
    epoch_loss_vq = 0
    epoch_total_loss = 0 # <--- 添加 epoch 总损失
    
    train_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", file=sys.stdout)
    for i, batch in enumerate(train_pbar):
        hrtf = batch["hrtf"].to(device) 
        pos = batch["position"].to(device)

        optimizer.zero_grad(set_to_none=True) 
        reconstructed_hrtf, vq_loss = model(hrtf, pos)
        recon_loss = reconstruction_loss_fn(reconstructed_hrtf, hrtf)
        total_loss = recon_loss + vq_loss 
        total_loss.backward()
        optimizer.step()
        
        epoch_loss_recon += recon_loss.item()
        epoch_loss_vq += vq_loss.item()
        epoch_total_loss += total_loss.item() # <--- 累加总损失
        
        # <--- 添加 TensorBoard 日志 (per step)
        writer.add_scalar('Loss/Train/Reconstruction_step', recon_loss.item(), global_step_train)
        writer.add_scalar('Loss/Train/VQ_step', vq_loss.item(), global_step_train)
        writer.add_scalar('Loss/Train/Total_step', total_loss.item(), global_step_train)
        global_step_train += 1
        train_pbar.set_postfix_str(f"RcnL: {epoch_loss_recon/(i+1):.4f}, VqL: {epoch_loss_vq/(i+1):.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

    avg_recon_loss_train = epoch_loss_recon / len(train_loader)
    avg_vq_loss_train = epoch_loss_vq / len(train_loader)
    avg_total_loss_train = epoch_total_loss / len(train_loader) # <--- 计算平均总损失
    
    # <--- 添加 TensorBoard 日志 (per epoch)
    writer.add_scalar('Loss/Train/Reconstruction_epoch', avg_recon_loss_train, epoch)
    writer.add_scalar('Loss/Train/VQ_epoch', avg_vq_loss_train, epoch)
    writer.add_scalar('Loss/Train/Total_epoch', avg_total_loss_train, epoch)
    writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
    
    print(f"Epoch {epoch+1} Train Avg: Recon Loss: {avg_recon_loss_train:.4f}, VQ Loss: {avg_vq_loss_train:.4f}, Total Loss: {avg_total_loss_train:.4f}")

    model.eval()
    val_epoch_loss_recon = 0 # <--- 重命名以区分
    val_epoch_loss_vq = 0    # <--- 重命名以区分
    val_epoch_total_loss = 0 # <--- 添加 epoch 总损失
    with torch.no_grad():
        val_pbar = tqdm.tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]", file=sys.stdout)
        for step, batch in enumerate(val_pbar):
            hrtf_val = batch["hrtf"].to(device)
            pos_val = batch["position"].to(device)
            reconstructed_hrtf_val, vq_loss_val = model(hrtf_val, pos_val)
            recon_loss_val = reconstruction_loss_fn(reconstructed_hrtf_val, hrtf_val)
            total_loss_val = recon_loss_val + vq_loss_val # <--- 计算验证总损失
            
            val_epoch_loss_recon += recon_loss_val.item()
            val_epoch_loss_vq += vq_loss_val.item()
            val_epoch_total_loss += total_loss_val.item() # <--- 累加验证总损失

            # <--- 添加 TensorBoard 日志 (per step for validation if desired, or remove for less verbose logging)
            # writer.add_scalar('Loss/Validation/Reconstruction_step', recon_loss_val.item(), global_step_val)
            # writer.add_scalar('Loss/Validation/VQ_step', vq_loss_val.item(), global_step_val)
            # writer.add_scalar('Loss/Validation/Total_step', total_loss_val.item(), global_step_val)
            # global_step_val += 1
            

            val_pbar.set_postfix_str(f"RcnL: {val_epoch_loss_recon/(step+1):.4f}, VqL: {val_epoch_loss_vq/(step+1):.4f}")
    
    avg_recon_loss_val = val_epoch_loss_recon / len(test_loader)
    avg_vq_loss_val = val_epoch_loss_vq / len(test_loader)
    avg_total_loss_val = val_epoch_total_loss / len(test_loader) # <--- 计算平均验证总损失
    # <--- 添加 TensorBoard 日志 (per epoch for validation)
    writer.add_scalar('Loss/Validation/Reconstruction_epoch', avg_recon_loss_val, epoch)
    writer.add_scalar('Loss/Validation/VQ_epoch', avg_vq_loss_val, epoch)
    writer.add_scalar('Loss/Validation/Total_epoch', avg_total_loss_val, epoch)
    scheduler.step()
    
    if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
        save_path = f"{weightdir}/model-{model.model_name}-epoch{epoch+1}-diff{str(usediff)}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

print("Training finished.")
writer.close() # <--- 关闭 writer