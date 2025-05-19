# latent_dim = 256 
# pos_dim_for_each_row = 3
# num_hrtf_rows = 793       # HRTF的行数 (序列长度)
# width_per_hrtf_row = 108  # HRTF每行的宽度 (特征维度 d_model)
# current_encoder_type = "transformer"
# encoder_out_vec_num = 3
# # 为Transformer编码器配置 (如果选择 "transformer")
# # d_model (hrtf_row_width=108) 必须能被 num_heads 整除
# transformer_encoder_settings = {
#     "num_heads": 6,             # 例如 2, 3, 4, 6, 9, 12 (108 % num_heads == 0)
#     "num_encoder_layers": 15,
#     "dim_feedforward": 512,     # 通常是 d_model 的 2-4 倍
#     "dropout": 0.05
# }

# # 为解码器MLP配置
# decoder_mlp_layers = [256, 256, 256, 256, 128] # 可根据需要调整

# # VQ-VAE 特定参数
# num_codebook_embeddings = 256 
# commitment_cost_beta = 0.25

latent_dim = 256 
pos_dim_for_each_row = 3
num_hrtf_rows = 2562       # HRTF的行数 (序列长度)
width_per_hrtf_row = 90  # HRTF每行的宽度 (特征维度 d_model)
current_encoder_type = "transformer"
encoder_out_vec_num = 3
# 为Transformer编码器配置 (如果选择 "transformer")
# d_model (hrtf_row_width=108) 必须能被 num_heads 整除
transformer_encoder_settings = {
    "num_heads": 6,             # 例如 2, 3, 4, 6, 9, 12 (108 % num_heads == 0)
    "num_encoder_layers": 15, # 8
    "dim_feedforward": 512,     # 通常是 d_model 的 2-4 倍
    "dropout": 0.05
}

# 为解码器MLP配置
decoder_mlp_layers = [256, 256, 256, 256, 128] # 可根据需要调整

# VQ-VAE 特定参数
num_codebook_embeddings = 256 
commitment_cost_beta = 0.25