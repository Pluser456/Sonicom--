
embed_dim = 192
pos_dim_for_each_row = 8
# num_hrtf_rows = 793       # HRTF的行数 (序列长度)
num_hrtf_rows = 2562       # HRTF的行数 (序列长度)
# width_per_hrtf_row = 108  # HRTF每行的宽度 (特征维度 embed_dim)
hrtf_row_len = 90  # HRTF每行的宽度 (特征维度 embed_dim)
current_encoder_type = "transformer"
encoder_out_vec_num = 8
# 为Transformer编码器配置 (如果选择 "transformer")
# embed_dim (hrtf_row_width=108) 必须能被 num_heads 整除
transformer_encoder_settings = {
    "num_heads": 6,          # 例如 2, 3, 4, 6, 9, 12 (108 % num_heads == 0)
    "num_encoder_layers": 4, # 15
    "dim_feedforward": 512,     # 通常是 embed_dim 的 2-4 倍
    "dropout": 0.0
}
transformer_decoder_settings = {
    "num_heads": 6,          # 例如 2, 3, 4, 6, 9, 12 (108 % num_heads == 0)
    "num_decoder_layers": 10, # 15
    "dim_feedforward": 512,  # 通常是 embed_dim 的 2-4 倍
    "dropout": 0.0
}


# VQ-VAE 特定参数
num_codebook_embeddings = 8 # 码本大小
use_VQ = False  # 是否使用向量量化
input_pos_as_seq = False  # 是否将位置作为序列输入给解码器

# 码本相关参数
commitment_cost_beta = 0.25
tolerance_for_calc_threshold = 330  # 采用“容忍度”计算码本向量被视为“死”的阈值，为 None 时，直接采用固定阈值
decay = 0.99  # EMA 更新的衰减率