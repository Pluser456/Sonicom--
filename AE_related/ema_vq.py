import torch
from torch import nn, einsum
import torch.nn.functional as F

def cdist_sq(x, y, eps = 1e-16):
    """计算两组向量之间的成对距离
    Args:
    x: (n, d), y: (c, d)
    
    Returns:
    dist: (n, c)"""
    x2 = torch.sum(x ** 2, dim=1)  # (n,)
    y2 = torch.sum(y ** 2, dim=1)  # (c,)
    xy = einsum('n d, c d -> n c', x, y) * -2
    # xy = torch.matmul(x, y.t()) * -2  # (n, c)

    return (x2.unsqueeze(1) + y2.unsqueeze(0) + xy)


def kmeans(
    x,
    num_clusters,
    num_iters,
    use_cosine_sim = False,
):
    dim, dtype, device = x.shape[-1], x.dtype, x.device
    assert num_clusters < x.shape[0], 'Number of clusters must be less than number of points'
    indices = torch.randperm(x.shape[0], device=device)[:num_clusters]
    means = x[indices]

    for _ in range(num_iters):
        dists = (x @ means.t()) if use_cosine_sim else -cdist_sq(x, means)

        buckets = torch.argmax(dists, dim = -1)
        bins = torch.bincount(buckets, minlength=num_clusters)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means = new_means.scatter_add_(0, buckets.unsqueeze(-1).expand(-1, dim), x)
        new_means = new_means / bins_min_clamped.unsqueeze(-1)

        if use_cosine_sim:
            new_means = F.normalize(new_means, dim=-1, eps=1e-16) # L2 归一化

        means = torch.where(
            zero_mask.unsqueeze(-1),
            means,
            new_means
        )

    return means, bins

class VectorQuantization(nn.Module):
    """矢量量量化模块，采用EMA更新码本，编码和量化矢量之间距离为L2距离

    Args:
        dim (int): 输入特征的维度
        codebook_size (int): 码本大小
        codebook_dim (int): 码本中每个向量的维度
        decay (float): EMA的衰减系数
        epsilon (float): 防止除零错误的小常数
        kmeans_init (bool): 是否使用K-means初始化码本
        kmeans_iters (int): K-means初始化的迭代次数（仅在kmeans_init为True时使用）
        threshold_ema_dead_code (int): 码本向量被视为“死”的阈值
        tolerance_for_calc_threshold (float | None): 采用“容忍度”计算上述阈值，为 None 时，直接采用上述阈值
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: int = 2,
        tolerance_for_calc_threshold: float | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.epsilon = epsilon
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.tolerance_for_calc_threshold = tolerance_for_calc_threshold

        # codebook_dim != dim时，使用线性层调整维度
        self.project_in = nn.Linear(dim, codebook_dim) if codebook_dim != dim else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if codebook_dim != dim else nn.Identity()

        self.codebook = self.codebook_init()

    def codebook_init(self):
        """定义码本"""
        if self.kmeans_init:
            codebook = torch.zeros(self.codebook_size, self.codebook_dim)
            self.register_buffer("cluster_size", torch.zeros(self.codebook_size))
        else:
            codebook = torch.randn(self.codebook_size, self.codebook_dim)
            self.register_buffer("cluster_size", torch.ones(self.codebook_size))
        self.register_buffer("embed", codebook)
        self.register_buffer("embed_sum", codebook.clone())
        self.register_buffer("inited", torch.Tensor([False]))
        return codebook
        
    def init_embed(self, x: torch.Tensor):
        """初始化码本"""
        if self.inited:
            return
        self.avg_cluster_size = x.shape[0] / self.codebook_size
        if self.tolerance_for_calc_threshold:
            self.threshold_ema_dead_code = self.decay ** self.tolerance_for_calc_threshold * self.avg_cluster_size
        if self.kmeans_init:
            print("Using K-means to initialize codebook")
            means, cluster_size = kmeans(
                x,
                self.codebook_size,
                self.kmeans_iters,
                use_cosine_sim = False,
            )
            self.embed.data.copy_(means)
            self.cluster_size.data.copy_(cluster_size)
            self.embed_sum.data.copy_(means*cluster_size.unsqueeze(-1))
            self.inited = torch.Tensor([True])

    def ema_inplace(self, old, new):
        old.data.mul_(self.decay).add_(new.data, alpha=(1 - self.decay))

    def update_embed(self):
        embed_normalized = self.embed_sum / self.cluster_size.clamp(min=self.epsilon).unsqueeze(-1)
        self.embed.data.copy_(embed_normalized)

    def handle_dead_code(self, x_flat):
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        num_expired = expired_codes.sum().item()
        if num_expired > 0:
            print(f"\nReinitializing {num_expired} dead codebook vectors")
            new_vecs = x_flat[torch.randperm(x_flat.shape[0], device=x_flat.device)[:num_expired]]
            new_vec_cnt = new_vecs.shape[0]
            expired_idx = torch.nonzero(expired_codes, as_tuple=False).squeeze(1)[:new_vec_cnt]
            with torch.no_grad():
                self.embed.index_copy_(0, expired_idx, new_vecs)
                self.cluster_size.index_fill_(0, expired_idx, self.avg_cluster_size)
                self.embed_sum.index_copy_(0, expired_idx, new_vecs * self.avg_cluster_size)

    def codebook_forward(self, x: torch.Tensor):
        """码本前向传播，计算量化结果和相关损失"""
        shape = x.shape
        x_flat = x.reshape(-1, self.codebook_dim)  # (batch_size * num_vectors, codebook_dim)
        self.init_embed(x_flat) # 仅在kmeans_init为True且未初始化时调用
        # 计算距离（的平方）
        dists = cdist_sq(x_flat, self.embed) # (batch_size * num_vectors, codebook_size)
        embed_ind = torch.argmin(dists, dim=-1) # (batch_size * num_vectors,)
        cluster_size = torch.bincount(embed_ind, minlength=self.codebook_size) # (codebook_size,)

        if self.training:
            # 进行移动指数平均
            self.ema_inplace(self.cluster_size, cluster_size)
            embed_sum = torch.zeros_like(self.embed)
            embed_sum = embed_sum.scatter_add_(0, embed_ind.unsqueeze(-1).expand(-1, self.codebook_dim), x_flat)
            self.ema_inplace(self.embed_sum, embed_sum)
            self.update_embed()
            # 处理“死码本”问题
            self.handle_dead_code(x_flat)
        
        embed_ind = embed_ind.view(*shape[:-1]) # (batch_size, num_vectors)
        # 量化
        quantize = F.embedding(embed_ind, self.embed) # (batch_size, num_vectors, codebook_dim)

        return quantize, embed_ind


    def forward(self, x: torch.Tensor):
        """前向传播

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, num_vectors, dim)

        Returns:
            quantize (torch.Tensor): 量化后的张量，形状与输入相同
            embed_ind (torch.Tensor): 码本索引，形状为 (batch_size, num_vectors)
            commit_loss (torch.Tensor): 承诺损失
            quant_loss (torch.Tensor): 量化损失
        """
        x = self.project_in(x) # (batch_size, num_vectors, codebook_dim)
        q ,embed_ind = self.codebook_forward(x)
        loss = torch.zeros(1, device=x.device, requires_grad=False)

        if self.training:
            q = x + (q - x).detach()  # Straight Through Estimator
            commit_loss = F.mse_loss(q.detach(), x)
            loss = commit_loss

        q = self.project_out(q) # (batch_size, num_vectors, dim)

        return q, embed_ind, loss

if __name__ == "__main__":
    # Simple test code
    torch.manual_seed(0)
    vq = VectorQuantization(dim=4, codebook_size=64, codebook_dim=4, kmeans_init=False,kmeans_iters=100,threshold_ema_dead_code=0.9)
    for _ in range(200):
        x = torch.randn(2, 1, 4)
        quantize, embed_ind, loss = vq(x)
        print("Input shape:", x.shape)
        print("Quantized shape:", quantize.shape)
        print("Indices shape:", embed_ind.shape)
        print("Loss:", loss.item())