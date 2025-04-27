import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # 不要注释掉，否则不能使用3D绘图

# 构建一个二维高斯函数，作为网络将要拟合的对象
class Gaussian2D(nn.Module):
    def __init__(self, mean, cov):
        super(Gaussian2D, self).__init__()
        self.mean = mean
        self.cov = cov

    def forward(self, x):
        dist = distributions.MultivariateNormal(self.mean, self.cov)
        return torch.exp(dist.log_prob(x))
    
class DataGenerator():
    def __init__(self, batch_size, mean, cov, max_samples=200):
        self.mean = mean
        self.cov = cov
        self.max_samples = max_samples
        self.gaussian = Gaussian2D(mean, cov)
        self.batch_size = batch_size
        self.train_positions, self.test_positions = self.get_positions()

    def get_train_samples(self):
        # 生成训练样本
        # train_batch_num = torch.randint(10, self.max_samples+1, (1,)).item()
        train_batch_num = self.max_samples
        # train_target_num = torch.randint(4, train_batch_num//2, (1,)).item()
        train_target_num = train_batch_num // 2
        batch_indices = np.zeros((self.batch_size, train_batch_num), dtype=int)
        for i in range(self.batch_size):
            batch_indices[i] = np.random.choice(self.train_positions.size(0), train_batch_num, replace=False)
        
        target_indices = batch_indices[:, :train_target_num]
        context_indices = batch_indices[:, train_target_num:]
        target_x = self.train_positions[target_indices]
        target_y = self.gaussian(target_x).unsqueeze(-1)
        context_x = self.train_positions[context_indices]
        context_y = self.gaussian(context_x).unsqueeze(-1)

        return (target_x, target_y), (context_x, context_y)
    
    def get_test_samples(self, context_num=1000, target_num=125):
        # 生成测试样本
        test_target_num = target_num
        test_context_num = context_num

        target_indices = np.zeros((self.batch_size, test_target_num), dtype=int)
        context_indices = np.zeros((self.batch_size, test_context_num), dtype=int)
        for i in range(self.batch_size):
            target_indices[i] = np.random.choice(self.test_positions.size(0), test_target_num, replace=False)
            context_indices[i] = np.random.choice(self.train_positions.size(0), test_context_num, replace=False)
            
        target_x = self.test_positions[target_indices]
        target_y = self.gaussian(target_x).unsqueeze(-1)
        context_x = self.train_positions[context_indices]
        context_y = self.gaussian(context_x).unsqueeze(-1)

        return (target_x, target_y), (context_x, context_y)
    
    def get_positions(self):
        x = torch.linspace(-4, 4, 100)
        y = torch.linspace(-4, 4, 100)
        X, Y = torch.meshgrid(x, y)
        pos = torch.stack([X.flatten(), Y.flatten()], dim=-1)
        # 生成随机索引，概率为0.3
        train_indices = sorted(np.random.choice(pos.size(0), size=int(0.3 * pos.size(0)), replace=False))
        test_indices = [i for i in range(pos.size(0)) if i not in train_indices]
        return (pos[train_indices], pos[test_indices])

def batch_mlp(input_dim, hidden_sizes):
    """创建一个多层感知机，且最后一层不使用激活函数"""
    layers = []
    prev_size = input_dim
    for size in hidden_sizes[:-1]:
        layers.append(nn.Linear(prev_size, size))
        layers.append(nn.ReLU())
        prev_size = size
    layers.append(nn.Linear(prev_size, hidden_sizes[-1]))
    return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim: list):
        super(Encoder, self).__init__()
        self.mlp = batch_mlp(input_dim, hidden_dim)

    def forward(self, context_x, context_y):
        encoder_input = torch.cat([context_x, context_y], dim=-1)
        r = self.mlp(encoder_input)
        return r

class Attention(nn.Module):
    """
    注意力机制实现。但是调换了输入变量的维度顺序。
    同时增加batchsize维度。
    """
    
    def __init__(self):
        """
        初始化。
        """
        super(Attention, self).__init__()
    
    def forward(self, query, key, value):
        '''
        前向传播函数。
        
        参数:
            query: 查询向量，形状为 (batch_size, query_num, dim_k)
            key: 键向量，形状为 (batch_size, pair_num, dim_k)
            value: 值向量，形状为 (batch_size, pair_num, dim_v)
        '''
        # 计算注意力权重
        scores = torch.bmm(query, key.transpose(1, 2)) / (key.size(-1) ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.bmm(attn_weights, value)

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制实现。采用并行计算的方式。
    """
    
    def __init__(self, num_heads, output_num, dim_k, dim_v):
        """
        初始化多头注意力机制。
        
        参数:
            num_heads: 注意力头数量
            output_num: 输出维度
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.output_num = output_num
        self.attention = Attention()
        self.hidden_dim = output_num // num_heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.Wq = nn.Linear(self.dim_k, self.output_num, bias=False)
        self.Wk = nn.Linear(self.dim_k, self.output_num, bias=False)    
        self.Wv = nn.Linear(self.dim_v, self.output_num, bias=False)
        self.Wo = nn.Linear(self.output_num, self.output_num, bias=False)
        
    def qkv_transpose(self, X, hidden_dim):
        '''
        通过将输入X的维度进行变换，来实现并行计算多头注意力机制。       

        输入:
            X: 输入张量，形状为 (batch_size, any_num, num_heads * hidden_dim) 
        输出:
            X: 变换后的张量，形状为 (batch_size*num_heads, any_num, hidden_dim)
        '''
        X = X.reshape(X.size(0), X.size(1), self.num_heads, hidden_dim)  # (batch_size, any_num, num_heads, hidden_dim)
        X = X.transpose(1, 2)  # (batch_size, num_heads, any_num, hidden_dim)
        X = X.reshape(-1, X.size(2), hidden_dim)
        # (batch_size*num_heads, any_num, hidden_dim)
        return X
    def qkv_itranspose(self, X, hidden_dim):
        '''
        通过将输入X的维度进行变换，来实现并行计算多头注意力机制。   
        输入:
            X: 输入张量，形状为 (batch_size*num_heads, any_num, hidden_dim) 
        输出:
            X: 变换后的张量，形状为 (batch_size, any_num, num_heads * hidden_dim)
        '''
        X = X.reshape(-1, self.num_heads, X.size(1), hidden_dim)  # (batch_size, num_heads, any_num, hidden_dim)
        X = X.transpose(1, 2)  # (batch_size, any_num, num_heads, hidden_dim)
        X = X.reshape(X.size(0), X.size(1), -1)  # (batch_size, any_num, num_heads * hidden_dim)
        return X

    def forward(self, query, key, value):
        '''        
        参数:
            query: 查询向量，形状为 (batch_size, query_num, dim_k)
            key: 键向量，形状为 (batch_size, pair_num, dim_k)
            value: 值向量，形状为 (batch_size, pair_num, dim_v)
        '''
        query = self.qkv_transpose(self.Wq(query), self.hidden_dim)
        key = self.qkv_transpose(self.Wk(key), self.hidden_dim)
        value = self.qkv_transpose(self.Wv(value), self.hidden_dim)
        output = self.attention(query, key, value)
        output = self.qkv_itranspose(output, self.hidden_dim)
        return output
    

class AttentionAggregator(nn.Module):
    def __init__(self, num_heads, output_num, dim_k, dim_v, dim_x):
        super(AttentionAggregator, self).__init__()
        self.attention = MultiHeadAttention(num_heads, output_num, dim_k, dim_v)
        self.mlp_key = batch_mlp(dim_x, [dim_k, dim_k])
        self.mlp_query = batch_mlp(dim_x, [dim_k, dim_k])
        
    def forward(self, query, key, value):
        # 这里query是target_x，key是上下文context_x，value是上下文特征r
        # value: (batch_size, context_num, dim_r)
        query = self.mlp_query(query) # (batch_size, target_num, dim_k)
        key = self.mlp_key(key) # (batch_size, context_num, dim_k)
        attention_output = self.attention(query, key, value) # (batch_size, target_num, output_num)
        return attention_output

class Decoder(nn.Module):
    def __init__(self, dim_r, dim_x, dim_y, hidden_dim):
        super(Decoder, self).__init__()
        self.mlp = batch_mlp(dim_r + dim_x, hidden_dim + [dim_y])

    def forward(self, r, target_x):
        decoder_input = torch.cat([r, target_x], dim=-1)
        # decoder_output = self.mlp(decoder_input)
        # mu, log_var = decoder_output.split(decoder_output.size(-1) // 2, dim=-1)
        mu = self.mlp(decoder_input)
        return mu

class ANP(nn.Module):
    def __init__(self, num_heads, output_num, dim_k, dim_v, dim_x, dim_y, encoder_sizes, decoder_sizes):
        super(ANP, self).__init__()
        self.encoder = Encoder(dim_x + dim_y, encoder_sizes)
        self.attention_aggregator = AttentionAggregator(num_heads, output_num, dim_k, dim_v, dim_x)
        self.decoder = Decoder(output_num, dim_x, dim_y, decoder_sizes)

    def forward(self, context_x, context_y, target_x):
        # context_x: 上下文输入，形状为 (batch_size, context_num, 2)
        # context_y: 上下文输出，形状为 (batch_size, context_num, 1)
        # target_x: 目标输入，形状为 (batch_size, target_num, 2)
        batch_r = self.encoder(context_x, context_y) # (batch_size, context_num, 64)
        r = self.attention_aggregator(target_x, context_x, batch_r)
        mu = self.decoder(r, target_x)
        return mu
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dim_v = 32
encoder_sizes = [64, dim_v]
decoder_sizes = [64, 64]
model = ANP(num_heads=4, output_num=64, 
            dim_k=32, dim_v=dim_v, dim_x=2, 
            dim_y=1, encoder_sizes=encoder_sizes, 
            decoder_sizes=decoder_sizes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
train_data_generator = DataGenerator(batch_size=32, 
                                     mean=torch.tensor([0.0, 0.0]), 
                                     cov=torch.tensor([[1.0, 0.5], [0.5, 1.0]]),
                                     max_samples=1000)

# 绘制二维高斯函数的三维曲面
def plot_gaussian_surface(mean, cov, model):
    x = torch.linspace(-4, 4, 100)
    y = torch.linspace(-4, 4, 100)
    X, Y = torch.meshgrid(x, y)
    pos = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    
    gaussian = Gaussian2D(mean.to(device), cov.to(device))  # 添加 to(device)
    train_data_generator = DataGenerator(batch_size=1, mean=mean, cov=cov)
    (target_x, target_y), (context_x, context_y) = train_data_generator.get_test_samples()
    
    # 将输入移至正确的设备
    target_x = target_x.to(device)
    target_y = target_y.to(device)
    context_x = context_x.to(device)
    context_y = context_y.to(device)
    
    Z = gaussian(pos.to(device)).reshape(X.shape).cpu().detach().numpy()  # 添加 to(device) 和 cpu()
    with torch.no_grad():
        # mu, _ = model(context_x, context_y, target_x)
        mu = model(context_x, context_y, target_x)
        pred_target_y = mu.squeeze(0).cpu().numpy()
    
    # 确保绘图时所有张量都在 CPU 上
    target_x = target_x.squeeze(0).cpu()
    target_y = target_y.squeeze(0).cpu()
    context_x = context_x.squeeze(0).cpu()
    context_y = context_y.squeeze(0).cpu()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X.numpy(), Y.numpy(), Z, cmap='PuBuGn', alpha=0.5)
    ax.scatter(target_x[:, 0].numpy(), target_x[:, 1].numpy(), target_y.numpy(), color='r', label='Actual Target Samples')
    ax.scatter(target_x[:, 0].numpy(), target_x[:, 1].numpy(), pred_target_y, color='g', label='Predicted Target Samples')
    ax.scatter(context_x[:, 0].numpy(), context_x[:, 1].numpy(), context_y.numpy(), color='b', label='Context Samples')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Probability Density')
    plt.legend()
    plt.show()

# plot_gaussian_surface(torch.tensor([0.0, 0.0]), torch.tensor([[1.0, 0.5], [0.5, 1.0]]))
criterion = nn.MSELoss()

def get_loss(data_generator, model, criterion, context_num, target_num):
    (target_x, target_y), (context_x, context_y) = data_generator.get_test_samples(context_num=context_num, target_num=target_num)
    target_x = target_x.to(device)
    target_y = target_y.to(device)
    context_x = context_x.to(device)
    context_y = context_y.to(device)
    mu = model(context_x, context_y, target_x)
    loss = criterion(mu, target_y)
    return loss.item()

# 训练模型
for epoch in range(6001):
    model.train()
    optimizer.zero_grad()
    (target_x, target_y), (context_x, context_y) = train_data_generator.get_train_samples()
    target_x = target_x.to(device)
    target_y = target_y.to(device)
    context_x = context_x.to(device)
    context_y = context_y.to(device)
    mu = model(context_x, context_y, target_x)
    # dist = distributions.Normal(mu, torch.exp(log_var / 2))
    # loss = -dist.log_prob(target_y).mean()
    loss = criterion(mu, target_y)
    loss.backward()
    optimizer.step()
    if epoch % 600 == 0:
        print(f'Epoch {epoch}, Train Loss: {loss.item():.4e}')
        with torch.no_grad():
            # 生成测试样本
            loss = get_loss(train_data_generator, model, criterion, context_num=3000, target_num=125)
            print(f'Epoch {epoch}, Test Loss(Ultimate High Context Num): {loss:.4e}')
            loss = get_loss(train_data_generator, model, criterion, context_num=1000, target_num=125)
            print(f'Epoch {epoch}, Test Loss(High Context Num): {loss:.4e}')
            loss = get_loss(train_data_generator, model, criterion, context_num=100, target_num=125)
            print(f'Epoch {epoch}, Test Loss(Low Context Num): {loss:.4e}')
            loss = get_loss(train_data_generator, model, criterion, context_num=10, target_num=125)
            print(f'Epoch {epoch}, Test Loss(Low Context Num): {loss:.4e}')
            # 绘制训练样本和拟合的高斯函数
        plot_gaussian_surface(torch.tensor([0.0, 0.0]), 
                              torch.tensor([[1.0, 0.5], [0.5, 1.0]]), model)
