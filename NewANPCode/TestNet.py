import torch
from torch import nn
from torch.nn import functional as F
from ResNet import resnet34 as ResNet
import numpy as np

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class FeatureExtractor(nn.Module):
    """图像特征提取网络"""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv_net = ResNet()

    def forward(self, image_left, image_right):
        # 分别提取左、右耳特征
        img_feat_left = self.conv_net(image_left)  # [batch, 256]
        img_feat_right = self.conv_net(image_right)  # [batch, 256]

        # 拼接图像特征
        img_feat = torch.cat([img_feat_left, img_feat_right], dim=1)  # [batch, 512]
        return self.imgfc(img_feat)  # [batch, 256]


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
        mu = self.mlp(decoder_input)
        return mu

class ANP(nn.Module):
    def __init__(self, feature_extractor, num_heads, output_num, dim_k, dim_v, dim_x, dim_y, encoder_sizes, decoder_sizes, target_num=100):
        """
        初始化ANP模型。

        参数:
            feature_extractor: 预初始化的特征提取器模块
            num_heads: 注意力头数量
            output_num: 注意力聚合器的输出维度 (通常等于 dim_v)
            dim_k: 注意力机制中键/查询的维度
            dim_v: 注意力机制中值的维度 (也是编码器输出维度)
            dim_x: 输入特征的维度 (图像特征 + 位置)
            dim_y: 输出特征的维度 (HRTF)
            encoder_sizes: 编码器隐藏层大小列表 (最后一层大小应为 dim_v)
            decoder_sizes: 解码器隐藏层大小列表
            target_num: 训练时用作目标的点数
        """
        super(ANP, self).__init__()
        self.feature_extractor = feature_extractor
        self.encoder = Encoder(dim_x + dim_y, encoder_sizes)
        self.attention_aggregator = AttentionAggregator(num_heads, output_num, dim_k, dim_v, dim_x)
        self.decoder = Decoder(output_num, dim_x, dim_y, decoder_sizes)
        self.target_num = target_num
        self.dim_x = dim_x
        self.dim_y = dim_y

    def _prepare_features_target(self, left_image, right_image, pos, hrtf, device, is_training):
        """ 内部函数：准备特征和目标张量 """
        left_image = left_image.to(device)
        right_image = right_image.to(device)
        image_feature = self.feature_extractor(left_image, right_image)

        pos = pos.to(device)
        hrtf = hrtf.to(device)

        if is_training:
            num_positions = pos.shape[1]
            image_feature_repeated = image_feature.unsqueeze(1).repeat(1, num_positions, 1)
            features = torch.cat([image_feature_repeated, pos], dim=2)
            features = features.reshape(-1, features.shape[-1])
            target = hrtf.reshape(-1, hrtf.shape[-1])
        else:
            features = torch.cat([image_feature, pos], dim=1)
            target = hrtf

        expected_feature_dim = self.dim_x
        expected_target_dim = self.dim_y
        if expected_feature_dim is not None and features.shape[-1] != expected_feature_dim:
             raise ValueError(f"Feature dimension mismatch: expected {expected_feature_dim}, got {features.shape[-1]}")
        if expected_target_dim is not None and target.shape[-1] != expected_target_dim:
            raise ValueError(f"Target dimension mismatch: expected {expected_target_dim}, got {target.shape[-1]}")

        return features, target

    def forward(self, left_image, right_image, pos, hrtf, device, is_training=True, auxiliary_data=None):
        if is_training:
            features, target = self._prepare_features_target(left_image, right_image, pos, hrtf, device, is_training=True)
            num_total_points = features.shape[0]
            if num_total_points <= self.target_num:
                print(f"Warning: Not enough points ({num_total_points}) for target_num ({self.target_num}). Using all as target.")
                target_x = features
                target_y_for_loss = target
                if num_total_points > 0:
                   context_x = features[0:1]
                   context_y = target[0:1]
                else:
                   return torch.zeros(num_total_points, self.dim_y, device=device), torch.zeros(num_total_points, self.dim_y, device=device)
            else:
                indices = np.random.permutation(num_total_points)
                target_indices = indices[:self.target_num]
                context_indices = indices[self.target_num:]

                target_x = features[target_indices]
                target_y_for_loss = target[target_indices]
                context_x = features[context_indices]
                context_y = target[context_indices]

            target_x = target_x.unsqueeze(0)
            context_x = context_x.unsqueeze(0)
            context_y = context_y.unsqueeze(0)
        else:
            if auxiliary_data is None:
                raise ValueError("Auxiliary data must be provided during evaluation.")

            target_x, target_y_for_loss = self._prepare_features_target(left_image, right_image, pos, hrtf, device, is_training=False)
            aux_left = auxiliary_data["left_image"]
            aux_right = auxiliary_data["right_image"]
            aux_pos = auxiliary_data["position"]
            aux_hrtf = auxiliary_data["hrtf"]

            context_x, context_y = self._prepare_features_target(aux_left, aux_right, aux_pos, aux_hrtf, device, is_training=True)
            target_x = target_x.unsqueeze(0)
            context_x = context_x.unsqueeze(0)
            context_y = context_y.unsqueeze(0)

        batch_r = self.encoder(context_x, context_y) # (batch_size, context_num, 64)
        r = self.attention_aggregator(target_x, context_x, batch_r)
        mu = self.decoder(r, target_x)
        mu_squeezed = mu.squeeze(0)

        return mu_squeezed, target_y_for_loss

class TestNet(nn.Module):
    """完整网络，集成特征提取和ANP预测"""
    def __init__(self, target_num_anp=100):
        super(TestNet, self).__init__()
        self.feature_extractor = FeatureExtractor()
        img_feature_dim = 256
        pos_dim = 2
        hrtf_dim = 108

        dim_x = img_feature_dim + pos_dim
        dim_y = hrtf_dim

        dim_v = 256
        output_num = 256
        dim_k = 256

        self.anp = ANP(
            feature_extractor=self.feature_extractor,
            num_heads=4,
            output_num=output_num,
            dim_k=dim_k,
            dim_v=dim_v,
            dim_x=dim_x,
            dim_y=dim_y,
            encoder_sizes=[512, 256, dim_v],
            decoder_sizes=[512, 256],
            target_num=target_num_anp
        )

    def forward(self, left_image, right_image, pos, hrtf, device, is_training=True, auxiliary_data=None):
        return self.anp(left_image, right_image, pos, hrtf, device, is_training, auxiliary_data)