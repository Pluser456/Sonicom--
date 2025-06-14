from re import T
import torch
from torch import nn
from ResNet3D import resnet34_3d as resnet3d
from ResNet import resnet34 as resnet2d
import numpy as np
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """图像特征提取网络"""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv_net = resnet3d()

        self.imgfc = nn.Sequential(
            nn.Linear(2000, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )


    def forward(self, voxel_right):
        img_feat_right = self.conv_net(voxel_right)  # [batch, 256]

        return img_feat_right  # [batch, 256]

class FeatureExtractor2D(nn.Module):
    """图像特征提取网络"""
    def __init__(self):
        super(FeatureExtractor2D, self).__init__()
        self.conv_net = resnet2d()
        
    def forward(self, voxel_right):
        # 提取右耳特征
        img_feat_right = self.conv_net(voxel_right)  # [batch, 256]
        return img_feat_right  # [batch, 256]


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

class AttentionAggregator(nn.Module):
    def __init__(self, num_heads, output_num, dim_k, dim_v, dim_x):
        """
        初始化注意力聚合器。
        参数:
            num_heads: 注意力头的数量
            output_num: 注意力机制的输出维度 (也是 nn.MultiheadAttention 的 embed_dim)
            dim_k: 经过 MLP 处理后的 query 和 key 的维度 (也是 nn.MultiheadAttention 的 kdim)
            dim_v: 输入 value (即 context_r) 的维度 (也是 nn.MultiheadAttention 的 vdim)
            dim_x: 输入到 MLP 以生成 query 和 key 的特征维度
        """
        super(AttentionAggregator, self).__init__()
        self.mlp_key = batch_mlp(dim_x, [dim_k, dim_k])
        self.mlp_query = batch_mlp(dim_x, [dim_k, dim_k])
        
        # 使用 PyTorch 内置的多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=output_num, 
            num_heads=num_heads,
            kdim=dim_k,
            vdim=dim_v,
            batch_first=True  # 重要：确保输入输出的 batch 维度在第一位
        )
        
    def forward(self, target_x_features, context_x_features, context_r_features):
        """
        前向传播。
        参数:
            target_x_features: 用于生成 query 的目标点特征，形状 (batch_size, target_num, dim_x)
            context_x_features: 用于生成 key 的上下文点特征，形状 (batch_size, context_num, dim_x)
            context_r_features: 上下文点的表示 (value)，形状 (batch_size, context_num, dim_v)
        """
        # query_for_attention: (batch_size, target_num, dim_k)
        query_for_attention = self.mlp_query(target_x_features) 
        # key_for_attention: (batch_size, context_num, dim_k)
        key_for_attention = self.mlp_key(context_x_features)   
        # value_for_attention: (batch_size, context_num, dim_v)
        value_for_attention = context_r_features              

        # nn.MultiheadAttention 的输入顺序是 query, key, value
        # 输出是 (attn_output, attn_output_weights)
        # attn_output 形状: (batch_size, target_num, embed_dim) 
        # (在此处 embed_dim 等于初始化时的 output_num)
        attention_output, _ = self.attention(query_for_attention, key_for_attention, value_for_attention)
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
    def __init__(self, num_heads, output_num, dim_k, dim_v, dim_x, dim_y, encoder_sizes, decoder_sizes, target_num=100, positions_num=100):
        """
        初始化ANP模型。

        参数:
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
        # self.feature_extractor = feature_extractor
        self.encoder = Encoder(dim_x + dim_y, encoder_sizes)
        self.attention_aggregator = AttentionAggregator(num_heads, output_num, dim_k, dim_v, dim_x)
        self.decoder = Decoder(output_num, dim_x, dim_y, decoder_sizes)
        self.target_num = target_num
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.positions_num = positions_num
        
        # 添加缓存上下文的属性
        self.cached_context_x = None
        self.cached_context_y = None
        self.cached_context_r = None

        self.is_training = False

    def _prepare_features_target(self, feature_extractor, left_voxel, right_voxel, pos, hrtf, device, is_training):
        """ 内部函数：准备特征和目标张量 """
        max_chunk_batch_size = 40  # 设置最大批次大小限制
        if left_voxel.shape[0] > max_chunk_batch_size:
            voxel_feature_chunks = []
            # 将左右体素数据按max_chunk_batch_size分割成小批次
            left_voxel_chunks = torch.split(left_voxel, max_chunk_batch_size, dim=0)
            right_voxel_chunks = torch.split(right_voxel, max_chunk_batch_size, dim=0)

            for lv_chunk, rv_chunk in zip(left_voxel_chunks, right_voxel_chunks):
                # 对每个小批次提取特征
                lv_chunk = lv_chunk.to(device)
                rv_chunk = rv_chunk.to(device)
                vf_chunk = feature_extractor(lv_chunk, rv_chunk)
                voxel_feature_chunks.append(vf_chunk)
            
            # 合并所有小批次的特征提取结果
            voxel_feature = torch.cat(voxel_feature_chunks, dim=0)
        else:
            left_voxel = left_voxel.to(device)
            right_voxel = right_voxel.to(device)
            # 如果批次大小未超过限制，则直接提取特征
            voxel_feature = feature_extractor(left_voxel, right_voxel)

        
        # 释放不再需要的变量
        del left_voxel, right_voxel
        torch.cuda.empty_cache()  # 清理未使用的缓存

        pos = pos.to(device)
        hrtf = hrtf.to(device)
        
        num_positions = pos.shape[1]
        voxel_feature_repeated = voxel_feature.unsqueeze(1).repeat(1, num_positions, 1)
        features = torch.cat([voxel_feature_repeated, pos], dim=2)
        features = features.reshape(-1, features.shape[-1])
        target = hrtf.reshape(-1, hrtf.shape[-1])

        expected_feature_dim = self.dim_x
        expected_target_dim = self.dim_y
        if expected_feature_dim is not None and features.shape[-1] != expected_feature_dim:
             raise ValueError(f"Feature dimension mismatch: expected {expected_feature_dim}, got {features.shape[-1]}")
        if expected_target_dim is not None and target.shape[-1] != expected_target_dim:
            raise ValueError(f"Target dimension mismatch: expected {expected_target_dim}, got {target.shape[-1]}")

        return features, target

    def forward(self, feature_extractor, left_voxel, right_voxel, pos, hrtf, device, is_training=True, auxiliary_data=None):
        if is_training:
            features, target = self._prepare_features_target(feature_extractor, left_voxel, right_voxel, pos, hrtf, device, is_training=True)
        
            indices = np.random.permutation(self.positions_num)
            voxels_num = left_voxel.shape[0] # 体素个数
            # 选取目标点和上下文点，每个体素选取self.target_num个目标点，剩余的点作为上下文点
            target_indices = (indices[:self.target_num] + (np.arange(voxels_num) * self.positions_num).reshape(-1, 1)).flatten()
            context_indices = (indices[self.target_num:] + (np.arange(voxels_num) * self.positions_num).reshape(-1, 1)).flatten()

            target_x = features[target_indices]
            target_y_for_loss = target[target_indices]
            context_x = features[context_indices]
            context_y = target[context_indices]

            target_x = target_x.unsqueeze(0)
            context_x = context_x.unsqueeze(0)
            context_y = context_y.unsqueeze(0)
            # 计算编码器输出
            batch_r = self.encoder(context_x, context_y)
            if self.is_training != is_training:
                self.cached_context_x = context_x.detach().clone()
                self.cached_context_r = batch_r.detach().clone()
                self.is_training = is_training
            else:
                tmp_context_x = torch.cat([self.cached_context_x, context_x.detach().clone()], dim=1)
                tmp_context_r = torch.cat([self.cached_context_r, batch_r.detach().clone()], dim=1)
                # 使用缓存的数据，避免信息泄露
                context_x = self.cached_context_x
                batch_r = self.cached_context_r
                # 将缓存的上下文数据与当前上下文数据拼接
                self.cached_context_x = tmp_context_x
                self.cached_context_r = tmp_context_r

        else:
            # 处理非训练模式（评估/推理）
            target_x, target_y_for_loss = self._prepare_features_target(feature_extractor, left_voxel, right_voxel, pos, hrtf, device, is_training=False)
            target_x = target_x.unsqueeze(0)
            
            
            # 如果缓存不存在或辅助数据已更改，则重新计算上下文
            if self.is_training != is_training:
                
                aux_left = auxiliary_data["left_voxel"]
                aux_right = auxiliary_data["right_voxel"]
                aux_pos = auxiliary_data["position"]
                aux_hrtf = auxiliary_data["hrtf"]

                context_x, context_y = self._prepare_features_target(
                    feature_extractor, aux_left, aux_right, aux_pos, aux_hrtf, 
                    device, is_training=True
                )
                context_x = context_x.unsqueeze(0)
                context_y = context_y.unsqueeze(0)
                
                # 计算并缓存编码器输出
                batch_r = self.encoder(context_x, context_y)
                
                # 保存到缓存
                self.cached_context_x = context_x
                self.cached_context_y = context_y
                self.cached_context_r = batch_r
                self.is_training = is_training
            else:
                # 使用缓存的数据
                context_x = self.cached_context_x
                context_y = self.cached_context_y  # 虽然不直接使用，但保留以备将来可能需要
                batch_r = self.cached_context_r

        # 使用上下文(context)数据对目标(target)数据进行注意力聚合
        r = self.attention_aggregator(target_x, context_x, batch_r)
        mu = self.decoder(r, target_x)
        mu_squeezed = mu.squeeze(0)

        return mu_squeezed, target_y_for_loss

class TestNet(nn.Module):
    """完整网络，集成特征提取和ANP预测"""
    modelname = "3DResNetANP"
    def __init__(self, target_num_anp=100,positions_num=100):
        super(TestNet, self).__init__()
        self.feature_extractor = FeatureExtractor()
        img_feature_dim = 256
        pos_dim = 3
        hrtf_dim = 108

        dim_x = img_feature_dim + pos_dim
        dim_y = hrtf_dim

        dim_v = 256
        output_num = 256
        dim_k = 256

        self.anp = ANP(
            num_heads=8,
            output_num=output_num,
            dim_k=dim_k,
            dim_v=dim_v,
            dim_x=dim_x,
            dim_y=dim_y,
            encoder_sizes=[512, 256, dim_v],
            decoder_sizes=[512, 256],
            target_num=target_num_anp,
            positions_num=positions_num
        )

    def forward(self, left_voxel, right_voxel, pos, hrtf, device, is_training=True, auxiliary_data=None):
        return self.anp(self.feature_extractor, left_voxel, right_voxel, pos, hrtf, device, is_training, auxiliary_data)
    
class ResNet3D(nn.Module):
    """完整网络，集成特征提取和ANP预测"""
    modelname = "3DResNet"
    def __init__(self):
        super(ResNet3D, self).__init__()
        img_feature_dim = 256
        pos_dim = 3
        self.feature_extractor = FeatureExtractor()
        self.fc = batch_mlp(img_feature_dim + pos_dim, [512, 256, 256, 512, 256,108])

    def forward(self, left_voxel, right_voxel, pos, hrtf, device):
        max_chunk_batch_size = 40  # 设置最大批次大小限制
        if left_voxel.shape[0] > max_chunk_batch_size:
            voxel_feature_chunks = []
            # 将左右体素数据按max_chunk_batch_size分割成小批次
            left_voxel_chunks = torch.split(left_voxel, max_chunk_batch_size, dim=0)
            right_voxel_chunks = torch.split(right_voxel, max_chunk_batch_size, dim=0)

            for lv_chunk, rv_chunk in zip(left_voxel_chunks, right_voxel_chunks):
                # 对每个小批次提取特征
                lv_chunk = lv_chunk.to(device)
                rv_chunk = rv_chunk.to(device)
                vf_chunk = self.feature_extractor(lv_chunk, rv_chunk)
                voxel_feature_chunks.append(vf_chunk)
            
            # 合并所有小批次的特征提取结果
            voxel_feature = torch.cat(voxel_feature_chunks, dim=0)
        else:
            left_voxel = left_voxel.to(device)
            right_voxel = right_voxel.to(device)
            # 如果批次大小未超过限制，则直接提取特征
            voxel_feature = self.feature_extractor(left_voxel, right_voxel)

        
        # 释放不再需要的变量
        del left_voxel, right_voxel
        torch.cuda.empty_cache()  # 清理未使用的缓存
        pos = pos.to(device)
        hrtf = hrtf.to(device)
        
        num_positions = pos.shape[1]
        voxel_feature_repeated = voxel_feature.unsqueeze(1).repeat(1, num_positions, 1)
        features = torch.cat([voxel_feature_repeated, pos], dim=2)
        features = features.reshape(-1, features.shape[-1])
        target = hrtf.reshape(-1, hrtf.shape[-1])

        y_pred = self.fc(features)
        return y_pred, target
        
class ResNet2D(nn.Module):
    """完整网络，集成特征提取和ANP预测"""
    modelname = "2DResNet"
    def __init__(self):
        super(ResNet2D, self).__init__()
        img_feature_dim = 2000
        # pos_dim = 3
        self.feature_extractor = FeatureExtractor2D()
        # 第一个隐藏层: 使用 ResidualBlock (不带 BatchNorm，因为通常第一层后不加)
        mlp_layers = []
        mlp_hidden_dims = [512, 256, 256, 256]
        current_dim = img_feature_dim
        first_hidden_dim = mlp_hidden_dims[0]
        mlp_layers.append(ResidualBlock(current_dim, first_hidden_dim, use_batchnorm=False))
        current_dim = first_hidden_dim

        # 后续的隐藏层: 使用 ResidualBlock (带 BatchNorm)
        for i in range(1, len(mlp_hidden_dims)):
            current_hidden_dim = mlp_hidden_dims[i]
            mlp_layers.append(ResidualBlock(current_dim, current_hidden_dim, use_batchnorm=True))
            current_dim = current_hidden_dim
        
        # MLP 的输出层 (不使用残差块，直接线性输出)
        mlp_layers.append(nn.Linear(current_dim, 256))
        
        self.fc =  nn.Sequential(*mlp_layers)

    def forward(self, left_voxel, right_voxel, feature, device):
        max_chunk_batch_size = 40  # 设置最大批次大小限制
        if left_voxel.shape[0] > max_chunk_batch_size:
            voxel_feature_chunks = []
            # 将左右体素数据按max_chunk_batch_size分割成小批次
            left_voxel_chunks = torch.split(left_voxel, max_chunk_batch_size, dim=0)
            right_voxel_chunks = torch.split(right_voxel, max_chunk_batch_size, dim=0)

            for lv_chunk, rv_chunk in zip(left_voxel_chunks, right_voxel_chunks):
                # 对每个小批次提取特征
                lv_chunk = lv_chunk.to(device)
                rv_chunk = rv_chunk.to(device)
                vf_chunk = self.feature_extractor(lv_chunk, rv_chunk)
                voxel_feature_chunks.append(vf_chunk)
            
            # 合并所有小批次的特征提取结果
            voxel_feature = torch.cat(voxel_feature_chunks, dim=0)
        else:
            left_voxel = left_voxel.to(device)
            right_voxel = right_voxel.to(device)
            # 如果批次大小未超过限制，则直接提取特征
            voxel_feature = self.feature_extractor(left_voxel, right_voxel)

        
        # 释放不再需要的变量
        del left_voxel, right_voxel
        torch.cuda.empty_cache()  # 清理未使用的缓存

        feature = feature.to(device)
        target = feature
        y_pred = self.fc(voxel_feature)
        return y_pred, target
    
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

class ResNet2DClassifier(nn.Module):
    """修改为多头分类网络，每个位置使用独立的分类头"""
    modelname = "2DResNetClassifier"
    def __init__(self, num_classes=128):
        super(ResNet2DClassifier, self).__init__()
        self.feature_extractor = resnet2d(num_classes=num_classes)

    def forward(self, right_voxel, device):
        # 体素特征提取
        right_voxel = right_voxel.to(device)
        logits = self.feature_extractor(right_voxel) # [batch_size, img_feature_dim]

        # 释放内存
        del right_voxel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 获取预测类别
        predictions = torch.argmax(logits, dim=1)  # [batch_size]

        return predictions, logits
    
class ResNet3DClassifier(nn.Module):
    """修改为多头分类网络，每个位置使用独立的分类头，基于3D特征提取器"""
    modelname = "3DResNetClassifier"
    def __init__(self, num_classes=256):
        super(ResNet3DClassifier, self).__init__()
        self.feature_extractor = resnet3d(num_classes=num_classes)

    def forward(self, right_voxel, device):
        # 体素特征提取
        right_voxel = right_voxel.to(device)
        logits = self.feature_extractor(right_voxel) # [batch_size, img_feature_dim]

        # 释放内存
        del right_voxel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 获取预测类别
        predictions = torch.argmax(logits, dim=1)  # [batch_size]

        return predictions, logits