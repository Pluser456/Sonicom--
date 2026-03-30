import torch
from torch import nn
from .ResNet3D import resnet34_3d
from .ResNet3D import resnet18_3d
from .ResNet import resnet34
from .ResNet import resnet18

class FeatureExtractor(nn.Module):
    """图像特征提取网络"""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv_net = resnet34_3d()

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
        self.conv_net = resnet18()
        
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
    def __init__(self, num_classes=16, encoder_out_vec_num=8):
        super(ResNet2DClassifier, self).__init__()
        self.feature_extractor = resnet18(num_classes=1024)
        self.encoder_out_vec_num = encoder_out_vec_num
        self.num_classes = num_classes
        self.vq_layer = nn.ModuleList([ nn.Sequential(nn.Linear(1024, num_classes),
            # nn.Dropout(0.3),
            nn.ReLU(), nn.Linear(num_classes, num_classes)) for _ in range(encoder_out_vec_num)])

    def forward(self, right_voxel, device):
        # 体素特征提取
        right_voxel = right_voxel.to(device)
        features = self.feature_extractor(right_voxel) # [batch_size, img_feature_dim]
        logits = torch.zeros((features.shape[0], self.num_classes, self.encoder_out_vec_num)).to(device)  # [batch_size, num_classes, encoder_out_vec_num]
        for i in range(self.encoder_out_vec_num):
            logits[:, :, i] = self.vq_layer[i](features)

        # 释放内存
        del right_voxel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 获取预测类别
        predictions = torch.argmax(logits, dim=1)  # [batch_size, encoder_out_vec_num]

        return predictions, logits
    
class ResNet3DClassifier(nn.Module):
    """修改为多头分类网络，每个位置使用独立的分类头，基于3D特征提取器"""
    modelname = "3DResNetClassifier"
    def __init__(self, num_classes=16, encoder_out_vec_num=8):
        super(ResNet3DClassifier, self).__init__()
        self.feature_extractor = resnet18_3d(num_classes=1024)
        self.encoder_out_vec_num = encoder_out_vec_num
        self.num_classes = num_classes
        self.vq_layer = nn.ModuleList([ nn.Sequential(nn.Linear(1024, num_classes),
            # nn.Dropout(0.3),
            nn.ReLU(), nn.Linear(num_classes, num_classes)) for _ in range(encoder_out_vec_num)])

    def forward(self, right_voxel, device):
        # 体素特征提取
        right_voxel = right_voxel.to(device)
        features = self.feature_extractor(right_voxel) # [batch_size, img_feature_dim]
        logits = torch.zeros((features.shape[0], self.num_classes, self.encoder_out_vec_num)).to(device)  # [batch_size, num_classes, encoder_out_vec_num]
        for i in range(self.encoder_out_vec_num):
            logits[:, :, i] = self.vq_layer[i](features)

        # 释放内存
        del right_voxel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 获取预测类别
        predictions = torch.argmax(logits, dim=1)  # [batch_size, encoder_out_vec_num]

        return predictions, logits