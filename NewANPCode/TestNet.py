import torch
from torch import nn
from torch.nn import functional as F

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
        # ResNet风格的卷积部分
        self.conv_net = nn.Sequential(
            # 初始卷积层
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 残差块
            *resnet_block(64, 64, 2, first_block=True),
            *resnet_block(64, 256, 2),

            # 最终处理
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.imgfc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

    def forward(self, image_left, image_right):
        # 分别提取左、右耳特征
        img_feat_left = self.conv_net(image_left)  # [batch, 256]
        img_feat_right = self.conv_net(image_right)  # [batch, 256]

        # 拼接图像特征
        img_feat = torch.cat([img_feat_left, img_feat_right], dim=1)  # [batch, 512]
        return self.imgfc(img_feat)  # [batch, 256]

class PredictionNet(nn.Module):
    """基于特征的预测网络"""
    def __init__(self):
        super(PredictionNet, self).__init__()
        # 位置特征处理网络
        self.fc0 = nn.Linear(2, 128)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        
        # 输出层
        self.output = nn.Linear(256+64, 108)
        
    def forward(self, img_features, pos):
        # 处理位置信息
        pos_feat = F.relu(self.fc0(pos))
        pos_feat = F.relu(self.fc1(pos_feat))
        pos_feat = F.relu(self.fc2(pos_feat))
        pos_feat = F.relu(self.fc3(pos_feat))
        pos_feat = F.relu(self.fc4(pos_feat))
        
        # 特征融合
        combined = torch.cat([img_features, pos_feat], dim=1)
        return self.output(combined)

class TestNet(nn.Module):
    """完整网络，集成特征提取和预测"""
    def __init__(self):
        super(TestNet, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.prediction_net = PredictionNet()
        
    def forward(self, image_left, image_right, pos):
        # 特征提取
        img_features = self.feature_extractor(image_left, image_right)
        # 预测
        return self.prediction_net(img_features, pos)