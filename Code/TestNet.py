import torch
from torch import nn
from torch.nn import functional as F


def linear_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    )


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


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        # ResNet风格的卷积部分
        self.conv_net = nn.Sequential(
            # 初始卷积层
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 残差块
            *resnet_block(64, 64, 2, first_block=True),
            # *resnet_block(64, 128, 2),
            # *resnet_block(128, 256, 2),
            *resnet_block(64, 256, 2),

            # 最终处理
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc0 = linear_block(2, 128)
        self.fc1 = linear_block(128, 256)
        self.fc2 = linear_block(256, 256)
        self.fc3 = linear_block(256, 128)
        self.fc4 = linear_block(128, 64)
        self.net = nn.Sequential(self.fc0, self.fc1, self.fc2, self.fc3, self.fc4)
        # 保持原有全连接结构（输入维度自动匹配512 + positioncode_dim）
        self.imgfc = nn.Sequential(
            linear_block(512, 256),
            linear_block(256,256),
        )
        self.output = nn.Linear(256+64, 108)

    def forward(self, image_left, image_right, pos):
        # 分别提取左、右耳特征
        img_feat_left = self.conv_net(image_left)  # [batch, 512]
        img_feat_right = self.conv_net(image_right)  # [batch, 512]

        # 拼接图像特征
        img_feat = torch.cat([img_feat_left, img_feat_right], dim=1)  # [batch, 1024]
        img_feat = self.imgfc(img_feat)

        # 位置编码处理
        pos_feat = pos.squeeze(1)  # [batch, positioncode_dim]
        pos_feat = self.net(pos_feat)

        # 特征融合
        combined = torch.cat([img_feat, pos_feat], dim=1)
        return self.output(combined)

