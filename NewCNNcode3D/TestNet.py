import torch
from torch import nn
from torch.nn import functional as F

def linear_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.ReLU()
    )

class Residual3D(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        # 3D卷积核参数
        self.conv1 = nn.Conv3d(input_channels, num_channels,
                              kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv3d(num_channels, num_channels,
                              kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, num_channels,
                                  kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm3d(num_channels)
        self.bn2 = nn.BatchNorm3d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block_3d(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual3D(input_channels, num_channels,
                                 use_1x1conv=True, strides=2))
        else:
            blk.append(Residual3D(num_channels, num_channels))
    return blk

class TestNet3D(nn.Module):
    def __init__(self):
        super(TestNet3D, self).__init__()
        # 3D卷积网络部分
        self.conv_net = nn.Sequential(
            # 初始3D卷积层（输入通道设为1，假设输入为体数据）
            nn.Conv3d(1, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),

            # 3D残差块
            *resnet_block_3d(64, 64, 2, first_block=True),
            *resnet_block_3d(64, 256, 2),

            # 空间维度自适应池化
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # 保持深度维度
            nn.Flatten(start_dim=2),             # 展平空间维度
            nn.Linear(256, 512)                  # 调整特征维度
        )

        # 位置编码网络（保持原结构）
        self.fc0 = linear_block(2, 128)
        self.fc1 = linear_block(128, 256)
        self.fc2 = linear_block(256, 256)
        self.fc3 = linear_block(256, 128)
        self.fc4 = linear_block(128, 64)
        self.net = nn.Sequential(self.fc0, self.fc1, self.fc2, self.fc3, self.fc4)

        # 最终融合层
        self.imgfc = nn.Sequential(
            linear_block(1024, 256),  # 输入维度调整为2 * 512
            linear_block(256, 256),
        )
        self.output = nn.Linear(256+64, 108)

    def forward(self, image_left, image_right, pos):
        # 处理3D输入（添加通道维度）
        img_feat_left = self.conv_net(image_left.unsqueeze(1))  # [batch, 512]
        img_feat_right = self.conv_net(image_right.unsqueeze(1))  # [batch, 512]

        # 特征拼接和处理
        img_feat = torch.cat([img_feat_left, img_feat_right], dim=1)
        img_feat = self.imgfc(img_feat)

        # 位置编码处理
        pos_feat = self.net(pos)

        # 特征融合
        combined = torch.cat([img_feat, pos_feat], dim=1)
        return self.output(combined)