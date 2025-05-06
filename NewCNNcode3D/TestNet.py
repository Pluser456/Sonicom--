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
class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        
        self.dim_adapter = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(1),
            nn.ReLU()
        )
        
        self.conv_net = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            *resnet_block_3d(64, 64, 2, first_block=True),
            *resnet_block_3d(64, 256, 2),
            nn.AdaptiveAvgPool3d((16, 1, 1)),  # 关键修改
            nn.Flatten(start_dim=1),            # 输出 [batch, 256 * 16=4096]
            nn.Linear(4096, 512)                # 输入维度匹配4096
        )
        
        # 保持其他全连接层不变
        self.fc0 = linear_block(3, 128)
        self.fc1 = linear_block(128, 256)
        self.fc2 = linear_block(256, 256)
        self.fc3 = linear_block(256, 128)
        self.fc4 = linear_block(128, 64)
        self.net = nn.Sequential(self.fc0, self.fc1, self.fc2, self.fc3, self.fc4  )           # 输出 [batch, 64])
        
        self.imgfc = nn.Sequential(
            linear_block(1024, 256),
            linear_block(256, 256),
        )
        self.output = nn.Linear(256+64, 108)

    def forward(self, image_left, image_right, pos):
        x_left = self.dim_adapter(image_left).squeeze(2)
        x_right = self.dim_adapter(image_right).squeeze(2)
        
        img_feat_left = self.conv_net(x_left)
        img_feat_right = self.conv_net(x_right)
        
        img_feat = torch.cat([img_feat_left, img_feat_right], dim=1)
        img_feat = self.imgfc(img_feat)
        
              
        num_positions = pos.shape[1]
        features = img_feat.unsqueeze(1).repeat(1, num_positions, 1)
        # features = torch.cat([img_feat_repeated, pos], dim=2)
        features = features.reshape(-1, features.shape[-1])
        
        
        pos_feat = self.net(pos)
        pos_feat = pos_feat.reshape(-1, pos_feat.shape[-1])
        #   # 维度验证
        # print(f"图像特征维度: {img_feat.shape}")  # 应为[batch,256]
        # print(f"位置特征维度: {pos_feat.shape}")  # 应为[batch,64]
        combined = torch.cat([features, pos_feat], dim=1)
        return self.output(combined)