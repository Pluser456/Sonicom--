import torch
from torch import nn

def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def conv3x3(
    in_channels: int, out_channels: int, stride: int = 1, padding: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False,
    )

class BasicBlock(nn.Module):
    def __init__(
        self,
        inchannels: int,
        outchannels: int,
        stride: int = 1,
        downsample=None
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inchannels, outchannels, stride)
        self.bn1 = norm_layer(outchannels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outchannels, outchannels)
        self.bn2 = norm_layer(outchannels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class PRTFNet(nn.Module):
    """论文PRTFNet的复现"""
    def __init__(self):
        super(PRTFNet, self).__init__()
        self.one_hot_fc = nn.Sequential(
            nn.Linear(2562, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=(3,2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inchannels = 64
        layers = [3, 4, 6, 3]
        self.layer1 = self.make_layer(BasicBlock, 256, layers[0])
        self.layer2 = self.make_layer(BasicBlock, 512, layers[1], stride=2)
        self.layer3 = self.make_layer(BasicBlock, 1024, layers[2], stride=2)
        self.layer4 = self.make_layer(BasicBlock, 2048, layers[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 90)

    def forward(self, voxel, one_hot, device):
        voxel = voxel.to(device)
        one_hot = one_hot.to(device) # [batch, small_batch_num, 2562]
        voxel_repeated = voxel.unsqueeze(1).repeat(1, one_hot.shape[1], 1, 1, 1) # [batch, small_batch_num, 1, 256, 256]
        shape = voxel_repeated.shape
        voxel = voxel_repeated.reshape(-1, 1, voxel_repeated.shape[3], voxel_repeated.shape[4]) # [batch*small_batch_num, 1, 256, 256]
        one_hot = one_hot.reshape(-1, one_hot.shape[-1]) # [batch*small_batch_num, 2562]
        # 每一行是一个方位的one-hot编码
        one_hot = self.one_hot_fc(one_hot) # [batch*small_batch_num, 256]
        one_hot = one_hot.reshape(one_hot.shape[0], 1, one_hot.shape[1], 1) # [batch*small_batch_num, 1, 256, 1]
        convinput = torch.cat([voxel, one_hot], dim=3) # [batch*small_batch_num, 1, 256, 257]
        x = self.conv1(convinput) # [batch*small_batch_num, 64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # [batch*small_batch_num, 64, 64, 64]
        x = self.layer1(x) # [batch*small_batch_num, 256, 64, 64]
        x = self.layer2(x) # [batch*small_batch_num, 512, 32, 32]
        x = self.layer3(x) # [batch*small_batch_num, 1024, 16, 16]
        x = self.layer4(x) # [batch*small_batch_num, 2048, 16, 16]
        x = self.avgpool(x) # [batch*small_batch_num, 2048, 1, 1]
        x = torch.flatten(x, 1) # [batch*small_batch_num, 2048]
        x = self.fc(x) # [batch*small_batch_num, 90]
        x = x.reshape(shape[0], shape[1], x.shape[-1]) # [batch, small_batch_num, 90]
        del one_hot, voxel, voxel_repeated, convinput
        return x

    def make_layer(self, block, outchannels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None

        if self.inchannels != outchannels:
            downsample = nn.Sequential(
                conv1x1(self.inchannels, outchannels, stride),
                nn.BatchNorm2d(outchannels),
            )

        layers = []
        layers.append(
            block(
                self.inchannels,
                outchannels,
                stride,
                downsample,
            )
        )
        self.inchannels = outchannels
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inchannels,
                    outchannels,
                )
            )

        return nn.Sequential(*layers)
