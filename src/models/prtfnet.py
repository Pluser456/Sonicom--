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

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
    ) -> None:
        super().__init__()
        
        norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class PRTFNet(nn.Module):
    """论文PRTFNet的复现"""
    def __init__(self, pos_num, freq_num):
        super(PRTFNet, self).__init__()
        self.one_hot_fc = nn.Linear(pos_num, 256)
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=(3,2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inchannels = 64
        layers = [3, 4, 6, 3]
        self.layer1 = self.make_layer(Bottleneck, 64, layers[0])       # 输出 256
        self.layer2 = self.make_layer(Bottleneck, 128, layers[1], stride=2) # 输出 512
        self.layer3 = self.make_layer(Bottleneck, 256, layers[2], stride=2) # 输出 1024
        self.layer4 = self.make_layer(Bottleneck, 512, layers[3], stride=1) # 输出 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, freq_num)

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
        return x

    def make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        
        if stride != 1 or self.inchannels != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inchannels, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inchannels, planes, stride, downsample))
        
        self.inchannels = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inchannels, planes))
        return nn.Sequential(*layers)

if __name__ == "__main__":
    model = PRTFNet(pos_num=504, freq_num=39)  # 使用正确的维度
    dummy_img = torch.randn(1, 1, 256, 256)    # Batch=1, 1 通道，256x256
    dummy_dir = torch.randn(1, 16, 504)        # Batch=1, 16 个方向，504 维的 one-hot 编码
    output = model(dummy_img, dummy_dir, device='cpu')
    print(f"输出形状：{output.shape}") 
    # 预期：torch.Size([1, 16, 39]) (取决于 small_batch_num)