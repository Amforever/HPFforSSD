"""
    Author: zhenyuzhang
    All rights reserved.
"""
import torch
from torch import nn
import torch.nn.functional as F


 



def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)


class BasicBlockX(nn.Module):
    expansion = 2  #等于1时，downsample为 None；运行到out += residual时会报错，因为没有进行下采样操作，使得channe维度一致。

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(BasicBlockX, self).__init__()
        self.conv1 = conv3x3(inplanes, planes*2, stride)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes*2, planes*2, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(BottleneckX, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes*2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.conv2 = nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)  #进出的通道必须能被 num_group整除，默认是1；groups控制输入和输出之间的连接。
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, num_classes=2, num_group=32):   #num_classes=1000
        self.inplanes = 64
        super(ResNeXt, self).__init__()

        self.conv1_1 =  nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.conv1_8 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3,bias=False)
        # self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)  #(3,彩色图像 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_group)
        self.layer2 = self._make_layer(block, 128, layers[1], num_group, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], num_group, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], num_group, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #nn.AvgPool2d(2, stride=1)  #nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_group=num_group))  #进行下采样改变 通道个数维度层，
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))   #从第二层开始直接堆砌，采样通道不再变化

        return nn.Sequential(*layers)

    def forward(self, x):
        if x.shape.__len__()==4:
            batch_size, chnal, h, w = x.size()
            x = x.view(batch_size, chnal, h, w)  # 对于cnn模型，直接将conv1的channel的个数设置为8即可，
            x = self.conv1_8(x)
        else:
            batch_size,  h, w = x.size()
            x = x.view(batch_size, 1, h, w)
            x = self.conv1_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.logsoftmax(x)

        return x


def resnext18( **kwargs):
    """Constructs a ResNeXt-18 model.
    """
    model = ResNeXt(BasicBlockX, [2, 2, 2, 2], **kwargs)
    return model


def resnext34(**kwargs):
    """Constructs a ResNeXt-34 model.
    """
    model = ResNeXt(BasicBlockX, [3, 4, 6, 3], **kwargs)
    return model


def resnext50(**kwargs):
    """Constructs a ResNeXt-50 model.
    """
    model = ResNeXt(BottleneckX, [3, 4, 6, 3], **kwargs)
    return model


def resnext101(**kwargs):
    """Constructs a ResNeXt-101 model.
    """
    model = ResNeXt(BottleneckX, [3, 4, 23, 3], **kwargs)
    return model


def resnext152(**kwargs):
    """Constructs a ResNeXt-152 model.
    """
    model = ResNeXt(BottleneckX, [3, 8, 36, 3], **kwargs)
    return model





#变体
def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)


class BasicBlockX_varTanh(nn.Module):
    expansion = 2  #等于1时，downsample为 None；运行到out += residual时会报错，因为没有进行下采样操作，使得channe维度一致。

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(BasicBlockX_varTanh, self).__init__()
        self.conv1 = conv3x3(inplanes, planes*2, stride)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.tanh = nn.Tanh(inplace=True)
        self.conv2 = conv3x3(planes*2, planes*2, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.tanh(out)

        return out


class BottleneckX_varTanh(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(BottleneckX_varTanh, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes*2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.conv2 = nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)  #进出的通道必须能被 num_group整除，默认是1；groups控制输入和输出之间的连接。
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.tanh = nn.Tanh(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.tanh(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.tanh(out)

        return out


class ResNeXt_varTanh(nn.Module):

    def __init__(self, block, layers, num_classes=2, num_group=32):   #num_classes=1000
        self.inplanes = 64
        super(ResNeXt_varTanh, self).__init__()
        self.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3,bias=False)  #(3,彩色图像 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.tanh = nn.Tanh()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_group)
        self.layer2 = self._make_layer(block, 128, layers[1], num_group, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], num_group, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], num_group, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #nn.AvgPool2d(2, stride=1)  #nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_group=num_group))  #进行下采样改变 通道个数维度层，
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))   #从第二层开始直接堆砌，采样通道不再变化

        return nn.Sequential(*layers)

    def forward(self, x):
        # 使用滤波特征
        batch_size, chnal, h, w = x.size()
        x = x.view(batch_size, chnal, h, w)  # 对于cnn模型，直接将conv1的channel的个数设置为8即可，
        # # 使用原始的mfcc等特征
        # batch_size,  h, w = x.size()
        # x = x.view(batch_size, 1, h, w)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.tanh(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.logsoftmax(x)

        return x




def resnext34_varTanh(**kwargs):
    """Constructs a ResNeXt-34 model.
    """
    model = ResNeXt_varTanh(BasicBlockX, [3, 4, 6, 3], **kwargs)
    return model


def resnext50_varTanh(**kwargs):
    """Constructs a ResNeXt-50 model.
    """
    model = ResNeXt_varTanh(BottleneckX, [3, 4, 6, 3], **kwargs)
    return model

def resnext101_varTanh(**kwargs):
    """Constructs a ResNeXt-101 model.
    """
    model = ResNeXt_varTanh(BottleneckX, [3, 4, 23, 3], **kwargs)
    return model


#变体 Non BatchNorm
def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)


class BasicBlockX_varNonBN(nn.Module):
    expansion = 2  #等于1时，downsample为 None；运行到out += residual时会报错，因为没有进行下采样操作，使得channe维度一致。

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(BasicBlockX_varNonBN, self).__init__()
        self.conv1 = conv3x3(inplanes, planes*2, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes*2, planes*2, groups=num_group)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX_varNonBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(BottleneckX_varNonBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes*2, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)  #进出的通道必须能被 num_group整除，默认是1；groups控制输入和输出之间的连接。
        self.conv3 = nn.Conv2d(planes*2, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt_varNonBN(nn.Module):

    def __init__(self, block, layers, num_classes=2, num_group=32):   #num_classes=1000
        self.inplanes = 64
        super(ResNeXt_varNonBN, self).__init__()
        self.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3,bias=False)  #(3,彩色图像 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_group)
        self.layer2 = self._make_layer(block, 128, layers[1], num_group, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], num_group, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], num_group, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #nn.AvgPool2d(2, stride=1)  #nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_group=num_group))  #进行下采样改变 通道个数维度层，
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))   #从第二层开始直接堆砌，采样通道不再变化

        return nn.Sequential(*layers)

    def forward(self, x):
        # 使用滤波特征
        batch_size, chnal, h, w = x.size()
        x = x.view(batch_size, chnal, h, w)  # 对于cnn模型，直接将conv1的channel的个数设置为8即可，
        # # 使用原始的mfcc等特征
        # batch_size,  h, w = x.size()
        # x = x.view(batch_size, 1, h, w)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.logsoftmax(x)

        return x



def resnext34_varNonBN(**kwargs):
    """Constructs a ResNeXt-34 model.
    """
    model = ResNeXt_varNonBN(BasicBlockX, [3, 4, 6, 3], **kwargs)
    return model


def resnext50_varNonBN(**kwargs):
    """Constructs a ResNeXt-50 model.
    """
    model = ResNeXt_varNonBN(BottleneckX, [3, 4, 6, 3], **kwargs)
    return model


def resnext101_varNonBN(**kwargs):
    """Constructs a ResNeXt-101 model.
    """
    model = ResNeXt_varNonBN(BottleneckX, [3, 4, 23, 3], **kwargs)
    return model

