from functools import reduce
import torch
from torch import nn, autograd
import torchvision.models as models
from torch.nn import functional as F

import torch.nn as nn
import math
import numpy as np

device = 'cuda:2'

__all__ = ['Resnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class WALinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(WALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sub_num_classes = self.out_features//5
        self.WA_linears = nn.ModuleList()
        self.WA_linears.extend([nn.Linear(self.in_features, self.sub_num_classes, bias=False) for i in range(5)])

    def forward(self, x):
        out1 = self.WA_linears[0](x)
        out2 = self.WA_linears[1](x)
        out3 = self.WA_linears[2](x)
        out4 = self.WA_linears[3](x)
        out5 = self.WA_linears[4](x)

        return torch.cat([out1, out2, out3, out4, out5], dim = 1)
    
    def align_norms(self, step_b):
        # Fetch old and new layers
        new_layer = self.WA_linears[step_b]
        old_layers = self.WA_linears[:step_b]
        
        # Get weight of layers
        new_weight = new_layer.weight.cpu().detach().numpy()
        for i in range(step_b):
            old_weight = np.concatenate([old_layers[i].weight.cpu().detach().numpy() for i in range(step_b)])
        print("old_weight's shape is: ",old_weight.shape)
        print("new_weight's shape is: ",new_weight.shape)

        # Calculate the norm
        Norm_of_new = np.linalg.norm(new_weight, axis=1)
        Norm_of_old = np.linalg.norm(old_weight, axis=1)
        assert(len(Norm_of_new) == 20)
        assert(len(Norm_of_old) == step_b*20)
        
        # Calculate the Gamma
        gamma = np.mean(Norm_of_new) / np.mean(Norm_of_old)
        print("Gamma = ", gamma)

        # Update new layer's weight
        updated_new_weight = torch.Tensor(gamma * new_weight).to(device)
        print(updated_new_weight)
        self.WA_linears[step_b].weight = torch.nn.Parameter(updated_new_weight)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class Resnet(nn.Module):
    def __init__(self, depth, num_classes=1000, block_name='BasicBlock'):
        super(Resnet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.avgpool = nn.AdaptiveAvgPool2d((2,2))
        self.dropout_on = False
        
        # self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc = WALinear(64 * block.expansion, num_classes)
#         self.fc = WALinear(64 * block.expansion * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
#         print(x)
        x = self.relu(x)
        x = F.dropout(x,0.1,training=self.dropout_on)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
#         print(x)
        return x

    def weight_align(self, step_b):
        self.fc.align_norms(step_b)

