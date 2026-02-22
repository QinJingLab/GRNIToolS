
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import torch.nn.functional as F



class GRNCNN(nn.Module):  
    def __init__(self, input_channel=1, output_channel=64, kernel_size=3,stride=1,padding=1,hidden =128):  
        super(GRNCNN, self).__init__()  
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_channel, output_channel*2, kernel_size=kernel_size, stride=stride, padding=padding),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 15 * 15, 512),
            nn.ReLU(),
            nn.Linear(512, hidden),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):  
        # 输入x的shape应为[batch_size, 1, 60, 60]  
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output


class BasicBlock(nn.Module):  
    expansion = 1  
  
    def __init__(self, in_planes, planes, stride=1):  
        super(BasicBlock, self).__init__()  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(planes)  
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn2 = nn.BatchNorm2d(planes)  
  
        self.shortcut = nn.Sequential()  
        if stride != 1 or in_planes != self.expansion * planes:  
            self.shortcut = nn.Sequential(  
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),  
                nn.BatchNorm2d(self.expansion * planes)  
            )  
  
    def forward(self, x):  
        out = F.relu(self.bn1(self.conv1(x)))  
        out = self.bn2(self.conv2(out))  
        out += self.shortcut(x)  
        out = F.relu(out)  
        return out  
  
class GRNResNet(nn.Module):  
    
    def __init__(self, num_classes=128):  
        super(GRNResNet, self).__init__()  
        self.in_planes = 64  
  
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(64)  
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)  
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)  
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)  
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)  
        self.linear = nn.Linear(512 * 4, num_classes)  
        self.fc = nn.Linear(num_classes,1)


    def _make_layer(self, block, planes, num_blocks, stride):  
        strides = [stride] + [1] * (num_blocks - 1)  
        layers = []  
        for stride in strides:  
            layers.append(block(self.in_planes, planes, stride))  
            self.in_planes = planes * block.expansion  
        return nn.Sequential(*layers)  
  
    def forward(self, x):  
        out = F.relu(self.bn1(self.conv1(x)))  
        out = self.layer1(out)  
        out = self.layer2(out)  
        out = self.layer3(out)  
        out = self.layer4(out)  
        out = F.avg_pool2d(out, 4)  
        out = out.view(out.size(0), -1)  
        out = self.linear(out)  
        out = self.fc(out)
        return out 
    
  

