import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self,in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes !=planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,num_classes=100):
        super(ResNet,self).__init__()
        self.num_classes=num_classes
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3,16,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self.make_layer(16, 5, stride=1)
        self.layer2 = self.make_layer(32, 5, stride=2)
        self.layer3 = self.make_layer(64, 5, stride=2)
        self.linear = nn.Linear(64, num_classes)

    def make_layer(self, planes, num_block, stride):
        strides = [stride] +[1]*(num_block-1)
        layers=[]
        for stride in strides :
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x,8)
        x = x.view(x.size(0),-1)
        x = self.linear(x)
        return x