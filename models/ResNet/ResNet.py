import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        #self.drop = nn.Dropout2d(p=0.7)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        #self.drop = nn.Dropout2d(p=0.5)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        #out = self.drop(out)

        out = self.conv3(out)
        out = self.bn3(out)
        #out = self.drop(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, capacity, n_classes, in_channels):
        self.n_classes = n_classes
        self.baseplanes = 4
        self.inplanes = self.baseplanes*capacity*block.expansion
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, self.baseplanes*capacity*block.expansion, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.baseplanes*capacity*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        #self.drop = nn.Dropout2d(p=0.5)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.layer1 = self._make_layer(block, self.baseplanes*capacity*block.expansion, layers[0])
        self.layer2 = self._make_layer(block, self.baseplanes*2*capacity*block.expansion, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.baseplanes*4*capacity*block.expansion, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.baseplanes*8*capacity*block.expansion, layers[3], stride=2)
        #self.fc = nn.Linear(self.baseplanes*8*capacity*block.expansion, self.n_classes-3)

        '''self.classifier = nn.Sequential(
            nn.Conv2d(self.baseplanes*8*capacity*block.expansion, self.baseplanes*16*capacity, 3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(self.baseplanes*16*capacity*block.expansion, self.n_classes, 1),)

        self.score_pool2 = nn.Conv2d(self.baseplanes*capacity, self.n_classes, 1)
        self.score_pool3 = nn.Conv2d(self.baseplanes*2*capacity, self.n_classes, 1)
        self.score_pool4 = nn.Conv2d(self.baseplanes*4*capacity, self.n_classes, 1)'''

        #self.same = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=3, padding=1)
        self.up1 = Up(self.baseplanes*8*capacity*block.expansion*block.expansion)
        self.up2 = Up(self.baseplanes*4*capacity*block.expansion*block.expansion)
        self.up3 = Up(self.baseplanes*2*capacity*block.expansion*block.expansion)
        self.classifier = nn.Sequential(
            nn.Conv2d(self.baseplanes*capacity*block.expansion*block.expansion, self.baseplanes*2*capacity*block.expansion, 3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.ConvTranspose2d(self.baseplanes*2*capacity*block.expansion, self.n_classes, 2, 2),)

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
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        conv2 = self.layer1(out)
        conv3 = self.layer2(conv2)
        conv4 = self.layer3(conv3)
        conv5 = self.layer4(conv4)


        '''score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)
        score_pool3 = self.score_pool3(conv3)
        score_pool2 = self.score_pool2(conv2)'''

        score = self.up1(conv5)
        score = F.interpolate(score, conv4.size()[2:], mode='bilinear', align_corners=True)
        score += conv4
        score = self.up2(score)
        score = F.interpolate(score, conv3.size()[2:], mode='bilinear', align_corners=True)
        score += conv3
        score = self.up3(score)
        score = F.interpolate(score, conv2.size()[2:], mode='bilinear', align_corners=True)
        score += conv2
        score = self.classifier(score)
        #print(score.shape)
        '''score = F.interpolate(score, score_pool4.size()[2:],mode='bilinear',align_corners=True)

        score += score_pool4
        score = F.interpolate(score, score_pool3.size()[2:],mode='bilinear',align_corners=True)
        
        score += score_pool3
        score = F.interpolate(score, score_pool2.size()[2:],mode='bilinear',align_corners=True)
        score += score_pool2'''
        final = F.interpolate(score, x.size()[2:], mode='bilinear',align_corners=True)
        
        #print(final.sum())
        #score = self.up(score, output_size=x.shape)
        '''final = self.avgpool(conv5)
        final = torch.flatten(final, 1)
        #print(conv5.shape, final.shape)
        final = self.fc(final)
        check parallel print("\tIn Model: input size", x.size(),
              "output size", final.size())'''
        return final


def resnet18(pretrained=False, capacity=1, n_classes=1000, in_channels=3):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], capacity, n_classes, in_channels)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet34(pretrained=False, capacity=1, n_classes=1000, in_channels=3):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], capacity, n_classes, in_channels)
    return model

def resnet50(pretrained=False, capacity=1, n_classes=1000, in_channels=3):
    model = ResNet(Bottleneck, [3, 4, 6, 3], capacity, n_classes, in_channels)
    return model

class Up(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)