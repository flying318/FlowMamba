import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

class LatentSimilarity(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """
    def __init__(self, reduction='mean', T=1.0):
        super(LatentSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)
        self.T = T
        
    def forward(self, x1, x2, label_change):
        b,c,h,w = x1.size()
        x1 = F.softmax(x1/self.T, dim=1)
        x2 = F.softmax(x2/self.T, dim=1)
        x1 = x1.permute(0,2,3,1)
        x2 = x2.permute(0,2,3,1)
        x1 = torch.reshape(x1,[b*h*w,c])
        x2 = torch.reshape(x2,[b*h*w,c])
        
        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        #target = target - label_change.float()
        target = torch.reshape(target,[b*h*w])
        
        loss = self.loss_f(x1, x2, target)
        return loss

class Space_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(Space_Attention, self).__init__()
        self.SA = nn.Sequential( 
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()        
        A = self.SA(x)
        return A

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high + in_channels_low
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, low_feat):
        x = self.up(x)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)        
        return x

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def load_pretrained_weights(model, pretrained_path):
    state_dict = torch.load(pretrained_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict = state_dict
    model.load_state_dict(new_state_dict, strict=False)
    print("Loaded pretrained weights from {}".format(pretrained_path))
    return model
