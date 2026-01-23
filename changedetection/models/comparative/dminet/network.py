import sys
sys.path.append('/home/guoxin/MambaCD')
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from changedetection.models.comparative.utils.backbone import load_timm_backbone
from changedetection.models.comparative.changeos.decoder import FeaturePyramidNetworks, BottleNeck

def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
        
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        # print("++",x.size()[1],self.inp_dim,x.size()[1],self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class decode(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, out_channel,norm_layer=nn.BatchNorm2d):
        super(decode, self).__init__()
        self.conv_d1 = nn.Conv2d(in_channel_down, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(in_channel_left, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channel*2, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(out_channel)

    def forward(self, left, down):
        down_mask = self.conv_d1(down)
        left_mask = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')
            z1 = F.relu(left_mask * down_, inplace=True)
        else:
            z1 = F.relu(left_mask * down, inplace=True)

        if down_mask.size()[2:] != left.size()[2:]:
            down_mask = F.interpolate(down_mask, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_mask * left, inplace=True)

        out = torch.cat((z1, z2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class CrossAtt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels

        self.query1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key1   = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.value1 = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)

        self.query2 = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key2   = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.value2 = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)

        self.gamma = nn.Parameter(torch.zeros(1)) 
        self.softmax = nn.Softmax(dim = -1)

        self.conv_cat = nn.Sequential(nn.Conv2d(in_channels*2, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU()) # conv_f

    def forward(self, input1, input2):
        batch_size, channels, height, width = input1.shape
        q1 = self.query1(input1)
        k1 = self.key1(input1).view(batch_size, -1, height * width)
        v1 = self.value1(input1).view(batch_size, -1, height * width)

        q2 = self.query2(input2) 
        k2 = self.key2(input2).view(batch_size, -1, height * width)
        v2 = self.value2(input2).view(batch_size, -1, height * width)

        q = torch.cat([q1,q2],1).view(batch_size, -1, height * width).permute(0, 2, 1)
        attn_matrix1 = torch.bmm(q, k1)  
        attn_matrix1 = self.softmax(attn_matrix1)
        out1 = torch.bmm(v1, attn_matrix1.permute(0, 2, 1)) 
        out1 = out1.view(*input1.shape)
        out1 = self.gamma * out1 + input1

        attn_matrix2 = torch.bmm(q, k2) 
        attn_matrix2 = self.softmax(attn_matrix2)
        out2 = torch.bmm(v2, attn_matrix2.permute(0, 2, 1))  
        out2 = out2.view(*input2.shape)
        out2 = self.gamma * out2 + input2

        feat_sum = self.conv_cat(torch.cat([out1,out2],1))
        return feat_sum, out1, out2

class DMINet(nn.Module):
    def __init__(self, num_classes=2, normal_init=True, backbone=str,
                 out_indices=(1, 2, 3, 4), pretrain=False):
        super(DMINet, self).__init__()
        if backbone == 'resnet18':
            channels = [64,128,256,512]
        elif backbone == 'resnet50':
            channels = [256,512,1024,2048]
        else:
            raise ValueError(f"backbone only support resnet18 and resnet50, but got {backbone}")
        self.encoder = load_timm_backbone(backbone, pretrain=pretrain, out_indices=out_indices)
        if out_indices == (0, 1, 2, 3):
            channels = [64] + [x//2 for x in channels[1:]]
        self.channels = channels
        self.cross2 = CrossAtt(channels[3], channels[3])
        self.cross3 = CrossAtt(channels[2], channels[2])
        self.cross4 = CrossAtt(channels[1], channels[1])

        self.Translayer2_1 = BasicConv2d(channels[3],channels[2],1)
        self.fam32_1 = decode(channels[2],channels[2],channels[2]) # AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(channels[2],channels[1],1)
        self.fam43_1 = decode(channels[1],channels[1],channels[1]) # AlignBlock(64) # decode(64,64,64)

        self.Translayer2_2 = BasicConv2d(channels[3],channels[2],1)
        self.fam32_2 = decode(channels[2],channels[2],channels[2])
        self.Translayer3_2 = BasicConv2d(channels[2],channels[1],1)
        self.fam43_2 = decode(channels[1],channels[1],channels[1])

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.final = nn.Sequential(
            Conv(channels[1], channels[0], 3, bn=True, relu=True),
            Conv(channels[0], num_classes, 3, bn=False, relu=False))
        self.final2 = nn.Sequential(
            Conv(channels[1], channels[0], 3, bn=True, relu=True),
            Conv(channels[0], num_classes, 3, bn=False, relu=False))

        if normal_init:
            self.init_weights()

    def init_weights(self):
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final.apply(init_weights)
        self.final2.apply(init_weights)

    def forward(self, imgs1, imgs2):

        x1_0, x1_1, x1_2, x1_3 = self.encoder(imgs1)
        x2_0, x2_1, x2_2, x2_3 = self.encoder(imgs2)

        cross_result2, cur1_2, cur2_2 = self.cross2(x1_3, x2_3)
        cross_result3, cur1_3, cur2_3 = self.cross3(x1_2, x2_2)
        cross_result4, cur1_4, cur2_4 = self.cross4(x1_1, x2_1) 

        out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        out3_2 = self.fam32_2(torch.abs(cur1_3-cur2_3), self.Translayer2_2(torch.abs(cur1_2-cur2_2)))
        out4_2 = self.fam43_2(torch.abs(cur1_4-cur2_4), self.Translayer3_2(out3_2))

        out4_up = self.upsamplex4(out4)
        out4_2_up = self.upsamplex4(out4_2)
        out_1 = self.final(out4_up)
        out_2 = self.final2(out4_2_up)
        out = out_1 + out_2
        return out

class DMINet_BDA(DMINet):
    def __init__(self, out_loc=int, out_dmg=int, backbone='resnet50', 
                 out_indices=(1,2,3,4), pretrain=True):
        super(DMINet_BDA, self).__init__(backbone=backbone, out_indices=out_indices, pretrain=pretrain)
        self.fpn = FeaturePyramidNetworks(in_channels=self.channels, out_channel=self.channels[1])
        self.neck = BottleNeck(self.channels[1])
        self.cls_seg = nn.Sequential(
            Conv(self.channels[1], self.channels[0], 3, bn=True, relu=True),
            Conv(self.channels[0], 64, 3, bn=True, relu=True),
            nn.Conv2d(64, out_loc, 1))
        self.cls_dmg = nn.Sequential(
            Conv(self.channels[1], self.channels[0], 3, bn=True, relu=True),
            Conv(self.channels[0], 64, 3, bn=True, relu=True),
            nn.Conv2d(64, out_dmg, 1))
        del self.final, self.final2
    
    def forward(self, imgs1, imgs2):
        size = imgs1.shape[-2:]
        x1_0, x1_1, x1_2, x1_3 = self.encoder(imgs1)
        x2_0, x2_1, x2_2, x2_3 = self.encoder(imgs2)
        x1_1_down = F.interpolate(x1_1, size=x1_2.size()[2:], mode='bilinear', align_corners=True)
        x2_1_down = F.interpolate(x2_1, size=x2_2.size()[2:], mode='bilinear', align_corners=True)
        cross_result2, cur1_2, cur2_2 = self.cross2(x1_3, x2_3)
        cross_result3, cur1_3, cur2_3 = self.cross3(x1_2, x2_2)
        cross_result4, cur1_4, cur2_4 = self.cross4(x1_1_down, x2_1_down)
        cross_result4 = F.interpolate(cross_result4, size=x1_1.size()[2:], mode='bilinear', align_corners=True)
        cur1_4 = F.interpolate(cur1_4, size=x1_1.size()[2:], mode='bilinear', align_corners=True)
        cur2_4 = F.interpolate(cur2_4, size=x1_1.size()[2:], mode='bilinear', align_corners=True)


        out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        out3_2 = self.fam32_2((cur1_3-cur2_3), self.Translayer2_2((cur1_2-cur2_2)))
        out4_2 = self.fam43_2((cur1_4-cur2_4), self.Translayer3_2(out3_2))

        out4_up = self.upsamplex4(out4)
        out4_2_up = self.upsamplex4(out4_2)
        out_dmg = out4_up + out4_2_up

        fpn_seg = self.fpn([x1_0, x1_1, x1_2, x1_3])
        out_loc = self.cls_seg(fpn_seg)
        out_dmg = self.cls_dmg(out_dmg)
        out_loc = F.interpolate(out_loc, size=size, mode='bilinear', align_corners=True)
        out_dmg = F.interpolate(out_dmg, size=size, mode='bilinear', align_corners=True)
        return out_loc, out_dmg


if __name__ == '__main__':
    # model = DMINet(backbone='resnet50').to('cuda:0')
    model = DMINet_BDA(2, 5, backbone='resnet50',out_indices=(0,1,2,3)).eval().to('cuda:0')
    from torchsummary import summary
    x1 = torch.randn(1, 3, 256, 256).to('cuda:0')
    x2 = torch.randn(1, 3, 256, 256).to('cuda:0')
    y = model(x1, x2)
    # summary(model, [(3, 256, 256), (3, 256, 256)])
    # summary(model, (3, 256, 256))
    print(y[0].shape, y[1].shape)
    from thop import profile
    flops, params = profile(model, inputs=(x1, x2), verbose=False)
    print(f"FLOPs: {flops/1e9:.2f}G, Params: {params/1e6:.2f}M")
