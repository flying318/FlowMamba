#  Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery,
#  IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3310208. C. HAN, C. WU, H. GUO, M. HU, J.Li AND H. CHEN,
import sys
sys.path.append('/home/guoxin/MambaCD')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch import Tensor
from typing import Tuple
from changedetection.models.comparative.cgnet.util import BasicConv2d, ChangeGuideModule
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from changedetection.models.comparative.utils.backbone import load_timm_backbone
# from ..utils.backbone import load_timm_backbone

class HCGMNet(nn.Module):
    def __init__(self,):
        super(HCGMNet, self).__init__()
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 128
        self.down2 = vgg16_bn.features[12:22]  # 256
        self.down3 = vgg16_bn.features[22:32]  # 512
        self.down4 = vgg16_bn.features[32:42]  # 512

        self.conv_reduce_1 = BasicConv2d(128*2,128,3,1,1)
        self.conv_reduce_2 = BasicConv2d(256*2,256,3,1,1)
        self.conv_reduce_3 = BasicConv2d(512*2,512,3,1,1)
        self.conv_reduce_4 = BasicConv2d(512*2,512,3,1,1)

        self.up_layer4 = BasicConv2d(512,512,3,1,1)
        self.up_layer3 = BasicConv2d(512,512,3,1,1)
        self.up_layer2 = BasicConv2d(256,256,3,1,1)

        # self.decoder = nn.Sequential(BasicConv2d(1408,512,3,1,1),BasicConv2d(512,256,3,1,1),BasicConv2d(256,64,3,1,1),nn.Conv2d(64,1,3,1,1))
        self.deocde = nn.Sequential(BasicConv2d(1408, 512, 3, 1, 1), BasicConv2d(512, 256, 3, 1, 1),BasicConv2d(256, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))

        # self.decoder_final = nn.Sequential(BasicConv2d(1408,512,3,1,1),BasicConv2d(512,256,3,1,1),BasicConv2d(256,64,3,1,1),nn.Conv2d(64,1,3,1,1))
        self.deocde_final = nn.Sequential(BasicConv2d(1408, 512, 3, 1, 1), BasicConv2d(512, 256, 3, 1, 1),BasicConv2d(256, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))

        self.cgm_2 = ChangeGuideModule(256)
        self.cgm_3 = ChangeGuideModule(512)
        self.cgm_4 = ChangeGuideModule(512)

    # def forward(self, A,B=None):
    #     if B == None:
    #         B = A
    def forward(self,A,B):
        size = A.size()[2:]
        layer1_pre = self.inc(A)
        layer1_A = self.down1(layer1_pre)
        layer2_A = self.down2(layer1_A)
        layer3_A = self.down3(layer2_A)
        layer4_A = self.down4(layer3_A)

        layer1_pre = self.inc(B)
        layer1_B = self.down1(layer1_pre)
        layer2_B = self.down2(layer1_B)
        layer3_B = self.down3(layer2_B)
        layer4_B = self.down4(layer3_B)

        layer1 = torch.cat((layer1_B,layer1_A),dim=1)

        layer2 = torch.cat((layer2_B,layer2_A),dim=1)

        layer3 = torch.cat((layer3_B,layer3_A),dim=1)

        layer4 = torch.cat((layer4_B,layer4_A),dim=1)

        layer1 = self.conv_reduce_1(layer1)
        layer2 = self.conv_reduce_2(layer2)
        layer3 = self.conv_reduce_3(layer3)
        layer4 = self.conv_reduce_4(layer4)

        layer4 = self.up_layer4(layer4)

        layer3 = self.up_layer3(layer3)

        layer2 = self.up_layer2(layer2)

        layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer3_1 = F.interpolate(layer3, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer2_1 = F.interpolate(layer2, layer1.size()[2:], mode='bilinear', align_corners=True)

        feature_fuse = torch.cat((layer1,layer2_1,layer3_1,layer4_1),dim=1)
        change_map = self.deocde(feature_fuse)
        # ---------------注释这两句------------------------
        # if not self.training:
        #     feature_fuse = torch.cat((layer1,layer2_1,layer3_1,layer4_1), dim=1)
        #     feature_fuse = feature_fuse.cpu().detach().numpy()
        #     for num in range(0, 511):
        #         display = feature_fuse[0, num, :, :]  # 第几张影像，第几层特征0-511
        #         plt.figure()
        #         plt.imshow(display)  # [B, C, H,W]
        #         plt.savefig('./test_result/feature_fuse-v2/' + 'v2-fuse-' + str(num) + '.png')
        # change_map = self.decoder(torch.cat((layer1,layer2_1,layer3_1,layer4_1), dim=1))
        # ---------------注释这两句------------------------

        layer2 = self.cgm_2(layer2, change_map)
        layer3 = self.cgm_3(layer3, change_map)
        layer4 = self.cgm_4(layer4, change_map)

        layer4_2 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer3_2 = F.interpolate(layer3, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer2_2 = F.interpolate(layer2, layer1.size()[2:], mode='bilinear', align_corners=True)

        new_feature_fuse = torch.cat((layer1,layer2_2,layer3_2,layer4_2),dim=1)

        change_map = F.interpolate(change_map, size, mode='bilinear', align_corners=True)
        final_map = self.deocde_final(new_feature_fuse)

        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)

        return change_map, final_map
    
class HCGMNet_BDA(HCGMNet):
    def __init__(self, out_loc, out_dmg, backbone='resnet50'):
        super(HCGMNet_BDA, self).__init__()
        self.encoder = load_timm_backbone(backbone, out_indices=(1,2,3,4))
        self.seg_up_layer4 = nn.Sequential(
            BasicConv2d(2048, 1024, 3, 1, 1), nn.UpsamplingBilinear2d(scale_factor=2))
        self.seg_up_layer3 = nn.Sequential(
            BasicConv2d(1024*2, 512, 3, 1, 1), nn.UpsamplingBilinear2d(scale_factor=2))
        self.seg_up_layer2 = nn.Sequential(
            BasicConv2d(512*2, 256, 3, 1, 1), nn.UpsamplingBilinear2d(scale_factor=2))
        self.seg_up_layer1 = nn.Sequential(
            BasicConv2d(256*2, 128, 3, 1, 1), nn.UpsamplingBilinear2d(scale_factor=2))
        self.cls_seg = nn.Conv2d(128, out_loc, 1)
        self.conv_reduce_1 = BasicConv2d(256*2, 128, 3, 1, 1)
        self.conv_reduce_2 = BasicConv2d(512*2, 256, 3, 1, 1)
        self.conv_reduce_3 = BasicConv2d(1024*2, 512, 3, 1, 1)
        self.conv_reduce_4 = BasicConv2d(2048*2, 512, 3, 1, 1)
        self.cls_bda = nn.Sequential(
            BasicConv2d(1408, 512, 3, 1, 1), 
            BasicConv2d(512, 256, 3, 1, 1),
            BasicConv2d(256, 64, 3, 1, 1), 
            nn.Conv2d(64, out_dmg, 3, 1, 1))
        # del self.decode, self.decoder_final


    def decoder_damage(self, feats_a=Tuple[Tensor], feats_b=Tuple[Tensor], size=(256,256)):
        a_1, a_2, a_3, a_4 = feats_a
        b_1, b_2, b_3, b_4 = feats_b
        layer1 = torch.cat((a_1, b_1), dim=1)
        layer2 = torch.cat((a_2, b_2), dim=1)
        layer3 = torch.cat((a_3, b_3), dim=1)
        layer4 = torch.cat((a_4, b_4), dim=1)

        layer1 = self.conv_reduce_1(layer1)
        layer2 = self.conv_reduce_2(layer2)
        layer3 = self.conv_reduce_3(layer3)
        layer4 = self.conv_reduce_4(layer4)
        layer4 = self.up_layer4(layer4)
        layer3 = self.up_layer3(layer3)
        layer2 = self.up_layer2(layer2)
        layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer3_1 = F.interpolate(layer3, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer2_1 = F.interpolate(layer2, layer1.size()[2:], mode='bilinear', align_corners=True)
        feature_fuse = torch.cat((layer1, layer2_1, layer3_1, layer4_1), dim=1)
        change_map = self.cls_bda(feature_fuse)
        layer2 = self.cgm_2(layer2, change_map)
        layer3 = self.cgm_3(layer3, change_map)
        layer4 = self.cgm_4(layer4, change_map)
        layer4_2 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer3_2 = F.interpolate(layer3, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer2_2 = F.interpolate(layer2, layer1.size()[2:], mode='bilinear', align_corners=True)
        new_feature_fuse = torch.cat((layer1, layer2_2, layer3_2, layer4_2), dim=1)
        output = self.cls_bda(new_feature_fuse)
        output = F.interpolate(output, size, mode='bilinear', align_corners=True)
        return output
    
    def decoder_building(self, feats=Tuple[Tensor], size=(256,256)):
        feat1, feat2, feat3, feat4 = feats
        feat4 = self.seg_up_layer4(feat4)
        feat3 = self.seg_up_layer3(torch.cat([feat4, feat3], 1))
        feat2 = self.seg_up_layer2(torch.cat([feat3, feat2], 1))
        feat1 = self.seg_up_layer1(torch.cat([feat2, feat1], 1))
        output = self.cls_seg(feat1)
        output = F.interpolate(output, size, mode='bilinear', align_corners=True)
        return output

    def forward(self, A, B):
        size = A.size()[2:]
        feats_a = self.encoder(A)
        feats_b = self.encoder(B)
        out_loc = self.decoder_building(feats_a, size)
        out_bda = self.decoder_damage(feats_a, feats_b, size)
        return out_loc, out_bda

if __name__=='__main__':
    # 测试模型大小
    input_size = 256
    # model_restoration = HCGMNet().cuda()
    # model_restoration = CGNet().cuda()
    model_restoration = HCGMNet_BDA(2, 5).cuda()

    # 创建输入张量
    input_A = torch.rand(1, 3, input_size, input_size).cuda()
    input_B = torch.rand(1, 3, input_size, input_size).cuda()
    output = model_restoration(input_A, input_B)
    print(output[0].shape, output[1].shape)
    # 使用 fvcore 计算模型复杂度
    flops = FlopCountAnalysis(model_restoration, (input_A, input_B))
    macs = flops.total()
    params = sum(p.numel() for p in model_restoration.parameters())

    # 打印模型统计信息
    # print(parameter_count_table(model_restoration))
    print('{:<30}  {:<8}'.format('Computational complexity (GFLOPs): ', f"{macs / 1e9:.2f} G"))
    print('{:<30}  {:<8}'.format('Number of parameters: ', f"{params / 1e6:.2f} M"))
    from thop import profile
    flops, params = profile(model_restoration, inputs=(input_A, input_B), verbose=False)
    print(f"FLOPs: {flops/1e9:.2f}G, Params: {params/1e6:.2f}M")