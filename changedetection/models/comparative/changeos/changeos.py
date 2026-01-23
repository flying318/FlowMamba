import sys
sys.path.append('/home/guoxin/MambaCD')
import torch
import torch.nn as nn
import torch.nn.functional as F
from changedetection.models.comparative.changeos.decoder import FeaturePyramidNetworks, BottleNeck, ConvBnAct
from changedetection.models.comparative.utils.backbone import Backbone, load_timm_backbone
# from .decoder import FeaturePyramidNetworks, BottleNeck, ConvBnAct
# from ..utils.backbone import Backbone, load_timm_backbone

class ChangeOS(nn.Module):
    def __init__(self,
                 backbone='resnet50',
                 in_channels: list = None,
                 decoder_channel: int = 64,
                 out_loc: int = 1,
                 out_dmg: int = 5
                 ):
        super(ChangeOS, self).__init__()
        if in_channels is None:
            in_channels = [x *4 for x in [64, 128, 256, 512]]
            # in_channels = [64, 128, 256, 512]
        if len(in_channels) > 4:
            in_channels = in_channels[1:]
        # self.context_encoder = Backbone(arch=backbone, n_stages=5, out_indices=(1, 2, 3, 4))
        self.context_encoder = load_timm_backbone(backbone, pretrain=True, out_indices=(1, 2, 3, 4))
        self.in_channels = in_channels
        self.decoder_channel = decoder_channel
        self.num_bl_class = out_loc
        self.num_bda_class = out_dmg
        self.location_fpn = FeaturePyramidNetworks(self.in_channels, self.decoder_channel)
        self.damage_fpn = FeaturePyramidNetworks(self.in_channels, self.decoder_channel)
        self.location_bottle_neck = BottleNeck(self.decoder_channel)
        self.damage_bottle_neck = BottleNeck(self.decoder_channel)
        self.temporal_fusion_conv = ConvBnAct(self.decoder_channel * 2, self.decoder_channel, kernel_size=3, stride=1, padding=1)
        self.location_mask_generation = nn.Conv2d(self.decoder_channel, self.num_bl_class, kernel_size=1)
        self.damage_mask_generation = nn.Conv2d(self.decoder_channel, self.num_bda_class, kernel_size=1)

    def forward(self, pre_image, post_image):
        x1_2, x1_3, x1_4, x1_5 = self.context_encoder(pre_image)
        x2_2, x2_3, x2_4, x2_5 = self.context_encoder(post_image)
        location_p_out = self.location_fpn([x1_2, x1_3, x1_4, x1_5])
        damage_p_out = self.damage_fpn([x2_2, x2_3, x2_4, x2_5])
        damage_p_fused = self.temporal_fusion_conv(torch.cat([location_p_out, damage_p_out], dim=1))
        location_p = self.location_bottle_neck(location_p_out)
        damage_p = self.damage_bottle_neck(damage_p_fused)
        location_mask = self.location_mask_generation(location_p)
        damage_mask = self.damage_mask_generation(damage_p)
        location_mask = F.interpolate(location_mask, size=pre_image.size()[2:], mode='bilinear', align_corners=True)
        damage_mask = F.interpolate(damage_mask, size=post_image.size()[2:], mode='bilinear', align_corners=True)

        return location_mask, damage_mask

if __name__ == '__main__':
    model = ChangeOS()
    x1 = torch.randn(1, 3, 256, 256)
    x2 = torch.randn(1, 3, 256, 256)
    model.eval()
    print(model(x1, x2)[0].shape, model(x1, x2)[1].shape)
    from thop import profile
    flops, params = profile(model, inputs=(x1, x2), verbose=False)
    print(f"FLOPs: {flops/1e9:.2f}G, Params: {params/1e6:.2f}M")