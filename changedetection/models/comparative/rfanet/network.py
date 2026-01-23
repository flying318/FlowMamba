import sys
sys.path.append('/home/guoxin/MambaCD')
import torch
import torch.nn as nn
import torch.nn.functional as F
from changedetection.models.comparative.utils.backbone import load_timm_backbone
from changedetection.models.comparative.rfanet.util import *

class BaseNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, backbone='resnet50', en_dim=512):
        super(BaseNet, self).__init__()
        self.backbone = load_timm_backbone(backbone, out_indices=(0, 1, 2, 3, 4))
        self.channles = [64, 256, 512, 1024, 2048]
        self.en_d = en_dim
        self.mid_d = self.en_d * 2
        self.frm = FeatureReinforcementModule(self.channles, self.mid_d)
        self.tfm = TemporalFusionModule(self.mid_d, self.mid_d)
        self.gca = GlobalContextAggregation(self.mid_d, self.mid_d)
        self.decoder = Decoder(self.en_d * 2)

    def forward(self, x1, x2):
        # forward backbone resnet
        feats_1 = self.backbone(x1)
        feats_2 = self.backbone(x2)
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.frm(feats_1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.frm(feats_2)
        # temporal fusion
        c1, c2, c3, c4, c5 = self.tfm(x1_1, x1_2, x1_3, x1_4, x1_5, x2_1, x2_2, x2_3, x2_4, x2_5)
        # global context of high-level and low-level
        gc_c4 = self.gca(c4, c5)

        # fpn
        mask_p1 = self.decoder(c1, c2, c3, c4, c5, gc_c4)
        # change map
        out_logit = F.interpolate(mask_p1, scale_factor=(2, 2), mode='bilinear')
        return out_logit
    
class RFANet_BDA(BaseNet):
    def __init__(self, out_loc, out_dmg, backbone='resnet50', en_dim=32, **kwargs):
        super(RFANet_BDA, self).__init__(backbone=backbone, en_dim=en_dim, **kwargs)
        self.en_d = en_dim
        self.decoder_building = Decoder(self.en_d * 2, out_loc)
        self.decoder_damage = Decoder(self.en_d * 2, out_dmg)

    def forward(self, x1, x2):
        feats_1 = self.backbone(x1)
        feats_2 = self.backbone(x2)
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.frm(feats_1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.frm(feats_2)
        c1, c2, c3, c4, c5 = self.tfm(x1_1, x1_2, x1_3, x1_4, x1_5, x2_1, x2_2, x2_3, x2_4, x2_5)
        gc_dmg = self.gca(c4, c5)
        gc_seg = self.gca(x1_4, x1_5)
        out_loc = self.decoder_building(x1_1, x1_2, x1_3, x1_4, x1_5, gc_seg)
        out_dmg = self.decoder_damage(c1, c2, c3, c4, c5, gc_dmg)
        out_loc = F.interpolate(out_loc, scale_factor=(2, 2), mode='bilinear')
        out_dmg = F.interpolate(out_dmg, scale_factor=(2, 2), mode='bilinear')
        return out_loc, out_dmg

if __name__ == '__main__':
    # model = BaseNet()
    model = RFANet_BDA(2, 5)
    x1 = torch.randn(1, 3, 256, 256)
    x2 = torch.randn(1, 3, 256, 256)
    y = model(x1, x2)
    print(y[0].shape, y[1].shape)
        