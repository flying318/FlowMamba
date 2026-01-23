import sys
sys.path.append('xxx') # todo: change this to your project root path
import os
from changedetection.configs.config import get_config
import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from changedetection.models.Mamba_backbone import Backbone_VSSM
from classification.models.vmamba import LayerNorm2d
from typing import List
from torch import Tensor
from changedetection.models.ChangeDecoder import ChangeDecoder
from changedetection.models.SemanticDecoder import SemanticDecoder
from changedetection.utils_func.flow import FlowN


class STMambaBDA(nn.Module):
    def __init__(self, output_building, output_damage, pretrained, **kwargs):
        super(STMambaBDA, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        self.channel_first = self.encoder.channel_first

        print('channel_first:', self.channel_first)

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)


       
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        
        self.decoder_damage = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_building = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )
      
        self.main_clf = nn.Conv2d(in_channels=128, out_channels=output_damage, kernel_size=1)
        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_building, kernel_size=1)
        kwargs['norm_layer'] = norm_layer
        kwargs['ssm_act_layer'] = ssm_act_layer
        kwargs['mlp_act_layer'] = mlp_act_layer
        kwargs['channel_first'] = self.channel_first
        self.flow_kwargs = kwargs.copy()

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_data, post_data, val=False):
        # Encoder processing
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # Decoder processing - passing encoder outputs to the decoder
        output_building = self.decoder_building(pre_features)
        output_damage = self.decoder_damage(pre_features, post_features)
        
        output_building = self.aux_clf(output_building)
        output_building = F.interpolate(output_building, size=pre_data.size()[-2:], mode='bilinear')

        output_damage = self.main_clf(output_damage)
        output_damage = F.interpolate(output_damage, size=post_data.size()[-2:], mode='bilinear')
       
        return output_building, output_damage


class FlowMamba(STMambaBDA):
    def __init__(self, output_building, output_damage, pretrained, **kwargs):
        super(FlowMamba, self).__init__(output_building, output_damage, pretrained, **kwargs)
        self.criterion = nn.SmoothL1Loss()
        self.flow1 = FlowN(self.encoder.dims[0], 64, 64, **self.flow_kwargs)
        self.flow2 = FlowN(self.encoder.dims[1], 32, 32, **self.flow_kwargs)
        self.flow3 = FlowN(self.encoder.dims[2], 16, 16, **self.flow_kwargs)
        self.flow4 = FlowN(self.encoder.dims[3], 8, 8, **self.flow_kwargs)
        # self.flow = nn.ModuleList([self.flow1, self.flow2, self.flow3, self.flow4])
        self.flow = nn.ModuleList([self.flow1, self.flow2])
        
    def align(self, x1:List[Tensor], x2:List[Tensor], label=None, meta=None):
        loss = 0
        flows=[]
        for i in range(len(self.flow)):
            x2[i], flow = self.flow[i](x1[i], x2[i])
            flows.append(flow)
            if label is not None:
                mask = F.interpolate(label.float(), size=x2[i].size()[-2:], mode='nearest')
                mask = mask==1
                loss += self.criterion(x1[i]*mask, x2[i]*mask)
        if label is not None:
            return x2, loss/len(self.flow)
        return x2, flows


    def forward(self, pre_data, post_data, label=None, val=False, names=None):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)
        if label is not None and not val:
            ann=label.clone().unsqueeze(1).detach()
            post_features, loss = self.align(pre_features, post_features, ann)
        else:
            post_features, flow = self.align(pre_features, post_features)
        output_building = self.decoder_building(pre_features)
        output_damage = self.decoder_damage(pre_features, post_features)
        output_building = self.aux_clf(output_building)
        output_building = F.interpolate(output_building, size=pre_data.size()[-2:], mode='bilinear')
        output_damage = self.main_clf(output_damage)
        output_damage = F.interpolate(output_damage, size=post_data.size()[-2:], mode='bilinear')
        if label is not None and not val:
            return output_building, output_damage, loss
        return output_building, output_damage


if __name__ == '__main__':
    import argparse
    import torch.onnx
    parser = argparse.ArgumentParser('validation model')
    parser.add_argument('-cfg', default='changedetection/configs/vssm1/vssm_small_224.yaml', type=str)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    args = parser.parse_args()
    
    config = get_config(args)
    pretrained='xxx' # set to your pretrained model path
    model = FlowMamba(output_building=2, output_damage=4, pretrained=pretrained, 
    # model = STMambaBDA(output_building=2, output_damage=4, pretrained=pretrained, 
                patch_size=config.MODEL.VSSM.PATCH_SIZE, 
                in_chans=config.MODEL.VSSM.IN_CHANS, 
                num_classes=config.MODEL.NUM_CLASSES, 
                depths=config.MODEL.VSSM.DEPTHS, 
                dims=config.MODEL.VSSM.EMBED_DIM, 
                # ===================
                ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
                ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
                ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
                ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
                ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
                ssm_conv=config.MODEL.VSSM.SSM_CONV,
                ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
                ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
                ssm_init=config.MODEL.VSSM.SSM_INIT,
                forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
                # ===================
                mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
                mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
                mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
                # ===================
                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                patch_norm=config.MODEL.VSSM.PATCH_NORM,
                norm_layer=config.MODEL.VSSM.NORM_LAYER,
                downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
                patchembed_version=config.MODEL.VSSM.PATCHEMBED,
                gmlp=config.MODEL.VSSM.GMLP,
                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                use_ossm=False)
    model = model.cuda().eval()
    x1 = torch.randn(1, 3, 256, 256).cuda()
    x2 = torch.randn(1, 3, 256, 256).cuda()
    output_building, output_damage = model(x1, x2)
    print(f"Output Building Shape: {output_building.shape}, Output Damage Shape: {output_damage.shape}")
    from thop import profile
    flops, params = profile(model, inputs=(x1, x2), verbose=False)
    print(f"FLOPs: {flops/1e9:.2f}G, Params: {params/1e6:.2f}M")
