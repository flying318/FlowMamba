import sys
sys.path.append('changedetection/models/comparative')
from collections import OrderedDict
from typing import Tuple
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_
from rsmamba.util import Fuse_Block, Decoder_Block, VSSBlock, Permute
# from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

class RSM_CD(nn.Module):
    def __init__(
        self, 
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
        depths=[2, 2, 9, 2], 
        dims=[96, 192, 384, 768], 
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        # =========================
        drop_path_rate=0.2, 
        patch_norm=True, 
        norm_layer="LN",
        use_checkpoint=False,  
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        if isinstance(norm_layer, str) and norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]

        if isinstance(ssm_act_layer, str) and ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]

        if isinstance(mlp_act_layer, str) and mlp_act_layer.lower() in ["silu", "gelu", "relu"]:
            mlp_act_layer: nn.Module = _ACTLAYERS[mlp_act_layer.lower()]

        _make_patch_embed = self._make_patch_embed_v2
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer)

        _make_downsample = self._make_downsample_v3

        # self.encoder_layers = [nn.ModuleList()] * self.num_layers
        self.encoder_layers = []
        self.fuse_layers = []
        self.decoder_layers = []

        for i_layer in range(self.num_layers):
            # downsample = _make_downsample(
            #     self.dims[i_layer], 
            #     self.dims[i_layer + 1], 
            #     norm_layer=norm_layer,
            # ) if (i_layer < self.num_layers - 1) else nn.Identity()

            downsample = _make_downsample(
                self.dims[i_layer - 1], 
                self.dims[i_layer], 
                norm_layer=norm_layer,
            ) if (i_layer != 0) else nn.Identity()  # ZSJ 修改为i_layer != 0，也就是第一层不下采样，和论文的图保持一致，也方便我取出每个尺度处理好的特征

            self.encoder_layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            ))
            self.fuse_layers.append(Fuse_Block(in_channel=self.dims[i_layer]))
            if i_layer != 0:
                self.decoder_layers.append(Decoder_Block(in_channel=self.dims[i_layer], out_channel=self.dims[i_layer-1]))

        self.encoder_block1, self.encoder_block2, self.encoder_block3, self.encoder_block4 = self.encoder_layers
        self.fuse_block1, self.fuse_block2, self.fuse_block3, self.fuse_block4 = self.fuse_layers
        self.decoder_block1, self.decoder_block2, self.decoder_block3 = self.decoder_layers

        # self.classifier = nn.Sequential(OrderedDict(
        #     norm=norm_layer(self.num_features), # B,H,W,C
        #     permute=Permute(0, 3, 1, 2),
        #     avgpool=nn.AdaptiveAvgPool2d(1),
        #     flatten=nn.Flatten(1),
        #     head=nn.Linear(self.num_features, num_classes),
        # ))
        
        self.upsample_x4 = nn.Sequential(
            nn.Conv2d(self.dims[0], self.dims[0]//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.dims[0]//2),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.dims[0]//2, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.conv_out_change = nn.Conv2d(8, num_classes, kernel_size=7, stride=1, padding=3)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
        assert patch_size == 4
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            (Permute(0, 2, 3, 1) if patch_norm else nn.Identity()),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (Permute(0, 3, 1, 2) if patch_norm else nn.Identity()),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
            ))
        
        return nn.Sequential(OrderedDict(
            # ZSJ 把downsample放到前面来，方便我取出encoder中每个尺度处理好的图像，而不是刚刚下采样完的图像
            downsample=downsample,
            blocks=nn.Sequential(*blocks,),
        ))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1 = self.patch_embed(x1)
        x2 = self.patch_embed(x2)

        x1_1 = self.encoder_block1(x1)
        x1_2 = self.encoder_block2(x1_1)
        x1_3 = self.encoder_block3(x1_2)
        x1_4 = self.encoder_block4(x1_3)  # b,h,w,c

        x2_1 = self.encoder_block1(x2)
        x2_2 = self.encoder_block2(x2_1)
        x2_3 = self.encoder_block3(x2_2)
        x2_4 = self.encoder_block4(x2_3)  # b,h,w,c

        fuse_1 = self.fuse_block1(x1_1, x2_1)
        fuse_2 = self.fuse_block2(x1_2, x2_2)
        fuse_3 = self.fuse_block3(x1_3, x2_3)
        fuse_4 = self.fuse_block4(x1_4, x2_4)

        decode_3 = self.decoder_block3(fuse_4, fuse_3)
        decode_2 = self.decoder_block2(decode_3, fuse_2)
        decode_1 = self.decoder_block1(decode_2, fuse_1)

        output = self.upsample_x4(decode_1)
        output = self.conv_out_change(output)

        return output


class RSM_BDA(RSM_CD):
    def __init__(self, out_loc=int, out_dmg=int, **kwargs):
        super().__init__(num_classes=out_loc, **kwargs)
        self.cls_seg = nn.Conv2d(8, out_loc, kernel_size=7, stride=1, padding=3)
        self.cls_dmg = nn.Conv2d(8, out_dmg, kernel_size=7, stride=1, padding=3)
        self.decoder_seg_1, self.decoder_seg_2, self.decoder_seg_3 = self.decoder_layers
        self.decoder_dmg_1, self.decoder_dmg_2, self.decoder_dmg_3 = self.decoder_layers
        del self.decoder_layers, self.decoder_block1, self.decoder_block2, self.decoder_block3, \
            self.conv_out_change

    def permute(self, input: Tuple[torch.Tensor]):
        assert len(input) == 4, "input must be a tuple of 4 tensors"
        x1, x2, x3, x4 = input
        out1 = rearrange(x1, "b h w c -> b c h w").contiguous()
        out2 = rearrange(x2, "b h w c -> b c h w").contiguous()
        out3 = rearrange(x3, "b h w c -> b c h w").contiguous()
        out4 = rearrange(x4, "b h w c -> b c h w").contiguous()
        return out1, out2, out3, out4

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1 = self.patch_embed(x1)
        x2 = self.patch_embed(x2)

        x1_1 = self.encoder_block1(x1)
        x1_2 = self.encoder_block2(x1_1)
        x1_3 = self.encoder_block3(x1_2)
        x1_4 = self.encoder_block4(x1_3)  # b,h,w,c

        x2_1 = self.encoder_block1(x2)
        x2_2 = self.encoder_block2(x2_1)
        x2_3 = self.encoder_block3(x2_2)
        x2_4 = self.encoder_block4(x2_3)  # b,h,w,c

        # segmentation decoder
        seg_1, seg_2, seg_3, seg_4 = self.permute((x1_1.clone(), x1_2.clone(), 
                                                   x1_3.clone(), x1_4.clone()))
        feat_seg_3 = self.decoder_seg_3(seg_4, seg_3)
        feat_seg_2 = self.decoder_seg_2(feat_seg_3, seg_2)
        feat_seg_1 = self.decoder_seg_1(feat_seg_2, seg_1)
        feat_seg = self.upsample_x4(feat_seg_1)

        # change detection decoder
        fuse_1 = self.fuse_block1(x1_1, x2_1)
        fuse_2 = self.fuse_block2(x1_2, x2_2)
        fuse_3 = self.fuse_block3(x1_3, x2_3)
        fuse_4 = self.fuse_block4(x1_4, x2_4)

        feat_dmg_3 = self.decoder_dmg_3(fuse_4, fuse_3)
        feat_dmg_2 = self.decoder_dmg_2(feat_dmg_3, fuse_2)
        feat_dmg_1 = self.decoder_dmg_1(feat_dmg_2, fuse_1)
        feat_dmg = self.upsample_x4(feat_dmg_1)

        seg_logits = self.cls_seg(feat_seg)
        dmg_logits = self.cls_dmg(feat_dmg)
        return seg_logits, dmg_logits


if __name__ == "__main__":
    # model = RSM_CD(num_classes=2).to('cuda:1')
    model = RSM_BDA(2, 5).to('cuda:0')
    x1 = torch.randn(8, 3, 256, 256).to('cuda:0')
    x2 = torch.randn(8, 3, 256, 256).to('cuda:0')
    out = model(x1, x2)
    print(out.shape)

    

