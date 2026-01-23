import sys
sys.path.append('/home/guoxin/MambaCD')
import torch
import torch.nn as nn
import torch.nn.functional as F
from classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
from timm.models.layers import DropPath
# from .utils import LayerNorm, GRN
from changedetection.utils_func.freqfusion import FreqFusion
from changedetection.utils_func.upsample import DySample

class ChangeDecoder(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(ChangeDecoder, self).__init__()

        # Define the VSS Block for Spatio-temporal relationship modelling
        self.st_block_41 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'], use_ossm=kwargs['use_ossm']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_42 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'], use_ossm=kwargs['use_ossm']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),

        )
        self.st_block_43 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'], use_ossm=kwargs['use_ossm']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_31 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'], use_ossm=kwargs['use_ossm']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_32 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'], use_ossm=kwargs['use_ossm']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_33 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'], use_ossm=kwargs['use_ossm']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_21 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'], use_ossm=kwargs['use_ossm']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_22 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'], use_ossm=kwargs['use_ossm']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_23 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'], use_ossm=kwargs['use_ossm']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_11 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'], use_ossm=kwargs['use_ossm']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_12 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'], use_ossm=kwargs['use_ossm']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_13 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'], use_ossm=kwargs['use_ossm']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        # Fuse layer  
        self.fuse_layer_4 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())

        # Smooth layer
        self.smooth_layer_3 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_2 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_1 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        # upsample
        # self.up_3 = FreqFusion(128,128)
        # self.up_2 = FreqFusion(128,128)
        # self.up_1 = FreqFusion(128,128)
    
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_features, post_features):

        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features

        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

        '''
            Stage I
        '''
        p41 = self.st_block_41(torch.cat([pre_feat_4, post_feat_4], dim=1))
        B, C, H, W = pre_feat_4.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_42 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_42[:, :, :, ::2] = pre_feat_4  # Odd columns
        ct_tensor_42[:, :, :, 1::2] = post_feat_4  # Even columns
        p42 = self.st_block_42(ct_tensor_42)

        ct_tensor_43 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_43[:, :, :, 0:W] = pre_feat_4
        ct_tensor_43[:, :, :, W:] = post_feat_4
        p43 = self.st_block_43(ct_tensor_43)

        p4 = self.fuse_layer_4(torch.cat([p41, p42[:, :, :, ::2], p42[:, :, :, 1::2], p43[:, :, :, 0:W], p43[:, :, :, W:]], dim=1))
       

        '''
            Stage II
        '''
        p31 = self.st_block_31(torch.cat([pre_feat_3, post_feat_3], dim=1))
        B, C, H, W = pre_feat_3.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_32 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_32[:, :, :, ::2] = pre_feat_3  # Odd columns
        ct_tensor_32[:, :, :, 1::2] = post_feat_3  # Even columns
        p32 = self.st_block_32(ct_tensor_32)

        ct_tensor_33 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_33[:, :, :, 0:W] = pre_feat_3
        ct_tensor_33[:, :, :, W:] = post_feat_3
        p33 = self.st_block_33(ct_tensor_33)

        p3 = self.fuse_layer_3(torch.cat([p31, p32[:, :, :, ::2], p32[:, :, :, 1::2], p33[:, :, :, 0:W], p33[:, :, :, W:]], dim=1))
        p3 = self._upsample_add(p4, p3)
        # p3 = self.up_3(p3, p4)
        p3 = self.smooth_layer_3(p3)
       
        '''
            Stage III
        '''
        p21 = self.st_block_21(torch.cat([pre_feat_2, post_feat_2], dim=1))
        B, C, H, W = pre_feat_2.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_22 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_22[:, :, :, ::2] = pre_feat_2  # Odd columns
        ct_tensor_22[:, :, :, 1::2] = post_feat_2  # Even columns
        p22 = self.st_block_22(ct_tensor_22)

        ct_tensor_23 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_23[:, :, :, 0:W] = pre_feat_2
        ct_tensor_23[:, :, :, W:] = post_feat_2
        p23 = self.st_block_23(ct_tensor_23)

        p2 = self.fuse_layer_2(torch.cat([p21, p22[:, :, :, ::2], p22[:, :, :, 1::2], p23[:, :, :, 0:W], p23[:, :, :, W:]], dim=1))
        p2 = self._upsample_add(p3, p2)
        # p2 = self.up_2(p2, p3)
        p2 = self.smooth_layer_2(p2)
       
        '''
            Stage IV
        '''
        p11 = self.st_block_11(torch.cat([pre_feat_1, post_feat_1], dim=1))
        B, C, H, W = pre_feat_1.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_12 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_12[:, :, :, ::2] = pre_feat_1  # Odd columns
        ct_tensor_12[:, :, :, 1::2] = post_feat_1  # Even columns
        p12 = self.st_block_12(ct_tensor_12)

        ct_tensor_13 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_13[:, :, :, 0:W] = pre_feat_1
        ct_tensor_13[:, :, :, W:] = post_feat_1
        p13 = self.st_block_13(ct_tensor_13)

        p1 = self.fuse_layer_1(torch.cat([p11, p12[:, :, :, ::2], p12[:, :, :, 1::2], p13[:, :, :, 0:W], p13[:, :, :, W:]], dim=1))

        p1 = self._upsample_add(p2, p1)
        # p1 = self.up_1(p1, p2)
        p1 = self.smooth_layer_1(p1)
        return p1

class ChangeDecoderUser(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(ChangeDecoderUser, self).__init__()
        '''
        st_block_(n)1: spatio-temporal parallel modeling, torch.cat(x1,x2);
        st_block_(n)2: spatio-temporal cross modeling, x1 and x2 sequences assignd to odd and even columns;
        st_block_(n)3: spatio-temporal sequential modeling, x1 and x2 sequences scaned in order;
        '''
        # Define the VSS Block for Spatio-temporal relationship modelling
        self.st_block_41 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_42 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),

        )
        self.st_block_43 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_31 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_32 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_33 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_21 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_22 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_23 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_11 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_12 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_13 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        # Fuse layer  
        self.fuse_layer_4 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())

        # Smooth layer
        self.smooth_layer_3 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_2 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_1 = ResBlock(in_channels=128, out_channels=128, stride=1) 
    
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_features, post_features):

        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features

        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

        '''
            Stage I
        '''
        p41 = self.st_block_41(torch.cat([pre_feat_4, post_feat_4], dim=1)) # parallel modeling
        B, C, H, W = pre_feat_4.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_42 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_42[:, :, :, ::2] = pre_feat_4  # Odd columns
        ct_tensor_42[:, :, :, 1::2] = post_feat_4  # Even columns
        p42 = self.st_block_42(ct_tensor_42) # cross modeling

        ct_tensor_43 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_43[:, :, :, 0:W] = pre_feat_4
        ct_tensor_43[:, :, :, W:] = post_feat_4
        p43 = self.st_block_43(ct_tensor_43) # squential modeling

        p4 = self.fuse_layer_4(torch.cat([p41, p42[:, :, :, ::2], p42[:, :, :, 1::2], p43[:, :, :, 0:W], p43[:, :, :, W:]], dim=1))
       

        '''
            Stage II
        '''
        p31 = self.st_block_31(torch.cat([pre_feat_3, post_feat_3], dim=1))
        B, C, H, W = pre_feat_3.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_32 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_32[:, :, :, ::2] = pre_feat_3  # Odd columns
        ct_tensor_32[:, :, :, 1::2] = post_feat_3  # Even columns
        p32 = self.st_block_32(ct_tensor_32)

        ct_tensor_33 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_33[:, :, :, 0:W] = pre_feat_3
        ct_tensor_33[:, :, :, W:] = post_feat_3
        p33 = self.st_block_33(ct_tensor_33)

        p3 = self.fuse_layer_3(torch.cat([p31, p32[:, :, :, ::2], p32[:, :, :, 1::2], p33[:, :, :, 0:W], p33[:, :, :, W:]], dim=1))
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3(p3)
       
        '''
            Stage III
        '''
        p21 = self.st_block_21(torch.cat([pre_feat_2, post_feat_2], dim=1))
        B, C, H, W = pre_feat_2.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_22 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_22[:, :, :, ::2] = pre_feat_2  # Odd columns
        ct_tensor_22[:, :, :, 1::2] = post_feat_2  # Even columns
        p22 = self.st_block_22(ct_tensor_22)

        ct_tensor_23 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_23[:, :, :, 0:W] = pre_feat_2
        ct_tensor_23[:, :, :, W:] = post_feat_2
        p23 = self.st_block_23(ct_tensor_23)

        p2 = self.fuse_layer_2(torch.cat([p21, p22[:, :, :, ::2], p22[:, :, :, 1::2], p23[:, :, :, 0:W], p23[:, :, :, W:]], dim=1))
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2(p2)
       
        '''
            Stage IV
        '''
        p11 = self.st_block_11(torch.cat([pre_feat_1, post_feat_1], dim=1))
        B, C, H, W = pre_feat_1.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_12 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_12[:, :, :, ::2] = pre_feat_1  # Odd columns
        ct_tensor_12[:, :, :, 1::2] = post_feat_1  # Even columns
        p12 = self.st_block_12(ct_tensor_12)

        ct_tensor_13 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_13[:, :, :, 0:W] = pre_feat_1
        ct_tensor_13[:, :, :, W:] = post_feat_1
        p13 = self.st_block_13(ct_tensor_13)

        p1 = self.fuse_layer_1(torch.cat([p11, p12[:, :, :, ::2], p12[:, :, :, 1::2], p13[:, :, :, 0:W], p13[:, :, :, W:]], dim=1))

        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1(p1)

        return p1, [p1,p2,p3,p4]

class ChangeDecoder3D(nn.Module):
    def __init__(self, encoder_dims, norm_layer, **kwargs):
        super(ChangeDecoder3D, self).__init__()

        # Define the 3D Conv Block for Spatio-temporal relationship modelling
        self.st_block_4 = nn.Sequential(
            nn.Conv3d(in_channels=encoder_dims[-1], out_channels=128, kernel_size=(2, 1, 1)),
            nn.BatchNorm3d(128),
            # norm_layer(128),
            nn.ReLU(inplace=True)
        )
        self.st_block_3 = nn.Sequential(
            nn.Conv3d(in_channels=encoder_dims[-2], out_channels=128, kernel_size=(2, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.st_block_2 = nn.Sequential(
            nn.Conv3d(in_channels=encoder_dims[-3], out_channels=128, kernel_size=(2, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.st_block_1 = nn.Sequential(
            nn.Conv3d(in_channels=encoder_dims[-4], out_channels=128, kernel_size=(2, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )

        # Fuse layer  
        self.fuse_layer_4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_1 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1),
                                          nn.BatchNorm2d(128), nn.ReLU())

        # Smooth layer
        self.smooth_layer_3 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_2 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_1 = ResBlock(in_channels=128, out_channels=128, stride=1) 
    
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_features, post_features, out_concat=False):

        # Stack pre and post features along a new dimension
        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features
        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

        # Create 5D tensors for 3D convolution
        feat_4 = torch.stack([pre_feat_4, post_feat_4], dim=2)  # Shape: (B, C, 2, H, W)
        feat_3 = torch.stack([pre_feat_3, post_feat_3], dim=2)
        feat_2 = torch.stack([pre_feat_2, post_feat_2], dim=2)
        feat_1 = torch.stack([pre_feat_1, post_feat_1], dim=2)

        '''
            Stage I
        '''
        p4 = self.st_block_4(feat_4).squeeze(2)  # Shape: (B, 128, H, W)
        p4 = self.fuse_layer_4(p4)

        '''
            Stage II
        '''
        p3 = self.st_block_3(feat_3).squeeze(2)
        p3 = self.fuse_layer_3(p3)
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3(p3)
       
        '''
            Stage III
        '''
        p2 = self.st_block_2(feat_2).squeeze(2)
        p2 = self.fuse_layer_2(p2)
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2(p2)
       
        '''
            Stage IV
        '''
        p1 = self.st_block_1(feat_1).squeeze(2)
        p1 = self.fuse_layer_1(p1)
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1(p1)

        if out_concat:
            return p1, [p1, p2, p3, p4]
        return p1


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

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

class BDADecoder(ChangeDecoder):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, pretrained=None,**kwargs):
        super(BDADecoder, self).__init__(encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs)
        '''
        st_block_(n)1: spatio-temporal parallel modeling, torch.cat(x1,x2);
        st_block_(n)2: spatio-temporal cross modeling, x1 and x2 sequences assignd to odd and even columns;
        st_block_(n)3: spatio-temporal sequential modeling, x1 and x2 sequences scaned in order;
        '''
        self.st_block_seg_4 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_seg_3 = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_seg_2 = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_seg_1 = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.trans_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2], out_channels=128),
                                        nn.BatchNorm2d(128), nn.ReLU())
        self.trans_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3], out_channels=128),
                                        nn.BatchNorm2d(128), nn.ReLU())
        self.trans_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4], out_channels=128),
                                        nn.BatchNorm2d(128), nn.ReLU())
        
        # smooth layer
        self.smooth_layer_seg_3 = ResBlock(in_channels=128, out_channels=128, stride=1)
        self.smooth_layer_seg_2 = ResBlock(in_channels=128, out_channels=128, stride=1)
        self.smooth_layer_seg_1 = ResBlock(in_channels=128, out_channels=128, stride=1)
        self.smooth_layer_seg_0 = ResBlock(in_channels=128, out_channels=128, stride=1)

        # bda fuse layer
        self.bda_fuse_layer_4 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=256, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.bda_fuse_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=256, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.bda_fuse_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=256, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.bda_fuse_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=256, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())        
        # self.bda_fuse_layer_4 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128, out_channels=128),
        #                                   nn.BatchNorm2d(128), nn.ReLU())
        # self.bda_fuse_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128, out_channels=128),
        #                                   nn.BatchNorm2d(128), nn.ReLU())
        # self.bda_fuse_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128, out_channels=128),
        #                                   nn.BatchNorm2d(128), nn.ReLU())
        # self.bda_fuse_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128, out_channels=128),
        #                                   nn.BatchNorm2d(128), nn.ReLU())        
        self.load_pretrained(pretrained)
    def load_pretrained(self, ckpt=None, key="bda_decoder"):
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False) if _ckpt.get(key) is not None else print(f"Failed to load {key}")
            # incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(f'totol number of incompatibleKeys: {len(incompatibleKeys)}')
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, pre_features, post_features):

        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features

        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

        '''
            Stage I
        '''
        # loc features
        diff_st4 = self.st_block_seg_4(pre_feat_4-post_feat_4)

        # dmg features
        p41 = self.st_block_41(torch.cat([pre_feat_4, post_feat_4], dim=1)) # parallel modeling
        B, C, H, W = pre_feat_4.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_42 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_42[:, :, :, ::2] = pre_feat_4  # Odd columns
        ct_tensor_42[:, :, :, 1::2] = post_feat_4  # Even columns
        p42 = self.st_block_42(ct_tensor_42) # cross modeling

        ct_tensor_43 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_43[:, :, :, 0:W] = pre_feat_4
        ct_tensor_43[:, :, :, W:] = post_feat_4
        p43 = self.st_block_43(ct_tensor_43) # squential modeling

        post_st4 = self.fuse_layer_4(torch.cat([p41, p42[:, :, :, ::2], p42[:, :, :, 1::2], p43[:, :, :, 0:W], p43[:, :, :, W:]], dim=1))
        
        
        # p4 = self.bda_fuse_layer_4(pre_st4 - post_st4)
        p4 = self.bda_fuse_layer_4(torch.cat([diff_st4, post_st4], dim=1))

        '''
            Stage II
        '''
        # loc features
        diff_st3 = self.trans_layer_3(pre_feat_3-post_feat_3)
        diff_st3 = self._upsample_add(diff_st4, diff_st3)
        diff_st3 = self.smooth_layer_seg_3(diff_st3)
        diff_st3 = self.st_block_seg_3(diff_st3)   

        # dmg features
        p31 = self.st_block_31(torch.cat([pre_feat_3, post_feat_3], dim=1))
        B, C, H, W = pre_feat_3.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_32 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_32[:, :, :, ::2] = pre_feat_3  # Odd columns
        ct_tensor_32[:, :, :, 1::2] = post_feat_3  # Even columns
        p32 = self.st_block_32(ct_tensor_32)

        ct_tensor_33 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_33[:, :, :, 0:W] = pre_feat_3
        ct_tensor_33[:, :, :, W:] = post_feat_3
        p33 = self.st_block_33(ct_tensor_33)

        p3 = self.fuse_layer_3(torch.cat([p31, p32[:, :, :, ::2], p32[:, :, :, 1::2], p33[:, :, :, 0:W], p33[:, :, :, W:]], dim=1))
        p3 = self._upsample_add(p4, p3)
        post_st3 = self.smooth_layer_3(p3)

        # fusion features
        # p3 = self.bda_fuse_layer_3(pre_st3 - post_st3)
        p3 = self.bda_fuse_layer_3(torch.cat([diff_st3, post_st3], dim=1))
       
        '''
            Stage III
        '''
        # loc features
        diff_st2 = self.trans_layer_2(pre_feat_2-post_feat_2)
        diff_st2 = self._upsample_add(diff_st3, diff_st2)
        diff_st2 = self.smooth_layer_seg_2(diff_st2)
        diff_st2 = self.st_block_seg_2(diff_st2)

        # dmg features
        p21 = self.st_block_21(torch.cat([pre_feat_2, post_feat_2], dim=1))
        B, C, H, W = pre_feat_2.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_22 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_22[:, :, :, ::2] = pre_feat_2  # Odd columns
        ct_tensor_22[:, :, :, 1::2] = post_feat_2  # Even columns
        p22 = self.st_block_22(ct_tensor_22)

        ct_tensor_23 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_23[:, :, :, 0:W] = pre_feat_2
        ct_tensor_23[:, :, :, W:] = post_feat_2
        p23 = self.st_block_23(ct_tensor_23)

        p2 = self.fuse_layer_2(torch.cat([p21, p22[:, :, :, ::2], p22[:, :, :, 1::2], p23[:, :, :, 0:W], p23[:, :, :, W:]], dim=1))
        p2 = self._upsample_add(p3, p2)
        post_st2 = self.smooth_layer_2(p2)

        # fusion features
        # p2 = self.bda_fuse_layer_2(pre_st2 - post_st2)
        p2 = self.bda_fuse_layer_2(torch.cat([diff_st2, post_st2], dim=1))
       
        '''
            Stage IV
        '''
        # loc features
        diff_st1 = self.trans_layer_1(pre_feat_1-post_feat_1)
        diff_st1 = self._upsample_add(diff_st2, diff_st1)
        diff_st1 = self.smooth_layer_seg_1(diff_st1)
        diff_st1 = self.st_block_seg_1(diff_st1)

        # dmg features
        p11 = self.st_block_11(torch.cat([pre_feat_1, post_feat_1], dim=1))
        B, C, H, W = pre_feat_1.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_12 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_12[:, :, :, ::2] = pre_feat_1  # Odd columns
        ct_tensor_12[:, :, :, 1::2] = post_feat_1  # Even columns
        p12 = self.st_block_12(ct_tensor_12)

        ct_tensor_13 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_13[:, :, :, 0:W] = pre_feat_1
        ct_tensor_13[:, :, :, W:] = post_feat_1
        p13 = self.st_block_13(ct_tensor_13)

        p1 = self.fuse_layer_1(torch.cat([p11, p12[:, :, :, ::2], p12[:, :, :, 1::2], p13[:, :, :, 0:W], p13[:, :, :, W:]], dim=1))

        p1 = self._upsample_add(p2, p1)
        post_st1 = self.smooth_layer_1(p1)

        # fusion features
        # p1 = self.bda_fuse_layer_1(pre_st1 - post_st1)
        p1 = self.bda_fuse_layer_1(torch.cat([diff_st1, post_st1], dim=1))

        return diff_st1, p1

class MixDecoder(BDADecoder):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, pretrained=None,**kwargs):
        super(MixDecoder, self).__init__(encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs)
        self.st_block_pre_seg_1 = self.st_block_seg_1
        self.st_block_post_seg_1 = self.st_block_seg_1
        self.st_block_cd_1 = self.st_block_seg_1
        self.st_block_pre_seg_2 = self.st_block_seg_2
        self.st_block_post_seg_2 = self.st_block_seg_2
        self.st_block_cd_2 = self.st_block_seg_2
        self.st_block_pre_seg_3 = self.st_block_seg_3
        self.st_block_post_seg_3 = self.st_block_seg_3
        self.st_block_cd_3 = self.st_block_seg_3
        self.st_block_pre_seg_4 = self.st_block_seg_4
        self.st_block_post_seg_4 = self.st_block_seg_4
        self.st_block_cd_4 = self.st_block_seg_3
        self.trans_layer_pre_1 = self.trans_layer_1
        self.trans_layer_post_1 = self.trans_layer_1
        self.trans_layer_pre_2 = self.trans_layer_2
        self.trans_layer_post_2 = self.trans_layer_2
        self.trans_layer_pre_3 = self.trans_layer_3
        self.trans_layer_post_3 = self.trans_layer_3
        self.smooth_layer_pre_seg_1 = self.smooth_layer_1
        self.smooth_layer_post_seg_1 = self.smooth_layer_1
        self.smooth_layer_cd_1 = self.smooth_layer_1
        self.smooth_layer_pre_seg_2 = self.smooth_layer_2
        self.smooth_layer_post_seg_2 = self.smooth_layer_2
        self.smooth_layer_cd_2 = self.smooth_layer_2
        self.smooth_layer_pre_seg_3 = self.smooth_layer_3
        self.smooth_layer_post_seg_3 = self.smooth_layer_3
        self.smooth_layer_cd_3 = self.smooth_layer_3
        del self.st_block_seg_1, self.st_block_seg_2, self.st_block_seg_3, self.st_block_seg_4
        del self.trans_layer_1, self.trans_layer_2, self.trans_layer_3
        del self.smooth_layer_1, self.smooth_layer_2, self.smooth_layer_3
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="decoder"):
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            # print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False) if _ckpt.get(key) is not None else print(f"Failed to load {key}")
            # incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(f"Successfully update ckpt {ckpt}[{key}]")
            print(f'totol number of incompatibleKeys: {len(incompatibleKeys)}')
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, pre_features, post_features):
        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features
        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

        '''
            Stage I
        '''
        # loc_features
        pre_st4 = self.st_block_pre_seg_4(pre_feat_4)
        post_st4 = self.st_block_post_seg_4(post_feat_4)
        diff_st4 = self.st_block_cd_4(pre_st4-post_st4)

        # dmg_features
        p41 = self.st_block_41(torch.cat([pre_feat_4, post_feat_4], dim=1)) # parallel modeling
        B, C, H, W = pre_feat_4.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_42 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_42[:, :, :, ::2] = pre_feat_4  # Odd columns
        ct_tensor_42[:, :, :, 1::2] = post_feat_4  # Even columns
        p42 = self.st_block_42(ct_tensor_42) # cross modeling

        ct_tensor_43 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_43[:, :, :, 0:W] = pre_feat_4
        ct_tensor_43[:, :, :, W:] = post_feat_4
        p43 = self.st_block_43(ct_tensor_43) # squential modeling

        post_st4 = self.fuse_layer_4(torch.cat([p41, p42[:, :, :, ::2], p42[:, :, :, 1::2], p43[:, :, :, 0:W], p43[:, :, :, W:]], dim=1))

        # p4 = self.bda_fuse_layer_4(pre_st4 - post_st4)
        p4 = self.bda_fuse_layer_4(torch.cat([diff_st4, post_st4], dim=1))

        '''
            Stage II
        '''
        # loc_features
        pre_st3 = self.trans_layer_pre_3(pre_feat_3)
        pre_st3 = self._upsample_add(pre_st4, pre_st3)
        pre_st3 = self.smooth_layer_pre_seg_3(pre_st3)
        pre_st3 = self.st_block_pre_seg_3(pre_st3)

        post_st3 = self.trans_layer_post_3(post_feat_3)
        post_st3 = self._upsample_add(post_st4, post_st3)
        post_st3 = self.smooth_layer_post_seg_3(post_st3)
        post_st3 = self.st_block_post_seg_3(post_st3)

        diff_st3 = self.st_block_cd_3(pre_st3-post_st3)

        # dmg_features
        p31 = self.st_block_31(torch.cat([pre_feat_3, post_feat_3], dim=1)) # parallel modeling
        B, C, H, W = pre_feat_3.size()
        ct_tensor_32 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_32[:, :, :, ::2] = pre_feat_3  # Odd columns
        ct_tensor_32[:, :, :, 1::2] = post_feat_3  # Even columns
        p32 = self.st_block_32(ct_tensor_32) # cross modeling
        ct_tensor_33 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_33[:, :, :, 0:W] = pre_feat_3
        ct_tensor_33[:, :, :, W:] = post_feat_3
        p33 = self.st_block_33(ct_tensor_33) # squential modeling

        p3 = self.fuse_layer_3(torch.cat([p31, p32[:, :, :, ::2], p32[:, :, :, 1::2], p33[:, :, :, 0:W], p33[:, :, :, W:]], dim=1))
        p3 = self._upsample_add(p4, p3)
        post_st3 = self.smooth_layer_cd_3(p3)

        # fusion features
        # p3 = self.bda_fuse_layer_3(pre_st3 - post_st3)
        p3 = self.bda_fuse_layer_3(torch.cat([diff_st3, post_st3], dim=1))

        '''
            Stage III
        '''
        # loc_features
        pre_st2 = self.trans_layer_pre_2(pre_feat_2)
        pre_st2 = self._upsample_add(pre_st3, pre_st2)
        pre_st2 = self.smooth_layer_pre_seg_2(pre_st2)
        pre_st2 = self.st_block_pre_seg_2(pre_st2)

        post_st2 = self.trans_layer_post_2(post_feat_2)
        post_st2 = self._upsample_add(post_st3, post_st2)
        post_st2 = self.smooth_layer_post_seg_2(post_st2)
        post_st2 = self.st_block_post_seg_2(post_st2)
        diff_st2 = self.st_block_cd_2(pre_st2-post_st2)


        # dmg_features
        p21 = self.st_block_21(torch.cat([pre_feat_2, post_feat_2], dim=1))
        B, C, H, W = pre_feat_2.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_22 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_22[:, :, :, ::2] = pre_feat_2  # Odd columns
        ct_tensor_22[:, :, :, 1::2] = post_feat_2  # Even columns
        p22 = self.st_block_22(ct_tensor_22)

        ct_tensor_23 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_23[:, :, :, 0:W] = pre_feat_2
        ct_tensor_23[:, :, :, W:] = post_feat_2
        p23 = self.st_block_23(ct_tensor_23)

        p2 = self.fuse_layer_2(torch.cat([p21, p22[:, :, :, ::2], p22[:, :, :, 1::2], p23[:, :, :, 0:W], p23[:, :, :, W:]], dim=1))
        p2 = self._upsample_add(p3, p2)
        post_st2 = self.smooth_layer_cd_2(p2)

        # fusion features
        p2 = self.bda_fuse_layer_2(torch.cat([diff_st2, post_st2], dim=1))

        '''
            Stage IV
        '''
        # loc_features
        pre_st1 = self.trans_layer_pre_1(pre_feat_1)
        pre_st1 = self._upsample_add(pre_st2, pre_st1)
        pre_st1 = self.smooth_layer_pre_seg_1(pre_st1)
        pre_st1 = self.st_block_pre_seg_1(pre_st1)

        post_st1 = self.trans_layer_post_1(post_feat_1)
        post_st1 = self._upsample_add(post_st2, post_st1)
        post_st1 = self.smooth_layer_post_seg_1(post_st1)
        post_st1 = self.st_block_post_seg_1(post_st1)
        
        diff_st1 = self.st_block_cd_1(pre_st1-post_st1)

        # dmg_features
        p11 = self.st_block_11(torch.cat([pre_feat_1, post_feat_1], dim=1))
        B, C, H, W = pre_feat_1.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_12 = torch.empty(B, C, H, 2*W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_12[:, :, :, ::2] = pre_feat_1  # Odd columns
        ct_tensor_12[:, :, :, 1::2] = post_feat_1  # Even columns
        p12 = self.st_block_12(ct_tensor_12)

        ct_tensor_13 = torch.empty(B, C, H, 2*W).cuda()
        ct_tensor_13[:, :, :, 0:W] = pre_feat_1
        ct_tensor_13[:, :, :, W:] = post_feat_1
        p13 = self.st_block_13(ct_tensor_13)

        p1 = self.fuse_layer_1(torch.cat([p11, p12[:, :, :, ::2], p12[:, :, :, 1::2], p13[:, :, :, 0:W], p13[:, :, :, W:]], dim=1))

        p1 = self._upsample_add(p2, p1)
        post_st1 = self.smooth_layer_cd_1(p1)

        # fusion features
        p1 = self.bda_fuse_layer_1(torch.cat([diff_st1, post_st1], dim=1))

        
        return pre_st1, p1, [p1,p2,p3,p4]

class ConvnextBlock(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    
class ChangeUP(ChangeDecoder):
    def __init__(self, encoder_dims, channel_first=False, norm_layer=nn.LayerNorm, ssm_act_layer=nn.SiLU, mlp_act_layer=nn.GELU, **kwargs):
        super().__init__(encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs)
        pass

    def _upsample_add(self, x, y):
        upsample = DySample(x.size(1)).cuda()
        return upsample(x) + y

if __name__ == "__main__":
    # Test ChangeDecoder
    pre_features = [torch.rand(1, 128, 64, 64), torch.rand(1, 128, 32, 32), torch.rand(1, 128, 16, 16), torch.rand(1, 128, 8, 8)]
    post_features = [torch.rand(1, 128, 64, 64), torch.rand(1, 128, 32, 32), torch.rand(1, 128, 16, 16), torch.rand(1, 128, 8, 8)]
    decoder = ChangeDecoder3D([128, 128, 128, 128], channel_first=True, norm_layer=nn.BatchNorm2d, ssm_act_layer=nn.ReLU, mlp_act_layer=nn.ReLU)
    out = decoder(pre_features, post_features)
    print(out.shape)