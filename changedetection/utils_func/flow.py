import sys
sys.path.append('/home/guoxin/MambaCD')

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from timm.models.layers import DropPath, trunc_normal_
import math
from classification.models.vmamba import VSSBlock, Permute
from scipy.ndimage import label, find_objects
import numpy as np
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.vss_block1 = nn.Sequential(
            VSSBlock(hidden_dim=hidden_features, drop_path=0.1, norm_layer=kwargs['norm_layer'], channel_first=kwargs['channel_first'],
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=kwargs['ssm_act_layer'],
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=kwargs['mlp_act_layer'], mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'], use_ossm=kwargs['use_ossm']),
            )
        self.vss_block2 = nn.Sequential(
            VSSBlock(hidden_dim=hidden_features, drop_path=0.1, norm_layer=kwargs['norm_layer'], channel_first=kwargs['channel_first'],
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=kwargs['ssm_act_layer'],
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=kwargs['mlp_act_layer'], mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'], use_ossm=kwargs['use_ossm']),
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        # x = self.dwconv(x, H, W)
        x = self.vss_block1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.vss_block2(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class AttentionRealCrossChannel(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.cond = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W, cond):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
        kv = self.kv(cond).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).reshape(B, C, N).transpose(1, 2)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_output, attn_weights = self.multihead_attn(query, key, value)
        attn_output = self.dropout(attn_output)
        output = self.norm(attn_output + query)
        # return output, attn_weights
        return output

class flowmlp(nn.Module):
    def __init__(self, inplane, **kwargs):
        super(flowmlp, self).__init__()
        channel_first = kwargs['channel_first']
        self.gelu = nn.GELU()
        self.conv = nn.Conv2d(inplane, inplane, 3, 1, 1, bias=True, groups=inplane)
        self.vss_block = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=inplane, drop_path=0.1, norm_layer=kwargs['norm_layer'], channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=kwargs['ssm_act_layer'],
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=kwargs['mlp_act_layer'], mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'], use_ossm=kwargs['use_ossm']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity())

    def forward(self, x):
        x = self.conv(x)
        x = self.vss_block(x)
        x = self.gelu(x)
        return x
    # def forward(self, x):
    #     x = self.Conv_enlarge(x)
    #     x = self.dwconv(x)
    #     # x = self.vss_block(x)
    #     x = self.gelu(x)
    #     x = self.Conv_shrink(x)

    #     return x

class FlowN(nn.Module):
    def __init__(self, inplane, h, w, **kwargs):
        """
        implementation of diffflow
        :param inplane:
        :param norm_layer:
        """
        super(FlowN, self).__init__()
        self.flowmlp1 = flowmlp(inplane, **kwargs)
        self.flowmlp2 = flowmlp(inplane, **kwargs)
        self.flow_make1 = nn.Conv2d(inplane *2 , 2, kernel_size=3, padding=1, bias=False)
        self.flow_make2 = nn.Conv2d(inplane *2 , 2, kernel_size=3, padding=1, bias=False)
        self.channel_align = CrossAttention(inplane)
        # self.channel_align = AttentionRealCrossChannel(inplane)
        self.mlp = Mlp(inplane, inplane, inplane, act_layer=nn.GELU, drop=0., **kwargs)
        # self.vss_block = nn.Sequential(
        #     Permute(0, 2, 3, 1) if not kwargs['channel_first'] else nn.Identity(),
        #     nn.LayerNorm(inplane),
        #     VSSBlock(hidden_dim=inplane, drop_path=0.1, norm_layer=kwargs['norm_layer'], channel_first=kwargs['channel_first'],
        #         ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=kwargs['ssm_act_layer'],
        #         ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
        #         forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=kwargs['mlp_act_layer'], mlp_drop_rate=kwargs['mlp_drop_rate'],
        #         gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'], use_ossm=kwargs['use_ossm']),
        #     Permute(0, 3, 1, 2) if not kwargs['channel_first'] else nn.Identity(),
        #     )
        self.drop = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(inplane)
        

    # def align(self, x, cond):
    #     b, c, h, w = x.size()
    #     x, cond = x.reshape(b, -1, c), cond.reshape(b, -1, c)
    #     x = x + self.drop(self.channel_align(self.norm(x), self.norm(cond), self.norm(cond)))
    #     x = x + self.drop(self.mlp(self.norm(x), h, w))
    #     x, cond = x.reshape(b, c, h, w), cond.reshape(b, c, h, w)
    #     return x
    def align(self, x, cond):
        b, c, h, w = x.size()
        x, cond = x.reshape(b, -1, c), cond.reshape(b, -1, c)
        # x = x + self.drop(self.channel_align(self.norm(x), h, w, self.norm(cond)))
        x = x + self.drop(self.channel_align(self.norm(x), self.norm(cond), self.norm(cond)))
        x, cond = x.reshape(b, c, h, w), cond.reshape(b, c, h, w)
        x = x + self.drop(self.mlp(self.norm(x.permute(0,2,3,1)), h, w)).permute(0,3,1,2)
        # x, cond = x.reshape(b, c, h, w), cond.reshape(b, c, h, w)
        # x, cond = x.reshape(b, h, w, c).permute(0,3,1,2), cond.reshape(b, h, w, c).permute(0, 3, 1, 2)
        # x = x + self.drop(self.vss_block(x))
        
        return x

    def denormalize_img(self, normalized_img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
        """Denormalize image by multiplying by std and adding mean."""
        denormalized_img = np.empty_like(normalized_img, np.float32)
        for i in range(3):  # Loop over color channels
            denormalized_img[i, ...] = (normalized_img[i, ...] * std[i]) + mean[i]
        return denormalized_img

    def visualize_flow(self, flow, meta):
        """
        可视化光流矢量场
        :param flow: 光流场，形状为 (n, 2, h, w)
        :param img: 原始图像，形状为 (h, w, 3)
        """
        import cv2
        from matplotlib import pyplot as plt
        import numpy as np
        assert all(key in meta for key in ['name', 'index', 'img'])
        flow_0 = F.interpolate(flow.clone(), size=(1024,1024), mode='bilinear', align_corners=False)
        flow_0 = flow_0.cpu().detach().numpy()
        h, w = flow_0.shape[2:]
        out_path = f"bda_results/exp_flowvis/feat_map/{meta['name'][0]}_{meta['index']}.jpg"
        # 生成用于显示的图像副本
        # vis = np.zeros((h, w, 3), dtype=np.uint8)
        # if meta['img'] is None:
        #     return
        # vis = meta['img'].squeeze().cpu().detach().numpy()
        # vis = self.denormalize_img(vis).astype(np.uint8)
        # if vis.dtype != np.uint8:
        #     vis = vis.astype(np.uint8)
        # 确保图像数组的形状是 (height, width, channels)
        # if vis.shape[0] == 3:
        #     vis = vis.transpose(1, 2, 0)
        path=f"dataset/test/images/{meta['name'][0]}_post_disaster.png"
        vis = cv2.imread(path)
        # 绘制光流矢量
        step = 100  # 设置箭头间隔
        length = 10
        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = flow_0[0, :, y, x]  # 取出光流的 x 和 y 方向分量
                start_point = (x, y)
                # end_point = (int(x + fx*10), int(y + fy*10))
                flow_length = np.sqrt(fx**2 + fy**2)
        
                # 归一化光流矢量并固定长度
                if flow_length > 0:
                    fx = (fx / flow_length) * length
                    fy = (fy / flow_length) * length
                
                end_point = (int(x + fx), int(y + fy))
                cv2.arrowedLine(vis, start_point, end_point, (0, 255, 0), 1, tipLength=0.3)
        if vis.shape[0] == 3:
            vis = vis.transpose(1, 2, 0)
        cv2.imwrite(out_path, vis)
        return

    # def forward(self, x1, x2):
    #     # channel alignment
    #     x2_align = self.align(x2, x1)
    #     x1 = self.flowmlp1(x1)
    #     x2 = self.flowmlp2(x2_align)

    #     size = x1.size()[2:]
    #     # flow1 = self.flow_make1(torch.cat([x1, x2], dim=1))
    #     flow = self.flow_make2(torch.cat([x2, x1], dim=1))
    #     # warping
    #     seg_flow_warp2 = self.flow_warp(x2, flow, size)
    #     # seg_flow_warp1 = self.flow_warp(x1, flow1, size) #A
    #     # diff1 = torch.abs(seg_flow_warp1 - x2)
    #     # self.visualize_flow(flow2, x2)
    #     return seg_flow_warp2
    
    # def forward(self, x1, x2, meta=None):
    def forward(self, x1, x2):
        # channel alignment
        x1 = self.flowmlp1(x1)
        x2 = self.flowmlp2(x2)
        size = x1.size()[2:]
        x2 = self.align(x2, x1)
        flow = self.flow_make2(torch.cat([x2, x1], dim=1))
        # warping
        seg_flow_warp2 = self.flow_warp(x2, flow, size)
        # seg_flow_warp1 = self.flow_warp(x1, flow1, size) #A
        # diff1 = torch.abs(seg_flow_warp1 - x2)
        return seg_flow_warp2, flow

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)

        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=False)
        return output

def find_largest_blob_center(loc):
    """
    找到 loc 中值为1的最大斑块的中心像素位置。

    参数:
    loc (torch.Tensor): 形状为 (1, 1, H, W) 的张量，包含值为0和1的像素。

    返回:
    (int, int): 最大斑块的中心像素位置 (px, py)。
    """
    # 将 loc 转换为 NumPy 数组
    loc_np = loc[0, 0].cpu().numpy()

    # 标记所有连通组件
    labeled_array, num_features = label(loc_np)

    # 找到每个连通组件的切片
    slices = find_objects(labeled_array)

    # 找到最大的连通组件
    max_size = 0
    max_slice = None
    for sl in slices:
        size = (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
        if size > max_size:
            max_size = size
            max_slice = sl

    # 计算最大连通组件的中心位置
    if max_slice is not None:
        center_y = (max_slice[0].start + max_slice[0].stop) // 2
        center_x = (max_slice[1].start + max_slice[1].stop) // 2
        return center_x, center_y
    else:
        return None, None

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    x1 = torch.randn(8, 128, 64, 64).cuda()
    x2 = torch.randn(8, 128, 64, 64).cuda()
    model = FlowN(128, 64, 64, channel_first=False).cuda()
    x2 = model(x1, x2)
    print(x1.shape, x2.shape)