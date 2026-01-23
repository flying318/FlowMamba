import torch
import torch.nn as nn
import torch.nn.functional as F
from . import resnet
from .util import get_norm_layer, Conv3x3, Identity, KaimingInitMixin
import os
import timm
from glob import glob

class Backbone(nn.Module, KaimingInitMixin):
    def __init__(
        self, 
        in_ch=3, out_ch=32,
        arch='resnet18',
        pretrained=True,
        out_indices=(1,2,3,4),
        n_stages=5
    ):
        super().__init__()

        expand = 1
        strides = (2,1,2,1,1)
        if arch == 'resnet18':
            self.resnet = resnet.resnet18(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet34':
            self.resnet = resnet.resnet34(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet50':
            self.resnet = resnet.resnet50(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
            expand = 1
        elif arch == 'resnet101':
            self.resnet = resnet.resnet101(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        else:
            raise ValueError

        self.n_stages = n_stages
        self.out_indices = out_indices
        if self.n_stages == 5:
            itm_ch = 512 * expand
        elif self.n_stages == 4:
            itm_ch = 256 * expand
        elif self.n_stages == 3:
            itm_ch = 128 * expand
        else:
            raise ValueError

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_out = Conv3x3(itm_ch, out_ch)

        self._trim_resnet()

        if in_ch != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_ch, 
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

        if not pretrained:
            self._init_weight()
    
    def multi_stage_forward(self, x):
        x0 = self.resnet.conv1(x)
        x1 = self.resnet.layer1(x0)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        return [x0, x1, x2, x3, x4]
    
    def one_stage(self, x):
        y = self.resnet.conv1(x)
        y = self.resnet.bn1(y)
        y = self.resnet.relu(y)
        y = self.resnet.maxpool(y)

        y = self.resnet.layer1(y)
        y = self.resnet.layer2(y)
        y = self.resnet.layer3(y)
        y = self.resnet.layer4(y)

        y = self.upsample(y)

        return self.conv_out(y)
    
    def forward(self, x):
        if self.out_indices:
            return self.multi_stage_forward(x)
        return self.one_stage(x)

    def _trim_resnet(self):
        if self.n_stages > 5:
            raise ValueError

        if self.n_stages < 5:
            self.resnet.layer4 = Identity()

        if self.n_stages <= 3:
            self.resnet.layer3 = Identity()

        self.resnet.avgpool = Identity()
        self.resnet.fc = Identity()

def load_timm_backbone(backbone, out_indices=(1,2,3,4), pretrain: bool = True, pretrain_dir=None):
    model = timm.create_model(model_name=backbone, pretrained=False, features_only=True, out_indices=out_indices)
    if not pretrain:
        return model
    pretrain_dir = '/home/guoxin/.cache/torch/hub/checkpoints' if pretrain_dir is None else pretrain_dir
    ckpt_path = glob(os.path.join(pretrain_dir, f'{backbone}-*.pth'))
    # download_weights(model, backbone)
    if len(ckpt_path) == 0:
        print(f'{backbone} pretrained model not found, downloading...')
        weights = timm.create_model(model_name=backbone, pretrained=True)
        print(f'Saving {backbone} pretrained model to {ckpt_path}')
        del weights
    ckpt_path = ckpt_path[0]
    assert os.path.exists(ckpt_path), f'{backbone} pretrained model not found'
    ckpt = torch.load(ckpt_path)
    filter_ckpt = {k: v for k, v in ckpt.items() if 'fc' not in k}
    model.load_state_dict(filter_ckpt, strict=False)
    return model

if __name__ == "__main__":
    model = Backbone(3, 5, arch="resnet18", pretrained=False, out_indices=(1,2,3,4)).cuda()
    # model = Backbone(3, 32, arch="resnet50", pretrained=False, out_indices=None).cuda()
    # path = '/home/guoxin/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth'
    model = timm.create_model('resnet50', num_classes=5, pretrained=True, features_only=True).cuda()
    x = torch.randn(8, 3, 224, 224).cuda()
    y = model(x)
    shape = [i.shape for i in y]
    print(shape)
    # print(y.shape)