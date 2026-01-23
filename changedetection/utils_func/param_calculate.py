import sys
sys.path.append('/home/guoxin/MambaCD')
import torch
from thop import profile, clever_format ## 导入thop模块
from thop.vision.basic_hooks import count_linear, count_convNd
from changedetection.configs.config import get_config
from changedetection.utils_func.build_model import get_models
import copy
import torch.nn as nn
import argparse
custom_ops = {'Conv2d': count_convNd}
def _purge_thop_attrs(model: nn.Module):
    for m in model.modules():
        if hasattr(m, 'total_ops'):
            try:
                delattr(m, 'total_ops')
            except Exception:
                pass
        if hasattr(m, 'total_params'):
            try:
                delattr(m, 'total_params')
            except Exception:
                pass

def param_calculate(model, batch_size, size, logger=None, channels=[3,3], bool_label=False):
    device = next(model.parameters()).device
    input1 = torch.randn(batch_size, channels[0], size, size).to(device)
    input2 = torch.randn(batch_size, channels[1], size, size).to(device)
    m = copy.deepcopy(model).eval()
    _purge_thop_attrs(m)
    if bool_label:
        label = torch.randint(0, 4, (batch_size, size, size)).to(device)
        flops, params = profile(m, inputs=(input1, input2, label), custom_ops=custom_ops)
    else:
        flops, params = profile(m, inputs=(input1, input2))
    flops, params = clever_format([flops, params], "%.3f")
    total = sum(p.numel() for p in model.parameters())
    print(f'flops: {flops}')			## 打印计算量
    print(f'params: {params}')			## 打印参数量
    print("Total params: %.3fM" % (total/1e6))
    logger.info(f'flops: {flops}, params: {params}, Total params: {total/1e6:.3f} M') if logger else None

def cal_params_flops(model, size):
    input1 = torch.randn(4, 3, size, size).to('cuda:0')
    input2 = torch.randn(4, 3, size, size).to('cuda:0')
    flops, params = profile(model, inputs=(input1, input2))
    print('flops',flops/1e9)			## 打印计算量
    print('params',params/1e6)			## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.3fM" % (total/1e6))
    # logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')

if __name__ == '__main__':
    model_type = ['changeos', 'HCGMNet', 'bit', 'rsmamba',
        'RFANet', 'SAM_CD', 'snunet', 'CGNet', 
        'USSFCNet', 'DMINet', 'ChangeFormer', 'MambaBDA', 'FlowMamba']
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='UltraLight_VM_UNet', choices=model_type)
    parser.add_argument('--cfg', type=str, default='changedetection/configs/vssm1/vssm_small_224.yaml', help='Path to model config file')
    parser.add_argument('--pretrained_weight_path', type=str, default=None, help='Path to pretrained weights')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    args = parser.parse_args()
    model = get_models(args)
    model = model.to('cuda:0')
    param_calculate(model, batch_size=8, size=256) 
    # model = UltraLight_VM_UNet().to('cuda:0')
    # model = model.to('cuda:0')
    # model.eval()
    # cal_params_flops(model, 1024)