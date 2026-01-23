from changedetection.configs.config import get_config
from changedetection.models.STMambaBDA import STMambaBDA, FlowMamba
from changedetection.models.comparative import BIT_BDA, ChangeOS, HCGMNet_BDA, CGNet_BDA, \
    RFANet_BDA, SAM_BDA, ChangeFormer_BDA, USSFCNet_BDA, DMINet_BDA, SNUNet_BDA, RSM_BDA
from mmengine.config import Config
import timm

def get_kwargs(args):
    config = get_config(args)
    kawrgs = dict(
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
        use_ossm=False,
    )
    return kawrgs

def get_models(args):
    if args.model_type.startswith('MambaBDA'):
        kwargs = get_kwargs(args)
        model = STMambaBDA(
            output_building=2, output_damage=5, 
            pretrained=args.pretrained_weight_path,
            **kwargs)
    elif args.model_type.startswith('FlowMamba'):
        kwargs = get_kwargs(args)
        model = FlowMamba(
            output_building=2, output_damage=5, 
            pretrained=args.pretrained_weight_path,
            **kwargs)
    elif args.model_type == 'bit':
        model = BIT_BDA(out_loc=2, out_dmg=5, backbone='resnet50')
    elif args.model_type == 'changeos':
        model = ChangeOS(out_loc=2, out_dmg=5, backbone='resnet50')
    elif args.model_type == 'HCGMNet':
        model = HCGMNet_BDA(out_loc=2, out_dmg=5, backbone='resnet50')
    elif args.model_type == 'CGNet':
        model = CGNet_BDA(out_loc=2, out_dmg=5, backbone='resnet50')
    elif args.model_type == 'RFANet':
        model = RFANet_BDA(out_loc=2, out_dmg=5, backbone='resnet50')
    elif args.model_type == 'SAM_CD':
        model = SAM_BDA(out_loc=2, out_dmg=5, backbone='pretrained/FastSAM-x.pt')
    elif args.model_type == 'ChangeFormer':
        model = ChangeFormer_BDA(out_loc=2, out_dmg=5, backbone='pretrained/changeformer_Tenc_x2.pt')
    elif args.model_type == 'DMINet':
        model = DMINet_BDA(out_loc=2, out_dmg=5, backbone='resnet50', out_indices=(0, 1, 2, 3))
    elif args.model_type == 'USSFCNet':
        model = USSFCNet_BDA(out_loc=2, out_dmg=5)
    elif args.model_type == 'snunet':
        model = SNUNet_BDA(out_loc=2, out_dmg=5)
    elif args.model_type == 'rsmamba':
        model = RSM_BDA(out_loc=2, out_dmg=5)
    else:
        raise NotImplementedError(f"Model type {args.model_type} not implemented")
    return model
