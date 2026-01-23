import sys
sys.path.append('/home/guoxin/MambaCD')

import argparse
import os
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import trange
from changedetection.configs.config import get_config
from changedetection.datasets.make_data_loader import make_data_loader, DamageAssessmentDatset, MultimodalDamageAssessmentDatset
from changedetection.utils_func.metrics import Evaluator
from glob import glob
from changedetection.utils_func import lovasz_loss as L
from changedetection.utils_func.ordinal_loss import *
from changedetection.utils_func.seed import set_seed
from changedetection.utils_func.param_calculate import param_calculate
from changedetection.utils_func.logger import get_logger, log_validation_metrics, log_dfc_metrics
from changedetection.models.STMambaBDA import FlowMamba
from changedetection.models.comparative import BIT_BDA, ChangeOS, HCGMNet_BDA, CGNet_BDA, \
    RFANet_BDA, SAM_BDA, ChangeFormer_BDA, USSFCNet_BDA, DMINet_BDA, SNUNet_BDA, RSM_BDA
# torch.autograd.set_detect_anomaly(True)
model_types = ['changeos', 'HCGMNet', 'bit', 'rsmamba',
              'RFANet', 'SAM_CD', 'snunet', 'CGNet', 
              'USSFCNet', 'DMINet', 'ChangeFormer', 'MambaBDA','FlowMamba']
class Trainer(object):
    def __init__(self, args):
        # set_seed()
        config = get_config(args)
        assert args.model_type in model_types, f"model_type should be one of {model_types}"
        self.args = args
        if args.model_type == 'bit':
            self.deep_model = BIT_BDA(out_loc=2, out_dmg=5, backbone='resnet50')
        elif args.model_type == 'changeos':
            self.deep_model = ChangeOS(out_loc=2, out_dmg=5, backbone='resnet50')
        elif args.model_type == 'HCGMNet':
            self.deep_model = HCGMNet_BDA(out_loc=2, out_dmg=5, backbone='resnet50')
        elif args.model_type == 'CGNet':
            self.deep_model = CGNet_BDA(out_loc=2, out_dmg=5, backbone='resnet50')
        elif args.model_type == 'RFANet':
            self.deep_model = RFANet_BDA(out_loc=2, out_dmg=5, backbone='resnet50')
        elif args.model_type == 'SAM_CD':
            self.deep_model = SAM_BDA(out_loc=2, out_dmg=5, backbone='pretrained/FastSAM-x.pt')
        elif args.model_type == 'ChangeFormer':
            self.deep_model = ChangeFormer_BDA(out_loc=2, out_dmg=5, backbone='pretrained/changeformer_Tenc_x2.pt')
        elif args.model_type == 'DMINet':
            self.deep_model = DMINet_BDA(out_loc=2, out_dmg=5, backbone='resnet50', out_indices=(0, 1, 2, 3))
        elif args.model_type == 'USSFCNet':
            self.deep_model = USSFCNet_BDA(out_loc=2, out_dmg=5)
        elif args.model_type == 'snunet':
            self.deep_model = SNUNet_BDA(out_loc=2, out_dmg=5)
        elif args.model_type == 'rsmamba':
            self.deep_model = RSM_BDA(out_loc=2, out_dmg=5)
        elif args.model_type == 'FlowMamba':
            self.deep_model = FlowMamba(
                output_building=2, output_damage=5, 
                pretrained=args.pretrained_weight_path,
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
                use_ossm=args.use_ossm,
                ) 
        else:
            raise NotImplementedError(f"model_type: {args.model_type} is not implemented")
        
        local_rank = args.local_rank
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.deep_model.to(device)
        self.model_save_path = args.model_param_path
        os.makedirs(self.model_save_path, exist_ok=True)
        self.train_data_loader = make_data_loader(args)
        self.evaluator_loc = Evaluator(num_class=2)
        self.evaluator_clf = Evaluator(num_class=4)
        self.logger = get_logger(args.log_file, logging.INFO)
        self.logger.info('-' * 10 + 'Training configs' + '-' * 10)
        channels = [3, 1] if args.dataset == 'bright' else [3, 3]
        param_calculate(self.deep_model, args.batch_size, args.crop_size, self.logger, bool_label=False, channels=channels)
        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size
        self.local_rank = local_rank
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
        self.scheduler = StepLR(self.optim, step_size=20000, gamma=0.5)
        self.eps = 1e-7

    def training(self):
        best_kc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)
        self.logger.info(f'Training option: {self.args.model_type} + lovasz + ce')
        self.logger.info('-'*10+'Start training'+'-'*10)
        start = time.time()
        custom_loss = torch.tensor(0.0).cuda()
        for _ in range(elem_num):
            
            itera, data = train_enumerator.__next__()
            pre_change_imgs, post_change_imgs, labels_loc, labels_clf, name = data

            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            labels_loc = labels_loc.cuda().long()
            labels_clf = labels_clf.cuda().long()
            print('labels_clf: False') if labels_clf is None else None
            valid_labels_clf = (labels_clf != 255).any()
            if not valid_labels_clf:
               continue
            if self.args.model_type == 'SAM_CD':
                output_loc, output_clf, custom_loss = self.deep_model(pre_change_imgs, post_change_imgs, labels_clf)
            else:
                output_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs)

            self.optim.zero_grad()
            ce_loss_loc = F.cross_entropy(output_loc, labels_loc, ignore_index=255)
            lovasz_loc = L.lovasz_softmax(F.softmax(output_loc, dim=1), labels_loc, ignore=255)
            loc_loss = ce_loss_loc + lovasz_loc * 0.5
            lovasz_loss_clf = L.lovasz_softmax(F.softmax(output_clf, dim=1), labels_clf, ignore=255)
            ce_loss_clf = F.cross_entropy(output_clf, labels_clf, ignore_index=255)
            clf_loss = ce_loss_clf + lovasz_loss_clf*0.75 + custom_loss

            final_loss = loc_loss + clf_loss

            final_loss.backward()

            self.optim.step()
            self.scheduler.step()
            if (itera + 1) % 50 == 0:
                end = time.time()
                itera_time = end - start
                eta = (elem_num - itera) * itera_time / 50
                eta = time.strftime('%H:%M:%S', time.gmtime(eta))
                log_info = f'Training | iter: {str(itera + 1)}/{str(elem_num)} | loss: {final_loss.item():.4f} | loc_loss: {loc_loss.item():.4f} | bda_loss: {clf_loss.item():.4f} | eta: {eta}'
                self.logger.info(log_info) if self.local_rank == 0 else None
                if (itera + 1) % 1000 == 0:
                    self.deep_model.eval()
                    loc_f1_score, harmonic_mean_f1, oaf1, damage_f1_score = self.validation()
                    if oaf1 > best_kc:
                        best_model_path = glob(os.path.join(self.model_save_path, 'best_model_*.pth'))
                        os.remove(best_model_path[0]) if len(best_model_path) > 0 else None
                        torch.save(self.deep_model.state_dict(),
                                   os.path.join(self.model_save_path, f'best_model_{itera + 1}.pth'))
                        best_kc = oaf1
                        best_round = dict(F1_oa=oaf1, F1_loc=loc_f1_score, F1_bda=harmonic_mean_f1, F1_subcls=damage_f1_score)
                    torch.cuda.empty_cache()
                    self.deep_model.train()
                start = time.time()

        log_validation_metrics(self.logger, best_round, final=True)

    @torch.no_grad()
    def validation(self):
        set_seed()
        print('-'*15+'starting evaluation'+'-'*15) if self.local_rank == 0 else None
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        # dataset = DamageAssessmentDatset(self.args.test_dataset_path, self.args.test_data_name_list, 512, None, 'test')
        dataset = MultimodalDamageAssessmentDatset(self.args.test_dataset_path, self.args.test_data_name_list, 1024, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=6, drop_last=False)
        vbar = trange(len(val_data_loader), ncols=120)
        val_enumerator = enumerate(val_data_loader)
        with torch.no_grad():
            for _ in vbar:
                _, data = val_enumerator.__next__()
                pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_loc = labels_loc.cuda().long()
                labels_clf = labels_clf.cuda().long()
                output_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs)
                output_loc = output_loc.data.cpu().numpy()
                output_loc = np.argmax(output_loc, axis=1)
                labels_loc = labels_loc.cpu().numpy()

                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                labels_clf = labels_clf.cpu().numpy()

                self.evaluator_loc.add_batch(labels_loc, output_loc)
                
                output_clf = output_clf[labels_loc > 0]
                labels_clf = labels_clf[labels_loc > 0]
                self.evaluator_clf.add_batch(labels_clf, output_clf)

        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        damage_f1_score = self.evaluator_clf.Damage_F1_socore()
        harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
        oaf1 = 0.3 * loc_f1_score + 0.7 * harmonic_mean_f1
        metrics = dict(F1_oa=oaf1, F1_loc=loc_f1_score, F1_bda=harmonic_mean_f1, F1_subcls=damage_f1_score)
        log_dfc_metrics(self.logger, metrics)
        return loc_f1_score, harmonic_mean_f1, oaf1, damage_f1_score


def main():
    parser = argparse.ArgumentParser(description="Training on xBD dataset")
    parser.add_argument('--cfg', type=str, default=None)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str, default=None)

    parser.add_argument('--dataset', type=str, default='xBD')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str, default='/data/ggeoinfo/datasets/xBD/train')
    parser.add_argument('--train_data_list_path', type=str, default='/data/ggeoinfo/datasets/xBD/xBD_list/train_all.txt')
    parser.add_argument('--test_dataset_path', type=str, default='/data/ggeoinfo/datasets/xBD/test')
    parser.add_argument('--test_data_list_path', type=str, default='/data/ggeoinfo/datasets/xBD/xBD_list/val_all.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='FlowMamba')
    parser.add_argument('--model_param_path', type=str, default='../saved_models')
    parser.add_argument('--log_file', type=str, default='../train_val.log')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    # parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--use_ossm', action='store_true', default=False)

    args = parser.parse_args()
    with open(args.train_data_list_path, "r") as f:
        data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = data_name_list

    with open(args.test_data_list_path, "r") as f:
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
