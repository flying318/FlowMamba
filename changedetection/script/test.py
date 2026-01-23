import sys
sys.path.append('xxx') # specify your project root path

import argparse
import os
import time
from datetime import timedelta
import numpy as np
import logging
from changedetection.configs.config import get_config

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from changedetection.datasets.make_data_loader import DamageAssessmentDatset
from changedetection.utils_func.metrics import Evaluator
import imageio
import changedetection.utils_func.lovasz_loss as L
from changedetection.utils_func.seed import set_seed
from changedetection.utils_func.logger import get_logger, log_test_metrics
from changedetection.utils_func.build_model import get_models

model_type = ['changeos', 'HCGMNet', 'bit', 'rsmamba','RFANet', 'SAM_CD', 'snunet', 
             'CGNet', 'USSFCNet', 'DMINet', 'ChangeFormer', 'FlowMamba', 'MambaBDA']
class Inference(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)
        self.color_map = dict(background=[0, 0, 0], 
                              no_damage=[0, 255, 0], 
                              minor_damage=[255, 128, 0], 
                              major_damage=[153, 51, 255], 
                              destroyed=[255, 0, 0])
        self.evaluator_loc = Evaluator(num_class=2)
        self.evaluator_clf = Evaluator(num_class=5)
        self.deep_model = get_models(args)
        self.deep_model = self.deep_model.cuda()

        self.change_map_saved_path = os.path.join(args.result_saved_path, 'result')
        self.feature_map_saved_path = args.feature_map_saved_path
        if not os.path.exists(self.change_map_saved_path):
            os.makedirs(self.change_map_saved_path)

        if not os.path.exists(self.feature_map_saved_path):
            os.makedirs(self.feature_map_saved_path)

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
            self.deep_model.load_state_dict(state_dict, strict=True)

        self.deep_model.eval()
        if args.export_onnx:
            print('Exporting ONNX file...')
            input = (torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 3, 256, 256).cuda())
            input_names = ['input1', 'input2']
            output_names = ['output1', 'output2']
            torch.onnx.export(self.deep_model, input, '/home/guoxin/MambaCD/bda.onnx', verbose=False,
                              input_names=input_names, output_names=output_names)
        self.logger = get_logger(args.log_file, logging.INFO, test=True)


    def infer(self):
        set_seed()
        test_start_time = time.time()
        self.logger.info('-'*20+'Inference'+'-'*20+'\n')
        torch.cuda.empty_cache()
        dataset = DamageAssessmentDatset(self.args.test_dataset_path, self.args.test_data_name_list, 1024, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=8, drop_last=False)
        torch.cuda.empty_cache()
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        infer_time = 0
        vbar = tqdm(val_data_loader, ncols=100)
        val_enumerator = enumerate(val_data_loader)
        for _ in vbar:
            itera, data = val_enumerator.__next__()
            pre_change_imgs, post_change_imgs, labels_loc, labels_clf, names = data
            if labels_loc.eq(0).all():
                continue
            pre_change_imgs = pre_change_imgs.cuda().float()
            post_change_imgs = post_change_imgs.cuda()
            labels_loc = labels_loc.cuda().long()
            labels_clf = labels_clf.cuda().long()
            start_time = time.time()
            with torch.no_grad():
                output_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs)
            infer_time += (time.time() - start_time)/self.args.batch_size

            output_loc = output_loc.data.cpu().numpy().transpose(0,2,3,1)
            output_loc = np.argmax(output_loc, axis=-1)
            labels_loc = labels_loc.cpu().numpy()
            output_clf = output_clf.data.cpu().numpy().transpose(0,2,3,1)
            output_clf = np.argmax(output_clf, axis=-1)
            labels_clf = labels_clf.cpu().numpy()

            self.evaluator_loc.add_batch(labels_loc, output_loc)
            
            output_clf_val = output_clf[labels_loc > 0]
            labels_clf_val = labels_clf[labels_loc > 0]
            self.evaluator_clf.add_batch(labels_clf_val, output_clf_val)

            loc_map = np.squeeze(output_loc)
            loc_map[loc_map==1] = 255
            output_clf = np.squeeze(output_clf)
            output_clf[loc_map==0] = 0
            clf_map = np.zeros(np.squeeze(output_clf).shape+(3,))
            assert len(clf_map.shape) >= 3, f'clf_map is expected to be 3 dim, but got {clf_map.shape}'
            for i, color in enumerate(self.color_map.values()):
                clf_map[output_clf==i] = color
            clf_map[(loc_map==1) & (output_clf==0)] = [255, 255, 255]
            if len(names) > 1:
                for i in range(clf_map.shape[0]):
                    pre_name = names[i] + f'_pre_disaster.png'
                    post_name = names[i] + f'_post_disaster.png'
                    imageio.imwrite(os.path.join(self.change_map_saved_path, pre_name), loc_map[i].astype(np.uint8))
                    imageio.imwrite(os.path.join(self.change_map_saved_path, post_name), clf_map[i].astype(np.uint8))
            else:
                pre_name = names[0] + f'_pre_disaster.png'
                post_name = names[0] + f'_post_disaster.png'
                imageio.imwrite(os.path.join(self.change_map_saved_path, pre_name), loc_map.astype(np.uint8))
                imageio.imwrite(os.path.join(self.change_map_saved_path, post_name), clf_map.astype(np.uint8))
        infer_time = timedelta(seconds=infer_time/len(val_data_loader))
        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        damage_f1_score = self.evaluator_clf.Damage_F1_score()
        uoc_index = self.evaluator_clf.uoc_index()
        acc = self.evaluator_clf.Pixel_Accuracy()
        harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
        oaf1 = 0.3 * loc_f1_score + 0.7 * harmonic_mean_f1
        
        metrics = dict(acc=acc, uoc=uoc_index, F1_oa=oaf1, F1_loc=loc_f1_score, F1_bda=harmonic_mean_f1, F1_subcls=damage_f1_score)
        test_end_time = time.time()
        test_time = time.strftime('%H:%M:%S', time.gmtime(test_end_time - test_start_time))
        log_test_metrics(self.logger, metrics)
        self.logger.info(f'Test time: {str(test_time)}')
            


def main():
    parser = argparse.ArgumentParser(description="Training on xBD dataset")
    parser.add_argument('--cfg', type=str, default='/home/guoxin/mambacd/classification/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--dataset', type=str, default='xBD')
    parser.add_argument('--test_dataset_path', type=str, default='/home/songjian/project/datasets/SYSU/test')
    parser.add_argument('--test_data_list_path', type=str, default='/home/songjian/project/datasets/SYSU/test_list.txt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--model_type', type=str, default='STMamBDA_base')
    parser.add_argument('--result_saved_path', type=str, default='../results')
    parser.add_argument('--feature_map_saved_path', type=str, default='../feature_maps')
    parser.add_argument('--use_ossm', action='store_true', default=False)
    parser.add_argument('--log_file', type=str, default='../train_val.log')
    parser.add_argument('--hazard_seperate', type=bool, default=False)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--export_onnx', action='store_true', default=False)
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    args = parser.parse_args()

    with open(args.test_data_list_path, "r") as f:
        test_data_name_list = [data_name.strip() for data_name in f.readlines()]
    args.test_data_name_list = test_data_name_list

    infer = Inference(args)
    infer.infer()


if __name__ == "__main__":
    main()
