import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import normal
import segmentation_models_pytorch as smp
import numpy as np
import cv2

class OhemCrossEntropy(nn.Module):
    """
        Ohem Cross Entropy Tensor Version
    """
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=256,
                 use_weight=True):
        super(OhemCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [0.5,0.5,2,2,1]).cuda()
            self.criterion = nn.CrossEntropyLoss(reduction="mean",
                                                weight=weight,
                                                ignore_index=ignore_index)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction="mean",
                                                ignore_index=ignore_index)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        # target = target.reshape(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
            #print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)

class SimLoss(nn.Module):
    def __init__(self,ignore_index=0, label_smooth=0.1, base: float = 0.3, class_weight=None):
        super(SimLoss, self).__init__()
        self.base = torch.tensor(base)
        self.ignore_index = ignore_index
        self.label_smooth = label_smooth
        class_weight = [1, 1, 1, 1, 1] if class_weight is None else class_weight
        self.class_weight = torch.tensor(class_weight).cuda()
        self.ohem_ce = OhemCrossEntropy(ignore_index=self.ignore_index)
        self.dice = smp.losses.DiceLoss(mode='multiclass', ignore_index=self.ignore_index)


    def mask(self, input:Tensor, num_cls:int):
        '''input: Tensor, shape (B, 1, H, W),
        num_cls: int, the number of classes'''
        label_onehot = F.one_hot(input.squeeze().long(), num_classes=num_cls).permute(0, 3, 1, 2).float()
        mask = label_onehot.sum((0, 2, 3)) > 0
        return mask.float()
    
    def distance_smooth_label(self, label:Tensor):
        label = label.squeeze().long()
        cls = torch.abs(torch.arange(self.num_cls).unsqueeze(0) - torch.arange(self.num_cls).unsqueeze(1)).to(label)
        out = cls.index_select(0, label.reshape(-1)).reshape(label.size() + (-1,))
        return out.permute(0, 3, 1, 2).float()

    def balanced_softmax_loss(self, labels, logits, sample_per_class):
        """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
        Args:
        labels: A int tensor of size [batch].
        logits: A float tensor of size [batch, no_of_classes].
        sample_per_class: A int tensor of size [no of classes].
        reduction: string. One of "none", "mean", "sum"
        Returns:
        loss: A float tensor. Balanced Softmax Loss.
        """
        assert logits.ndim==4 and labels.ndim==3, \
        f'The logits and labels should be in the shape of [N, C, H, W] and [N, H, W] respectively. Got {logits.shape} and {labels.shape} instead.'
        b, c, h, w = logits.shape
        logits = logits.permute(0,2,3,1).contiguous().view(-1, logits.shape[1]) #NCHW -> NHWC -> (NHW)C
        labels = labels.view(-1) # (NHW)
        spc = sample_per_class.type_as(logits)
        spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
        logits = logits + spc.log()
        logits = logits.view(b,h,w,c).permute(0,3,1,2).contiguous() # (NHW)C -> NHWC -> NCHW
        return logits

    def forward(self, pred_logits, label):
        self.num_cls = pred_logits.size(1)
        label_ = label.clone().detach()
        label_[label_==255] = 0
        dist_matrix = self.distance_smooth_label(label_)
        sim_matrix = self.base.pow(dist_matrix)
        sim_matrix = sim_matrix * (label_.unsqueeze(1) != 0).float()
        ce = F.cross_entropy(pred_logits, sim_matrix, weight=self.class_weight)
        dice = self.dice(pred_logits, label_)
        return ce+dice