import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import normal
import segmentation_models_pytorch as smp
import numpy as np
import cv2

class BackCELoss(nn.Module):
    def __init__(self, args, ignore_lb=255):
        super(BackCELoss, self).__init__()
        self.ignore_lb = ignore_lb
        self.class_num = args.num_classes
        self.criteria = nn.NLLLoss(ignore_index=ignore_lb, reduction='mean')
    def forward(self, logits, labels):
        total_labels = torch.unique(labels)
        new_labels = labels.clone()
        probs = torch.softmax(logits,1)
        fore_ = []
        back_ = []
        
        for l in range(self.class_num):
            if l in total_labels:
                fore_.append(probs[:,l,:,:].unsqueeze(1))
            else: 
                back_.append(probs[:,l,:,:].unsqueeze(1))
        Flag=False
        if not  len(fore_)==self.class_num:
            fore_.append(sum(back_))
            Flag=True
        
        for i,l in enumerate(total_labels):
            if Flag :
                new_labels[labels==l]=i
            else: 
                if l!=255:
                    new_labels[labels==l]=i
            
        probs  =torch.cat(fore_,1)
        logprobs = torch.log(probs+1e-7)
        return self.criteria(logprobs,new_labels.long())

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

class Ohem(nn.Module):
    # Based on https://arxiv.org/pdf/1812.05802.pdf
    def __init__(self, fraction=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.fraction = fraction

    def forward(self, y_pred, y_true):
        batch_size = y_true.size(0)
        losses = self.loss(y_pred, y_true).view(batch_size, -1)

        positive_mask = (y_true > 0).view(batch_size, -1)
        Cp = torch.sum(positive_mask, dim=1)
        Cn = torch.sum(~positive_mask, dim=1)
        Chn = torch.max((Cn / 4).clamp_min(5), 2 * Cp)

        loss, num_samples = 0, 0
        for i in range(batch_size):
            positive_losses = losses[i, positive_mask[i]]
            negative_losses = losses[i, ~positive_mask[i]]
            num_negatives = int(Chn[i])
            hard_negative_losses, _ = negative_losses.sort(descending=True)[:num_negatives]
            loss = positive_losses.sum() + hard_negative_losses.sum() + loss
            num_samples += positive_losses.size(0)
            num_samples += hard_negative_losses.size(0)
        loss /= float(num_samples)

        return loss

class CORAL(nn.Module):
    # Adapted to image segmentation based on https://github.com/Raschka-research-group/coral-cnn
    def __init__(self):
        super(CORAL, self).__init__()
        self.levels = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.float32)

    def forward(self, y_pred, y_true):
        device = y_pred.device
        levels = self.levels[y_true].to(device)
        logpt = F.logsigmoid(y_pred)
        loss = torch.sum(logpt * levels + (logpt - y_pred) * (1 - levels), dim=1)
        return -torch.mean(loss)

class OrdinalEntropyLoss(nn.Module):
    def __init__(self, alpha, ignore_index=-100):
        super(OrdinalEntropyLoss, self).__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index

    def ordinalentropy(self, features, gt,  mask=None):
        """
        Features: a certain layer's features
        gt: pixel-wise ground truth values, in depth estimation, gt.size()= n, h, w
        mask: In case values of some pixels do not exist. For depth estimation, there are some pixels lack the ground truth values
        """
        f_n, f_c, f_h, f_w = features.size()

        features = F.interpolate(features, size=[f_h // 4, f_w // 4], mode='nearest')
        features = features.permute(0, 2, 3, 1)  # n, h, w, c
        features = torch.flatten(features, start_dim=1, end_dim=2)

        gt = F.interpolate(gt, size=[f_h // 4, f_w // 4], mode='nearest')

        loss = 0

        for i in range(f_n):
            """
            mask pixels that without valid values
            """
            _gt = gt[i,:].view(-1)
            _mask = _gt > 0.001
            _mask = _mask.to(torch.bool)
            _gt = _gt[_mask]
            _features = features[i,:]
            _features = _features[_mask,:]

            """
            diverse part
            """
            u_value, u_index, u_counts = torch.unique(_gt, return_inverse=True, return_counts =True)
            center_f = torch.zeros([len(u_value), f_c]).cuda()
            center_f.index_add_(0, u_index, _features)
            u_counts = u_counts.unsqueeze(1)
            center_f = center_f / u_counts

            p = F.normalize(center_f, dim=1)
            _distance = self.euclidean_dist(p, p)
            _distance = self.up_triu(_distance)

            u_value = u_value.unsqueeze(1)
            _weight = self.euclidean_dist(u_value, u_value)
            _weight = self.up_triu(_weight)
            _max = torch.max(_weight)
            _min = torch.min(_weight)
            _weight = ((_weight - _min) / _max)

            _distance = _distance * _weight

            _entropy = torch.mean(_distance)
            loss = loss - _entropy

            """
            tightness part
            """
            _features = F.normalize(_features, dim=1)
            _features_center = p[u_index, :]
            _features = _features - _features_center
            _features = _features.pow(2)
            _tightness = torch.sum(_features, dim=1)
            _mask = _tightness > 0
            _tightness = _tightness[_mask]

            _tightness = torch.sqrt(_tightness)
            _tightness = torch.mean(_tightness)

            loss = loss + _tightness

        return loss/ f_n

    def euclidean_dist(self, x, y):
        """
        Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
        Returns:
        dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def up_triu(self, x):
        # return a flattened view of up triangular elements of a square matrix
        n, m = x.shape
        assert n == m
        _tmp = torch.triu(torch.ones(n, n), diagonal=1).to(torch.bool)
        return x[_tmp]
    
    def forward(self, pred, label):
        oe = self.ordinalentropy(pred, label)
        ce = F.cross_entropy(pred, label, ignore_index=self.ignore_index)
        return oe + ce
    
class CornLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(CornLoss, self).__init__()
        self.ignore_index = ignore_index

    def loss_corn(self, logits, y_train, num_classes):
        sets = []
        for i in range(num_classes-1):
            label_mask = y_train > i-1
            label_tensor = (y_train[label_mask] > i).to(torch.int64)
            sets.append((label_mask, label_tensor))

        num_examples = 0
        losses = 0.
        for task_index, s in enumerate(sets):
            train_examples = s[0]
            train_labels = s[1]

            if len(train_labels) < 1:
                continue

            num_examples += len(train_labels)
            pred = logits[train_examples, task_index]

            loss = -torch.sum(F.logsigmoid(pred)*train_labels
                            + (F.logsigmoid(pred) - pred)*(1-train_labels)
                            )
            losses += loss
        return losses/num_examples
    
    def loss_corn_seg(self, logits, y_train, num_classes):
        sets = []
        logits = logits.permute(0,2,3,1).reshape(-1, num_classes)
        y_train = y_train.view(-1)
        for i in range(1, num_classes):
            label_mask = y_train > i-1
            label_tensor = (y_train[label_mask] > i).to(torch.int64)
            sets.append((label_mask, label_tensor))
        num_examples = 0
        losses = 0.
        for task_index, s in enumerate(sets):
            train_examples = s[0]
            train_labels = s[1]

            if len(train_labels) < 1:
                continue

            num_examples += len(train_labels)
            pred = logits[train_examples, task_index]

            loss = -torch.sum(F.logsigmoid(pred)*train_labels
                            + (F.logsigmoid(pred) - pred)*(1-train_labels)
                            )
            losses += loss
        return losses/num_examples

    def forward(self, pred_logits, label):
        self.num_classes = pred_logits.size(1)
        # mask = label != self.ignore_index
        # pred_logits = torch.where(mask.unsqueeze(1), pred_logits, torch.full_like(pred_logits, -1e9))
        # # pred_logits = pred_logits[mask]
        # label = label[mask]
        return self.loss_corn_seg(pred_logits, label, self.num_classes)
    
class CECORNLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(CECORNLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, pred_logits, target):
        label = target.clone().detach()
        label[label == 255] = 0
        dice_loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=self.ignore_index)
        ce = F.cross_entropy(pred_logits, label, ignore_index=self.ignore_index)
        dice = dice_loss(pred_logits, label.squeeze().long())
        # _, corn_logits = pred_logits.split([1,4], dim=1)
        cornloss = CornLoss(self.ignore_index)
        corn = cornloss(pred_logits, label)
        # return ce + dice + corn
        return ce, dice, corn
    
class OrdistanceCELoss(nn.Module):
    def __init__(self,ignore_index=-100, label_smooth=0.1):
        super(OrdistanceCELoss, self).__init__()
        self.pow = nn.Parameter(torch.tensor([2., 2.]))
        self.ignore_index = ignore_index
        self.label_smooth = label_smooth
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')

    def mask(self, input:Tensor, num_cls:int):
        '''input: Tensor, shape (B, 1, H, W),
        num_cls: int, the number of classes'''
        label_onehot = F.one_hot(input.squeeze().long(), num_classes=num_cls).permute(0, 3, 1, 2).float().to(input)
        mask = label_onehot.sum((0, 2, 3)) > 0
        return mask.float()
    
    # def distance_smooth_label(self, label:Tensor):
    #     label = label.squeeze().long()
    #     cls = 1/torch.abs(torch.arange(self.num_cls).unsqueeze(0) - torch.arange(self.num_cls).unsqueeze(1)).to(label)
    #     cls = cls*self.label_smooth
    #     cls = torch.where(torch.isnan(cls), 1.0-self.label_smooth, cls) if self.label_smooth == 0 \
    #         else torch.where(torch.isinf(cls), 1.0-self.label_smooth, cls)
    #     # label = label.squeeze()
    #     out = cls.index_select(0, label.reshape(-1)).reshape(label.size() + (-1,))
    #     return out.permute(0, 3, 1, 2).float()
    
    # def forward(self, pred_logits, label):
    #     self.num_cls = pred_logits.size(1)
    #     label[label==self.ignore_index] = 0
    #     weight = self.mask(label, self.num_cls) # mask
    #     weight_ = weight[None, :, None, None]
    #     logits_sm = torch.softmax(pred_logits, dim=1)
    #     log_logits = torch.log(logits_sm)
    #     # ce = F.cross_entropy(pred_logits, label.long().squeeze(), reduction='mean', weight=weight)
    #     # label_onehot = F.one_hot(label.long().squeeze(), num_classes=logits_sm.size(1)).permute(0, 3, 1, 2).float()
    #     label_smooth = self.distance_smooth_label(label)
    #     ce = -log_logits * label_smooth * weight_
    #     ce = (torch.sum(ce, dim=1) * (label != 0)).mean()
    #     # ce = torch.sum(ce,dim=1)
    #     # ce[label==0] = 0
    #     # ce = ce.mean()
    #     # ce = torch.sum(ce * weight_, dim=1).mean()
    #     # ce = F.nll_loss(log_logits, label.long().squeeze(), reduction='mean', weight=weight_, ignore_index=self.ignore_index)
    #     # ce = F.nll_loss(log_logits, label.long().squeeze(), reduction='none', weight=weight)
    #     pred = torch.argmax(logits_sm, dim=1).unsqueeze(1)
    #     # ordinal = (torch.abs(pred - label) + 1.0).pow(self.pow[0])
    #     # ce = torch.mean(ce * ordinal)
    #     # calculate the distance bias of pred_logits ans label_logits
    #     feat_pred = F.nll_loss(log_logits, pred.long().squeeze(), reduction='none', weight=weight).unsqueeze(1)
    #     feat_label = F.nll_loss(log_logits, label.long().squeeze(), reduction='none', weight=weight).unsqueeze(1)
    #     # bias = torch.abs(pred - label) * torch.norm(feat_pred - feat_label, dim=1, p=1,keepdim=True)
    #     bias = torch.norm(feat_pred - feat_label, dim=1, p=1,keepdim=True).pow(torch.abs(pred - label))
    #     bias = torch.where(bias==1, 0, bias)
    #     # diceloss = MultiDiceLoss(self.alpha)
    #     # diceloss = smp.losses.DiceLoss(mode='multiclass')
    #     # dice = diceloss(pred_logits, label.squeeze().long())
    #     # iou_loss = smp.losses.jaccard.JaccardLoss(mode='multiclass')
    #     # iou = iou_loss(pred_logits, label.long())
    #     return ce + bias.mean()
    
    def distance_smooth_label(self, label:Tensor):
        label = label.squeeze().long()
        cls = torch.abs(torch.arange(self.num_cls).unsqueeze(0) - torch.arange(self.num_cls).unsqueeze(1)).to(label)
        cls = torch.exp(-cls)
        out = cls.index_select(0, label.reshape(-1)).reshape(label.size() + (-1,))
        return out.permute(0, 3, 1, 2).float()

    def forward(self, pred_logits, label):
        self.num_cls = pred_logits.size(1)
        # weight = self.mask(label, self.num_cls) # mask
        # weight = weight[None, :, None, None]
        label_ = label.clone().detach()
        pred_logits_ = pred_logits.clone().detach()
        label_[label_==self.ignore_index] = 0
        dist_matrix = self.distance_smooth_label(label_)
        dist_matrix = dist_matrix * (label_.unsqueeze(1) != 0).float()
        ce = F.cross_entropy(pred_logits, dist_matrix, weight=torch.tensor([0,1,2,2,1]).to(label))
        logits_sm = torch.softmax(pred_logits_, dim=1)
        log_logits = torch.log(logits_sm)
        pred = torch.argmax(logits_sm, dim=1).unsqueeze(1)
        feat_pred = F.nll_loss(log_logits, pred.long().squeeze(), reduction='none').unsqueeze(1)
        feat_label = F.nll_loss(log_logits, label_.long().squeeze(), reduction='none').unsqueeze(1)
        # bias = torch.norm(feat_pred - feat_label, dim=1, p=2,keepdim=True).pow(torch.abs(pred - label_.unsqueeze(1)))
        bias = nn.MSELoss(reduction='none')(feat_pred, feat_label)
        bias[bias==1] = 0
        bias[label_.unsqueeze(1)==0] = 0
        # bias = torch.where(torch.isnan(bias), 0, bias)
        return ce + bias.mean()
    
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

    # def forward(self, pred_logits, label):
    #     self.num_cls = pred_logits.size(1)
    #     label_ = label.clone().detach()
    #     label_[label_==self.ignore_index] = 0
    # #     logits_sm = torch.softmax(pred_logits, dim=1)
    #     sampler = torch.tensor([50, 100, 400, 400, 200])
    #     logits_sm = self.balanced_softmax_loss(label_, pred_logits, sampler)
    #     logits_sm = torch.softmax(logits_sm, dim=1)
    #     log_logits = torch.log(logits_sm)
    #     dist_matrix = self.distance_smooth_label(label_)
    #     sim_matrix = self.base.pow(dist_matrix)
    #     ce = -log_logits * sim_matrix
    #     ce = (torch.sum(ce, dim=1) * (label_ != 0)).mean()
    #     return ce
    
    # def forward(self, pred_logits, label):
    #     self.num_cls = pred_logits.size(1)
    #     label_ = label.clone().detach()
    #     label_[label_==self.ignore_index] = 0
    #     dist_matrix = self.distance_smooth_label(label_)
    #     sim_matrix = self.base.pow(dist_matrix)
    #     sim_matrix = sim_matrix * (label_.unsqueeze(1) != 0).float()
    #     # ce = F.cross_entropy(pred_logits, sim_matrix).to(label)
    #     ce = F.cross_entropy(pred_logits, sim_matrix, weight=torch.tensor([0,1,2,2,1]).to(label))
    #     return ce
    def forward(self, pred_logits, label):
        self.num_cls = pred_logits.size(1)
        label_ = label.clone().detach()
        label_[label_==255] = 0
        dist_matrix = self.distance_smooth_label(label_)
        sim_matrix = self.base.pow(dist_matrix)
        sim_matrix = sim_matrix * (label_.unsqueeze(1) != 0).float()
        # ce = F.cross_entropy(pred_logits, sim_matrix).to(label)
        ce = F.cross_entropy(pred_logits, sim_matrix, weight=self.class_weight)
        # ce = F.cross_entropy(pred_logits, sim_matrix, weight=torch.tensor([0,1,2,2,1]).to(label))
        # ce = self.ohem_ce(pred_logits, sim_matrix)
        dice = self.dice(pred_logits, label_)
        return ce+dice
    
class OrdinalLogLoss(nn.Module):
    def __init__(self,ignore_index=-100):
        super(OrdinalLogLoss, self).__init__()
        self.ignore_index = ignore_index
        
    def distance_matrix(self, label:Tensor, num_cls:int):
        '''num_cls: int, the number of classes'''
        label = label.squeeze()
        assert label.ndim == 3, 'label should not be one-hot'
        dist = torch.abs(torch.arange(num_cls).unsqueeze(0) - torch.arange(num_cls).unsqueeze(1)).to(label)
        # replae label last dim with distance matrix by its value
        out = dist.index_select(0, label.reshape(-1)).reshape(label.size() + (-1,)).to(label)
        return out.permute(0, 3, 1, 2).contiguous().float()
    
    def forward(self, pred_logits, label):
        self.pow = nn.Parameter(torch.tensor(1.5).to(label).float())
        label_ = label.clone().detach()
        label_[label_==self.ignore_index] = 0
        logits_sm = F.softmax(pred_logits, dim=1)
        dist = self.distance_matrix(label_, 5) + 1.0
        log_logits = -torch.log(1-logits_sm)
        oll = log_logits * torch.pow(dist, self.pow)
        # oll = (log_logits * dist).pow(self.pow)
        oll = torch.sum(oll, dim=1).squeeze()
        oll[label_== 0] = 0
        return oll.mean()
    
# TODO: add HO2Loss
class HO2Loss(nn.Module):
    def __init__(self, ignore_index=255):
        super(HO2Loss, self).__init__()
        self.ignore_index = ignore_index
        self.omega = 0.05
    
    def entropy_loss(self, logits):
        # https://en.wikipedia.org/wiki/Entropy_(information_theory)
        P = F.softmax(logits, dim=1)
        logP = F.log_softmax(logits, dim=1)
        # N = P.shape[0]
        return -torch.sum(P * logP, dim=1).mean()

    def neighbor_loss(self, margin, pred_logits, label):
        # margin = torch.tensor(margin, device=device)
        P = F.softmax(pred_logits, dim=1)
        K = P.shape[1]
        loss = 0
        for k in range(K-1):
            # force previous probability to be superior to next
            reg_gt = (label >= k+1).float() * F.relu(margin+P[:,  k]-P[:, k+1])
            reg_lt = (label <= k).float() * F.relu(margin+P[:, k+1]-P[:, k])
            loss += torch.mean(reg_gt + reg_lt)
        return loss

    def loss(self, pred_logits, label):
        return self.entropy_loss(pred_logits) + self.neighbor_loss(self.omega, pred_logits, label)
    
    def forward(self, pred_logits, label):
        return self.loss(pred_logits, label)
    
class CO2(nn.Module):
    def __init__(self, ignore_index=255):
        super(CO2, self).__init__()
        self.ignore_index = ignore_index
        self.omega = 0.05

    def neighbor_loss(self, margin, pred_logits, label):
        # margin = torch.tensor(margin, device=device)
        P = F.softmax(pred_logits, dim=1)
        K = P.shape[1]
        loss = 0
        for k in range(K-1):
            # force previous probability to be superior to next
            reg_gt = (label >= k+1).float() * F.relu(margin+P[:,  k]-P[:, k+1])
            reg_lt = (label <= k).float() * F.relu(margin+P[:, k+1]-P[:, k])
            loss += torch.mean(reg_gt + reg_lt)
        return loss

    def forward(self, pred_logits, label):
        ce = F.cross_entropy(pred_logits, label, ignore_index=self.ignore_index)
        neighbor_loss =self.neighbor_loss(self.omega, pred_logits, label)
        return ce + neighbor_loss

class OrdinalEncoder(nn.Module):
    def __init__(self, ignore_index=255):
        super(OrdinalEncoder, self).__init__()
        self.ignore_index = ignore_index
        
    # def ordinal_encode(self, Yhat, Y):
    #     # if K=4, then
    #     #     Y=0 => Y_=[0, 0, 0]
    #     #     Y=1 => Y_=[1, 0, 0]
    #     #     Y=2 => Y_=[1, 1, 0]
    #     #     Y=3 => Y_=[1, 1, 1]
    #     KK = torch.arange(self.num_cls).expand(Y.shape[0], -1)
    #     YY = (Y[:, None] > KK).float()
    #     return F.binary_cross_entropy_with_logits(Yhat, YY)
    def to_label(self, y_true):
        assert y_true.ndim == 3, 'y_true should be 3D tensor'
        # if K=4, then
        #     Y=0 => Y_=[0, 0, 0]
        #     Y=1 => Y_=[1, 0, 0]
        #     Y=2 => Y_=[1, 1, 0]
        #     Y=3 => Y_=[1, 1, 1]
        ordinal_matrix = torch.arange(self.num_cls)[None, None, None, :].expand(*y_true.shape, -1).to(y_true)
        y_true = y_true.unsqueeze(-1)
        label = (y_true > ordinal_matrix).float().to(y_true)
        return label.permute(0, 3, 1, 2).contiguous().float()
    def to_valid_label(self, y_true):
        assert y_true.ndim == 3, 'y_true should be 3D tensor'
        # if K=4, then
        #     Y=0 => Y_=[0, 0, 0]
        #     Y=1 => Y_=[1, 0, 0]
        #     Y=2 => Y_=[1, 1, 0]
        #     Y=3 => Y_=[1, 1, 1]
        ordinal_matrix = torch.arange(1, self.num_cls)[None, None, None, :].expand(*y_true.shape, -1).to(y_true)
        y_true = y_true.unsqueeze(-1)
        label = ((y_true > ordinal_matrix) * (y_true < self.ignore_index-1)).float().to(y_true)
        return label.permute(0, 3, 1, 2).contiguous().float()

    def to_proba(self, Yhat):
        # we need to convert mass distribution into probabilities
        # i.e. P(Y >= k) into P(Y = k)
        # P(Y=0) = 1-P(Y≥1)
        # P(Y=1) = P(Y≥1)-P(Y≥2)
        # ...
        # P(Y=K-1) = P(Y≥K-1)
        Phat = torch.sigmoid(Yhat)
        Phat = torch.cat((1-Phat[:, :1], Phat[:, :-1] - Phat[:, 1:], Phat[:, -1:]), 1)
        return torch.clamp(Phat, 0, 1)

    def to_valid_proba(self, Yhat):
        Phat = torch.sigmoid(Yhat)
        return torch.clamp(Phat, 0, 1)
    
    def forward(self, pred_logits, label):
        self.num_cls = pred_logits.size(1)
        label_ = label.clone().detach()
        label_[label==self.ignore_index] = 0
        self.label = self.to_label(label_)
        # pred = self.to_proba(pred_logits)
        # pred = torch.sigmoid(pred_logits)
        loss = F.binary_cross_entropy_with_logits(pred_logits, self.label, reduction='none')
        loss = torch.sum(loss, dim=1)
        loss[label==self.ignore_index] = 0 
        return loss.mean()
    
    # def forward(self, pred_logits, raw_label, ordinal_label):
    #     # pred = self.to_proba(pred_logits)
    #     # pred = torch.sigmoid(pred_logits)
    #     loss = F.binary_cross_entropy_with_logits(pred_logits, ordinal_label, reduction='none')
    #     loss = torch.sum(loss, dim=1)
    #     loss[raw_label==self.ignore_index] = 0 
    #     return loss.mean()
    

    # def forward(self, pred_logits, label):
    #     self.num_cls = pred_logits.size(1)
    #     label_ = label.clone().detach()
    #     # label_[label_==self.ignore_index] = 0
    #     # label_ = label_-1
    #     label_ = self.to_valid_label(label_)
    #     pred = torch.sigmoid(pred_logits)
    #     loc_logits, clf_logits = pred.split([1, self.num_cls-1], dim=1)
    #     loss = F.binary_cross_entropy_with_logits(clf_logits, label_, reduction='none')
    #     loss = torch.sum(loss, dim=1)
    #     loss[label==self.ignore_index] = 0 
    #     loss[loc_logits.squeeze()>0.5] = 0
    #     return loss.mean()

class boundary_iou_loss(nn.Module):
    def __init__(self, dilation_ratio=0.005):
        super(boundary_iou_loss, self).__init__()
        self.dilation_ratio = dilation_ratio

    def generate_boundary_mask(self, label):
        # label: [H, W], binary mask where 1 is the object and 0 is the background
        # dilation_ratio: controls the thickness of the boundary
        label = label.detach().cpu().numpy().astype(np.uint8)
        kernel_size = int(min(label.shape) * self.dilation_ratio)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(label, kernel)
        eroded = cv2.erode(label, kernel)
        boundary = dilated - eroded
        return torch.tensor(boundary, dtype=torch.float32)

    def forward(self, pred, target):
        boundary_mask = self.generate_boundary_mask(target)
        boundary_mask = boundary_mask.to(pred.device)
        pred = torch.argmax(torch.sigmoid(pred), dim=1)
        # Apply boundary mask
        pred_boundary = pred * boundary_mask
        target_boundary = target * boundary_mask
        
        # Calculate IoU on boundary
        intersection = (pred_boundary * target_boundary).sum()
        union = pred_boundary.sum() + target_boundary.sum() - intersection
        iou = intersection / union
        return 1 - iou

class BlvLoss(nn.Module):
#cls_nufrequency_list
    def __init__(self, cls_num_list, sigma=4, loss_name='BlvLoss'):
        super(BlvLoss, self).__init__()
        cls_list = torch.tensor(cls_num_list, dtype=torch.float32).cuda()
        frequency_list = torch.log(cls_list)
        self.frequency_list = torch.log(torch.tensor(sum(cls_num_list))).cuda() - frequency_list
        self.reduction = 'mean'
        self.sampler = normal.Normal(0, sigma)
        self._loss_name = loss_name

    def reduce_loss(self, loss, reduction):
        """Reduce loss as specified.

        Args:
            loss (Tensor): Elementwise loss tensor.
            reduction (str): Options are "none", "mean" and "sum".

        Return:
            Tensor: Reduced loss tensor.
        """
        reduction_enum = F._Reduction.get_enum(reduction)
        # none: 0, elementwise_mean:1, sum: 2
        if reduction_enum == 0:
            return loss
        elif reduction_enum == 1:
            return loss.mean()
        elif reduction_enum == 2:
            return loss.sum()

    def weight_reduce_loss(self, loss, weight=None, reduction='mean', avg_factor=None):
        """Apply element-wise weight and reduce loss.

        Args:
            loss (Tensor): Element-wise loss.
            weight (Tensor): Element-wise weights.
            reduction (str): Same as built-in losses of PyTorch.
            avg_factor (float): Average factor when computing the mean of losses.

        Returns:
            Tensor: Processed loss values.
        """
        # if weight is specified, apply element-wise weight
        if weight is not None:
            assert weight.dim() == loss.dim()
            if weight.dim() > 1:
                assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
            loss = loss * weight

        # if avg_factor is not specified, just reduce the loss
        if avg_factor is None:
            loss = self.reduce_loss(loss, reduction)
        else:
            # if reduction is mean, then average the loss by avg_factor
            if reduction == 'mean':
                # Avoid causing ZeroDivisionError when avg_factor is 0.0,
                # i.e., all labels of an image belong to ignore index.
                eps = torch.finfo(torch.float32).eps
                loss = loss.sum() / (avg_factor + eps)
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif reduction != 'none':
                raise ValueError('avg_factor can not be used with reduction="sum"')
        return loss

    def forward(self, pred, target, ignore_index, weight=None, avg_factor=None, reduction_override=None):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        viariation = self.sampler.sample(pred.shape).clamp(-1, 1).to(pred.device)

        pred = pred + (viariation.abs().permute(0, 2, 3, 1) / self.frequency_list.max() * self.frequency_list).permute(0, 3, 1, 2)

        loss = F.cross_entropy(pred, target, reduction='none',  ignore_index=ignore_index)

        if weight is not None:
            weight = weight.float()

        loss = self.weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        return loss