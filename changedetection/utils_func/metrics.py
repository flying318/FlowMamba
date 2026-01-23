import numpy as np
from copy import deepcopy
from sklearn.metrics import auc, confusion_matrix

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.longlong)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-7)
        mAcc = np.nanmean(Acc)
        return mAcc, Acc

    def Pixel_Precision_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        Pre = self.confusion_matrix[1, 1] / (self.confusion_matrix[0, 1] + self.confusion_matrix[1, 1])
        return Pre

    def Pixel_Recall_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1])
        return Rec
    
    def Pixel_Precision_Rate_multi_class(self):
        num_classes = self.confusion_matrix.shape[0]
        precision_per_class = []

        for i in range(1, num_classes):  # 从 1 开始，排除背景类别
            true_positive = self.confusion_matrix[i, i]
            predicted_positive = self.confusion_matrix[:, i].sum()
            if predicted_positive == 0:
                precision = 0.0
            else:
                precision = true_positive / predicted_positive
            precision_per_class.append(round(precision, 4))

        return precision_per_class
    
    def Pixel_Recall_Rate_multi_class(self):
        num_classes = self.confusion_matrix.shape[0]
        recall_per_class = []
    
        for i in range(1, num_classes):  # 从 1 开始，排除背景类别
            true_positive = self.confusion_matrix[i, i]
            actual_positive = self.confusion_matrix[i, :].sum()
            if actual_positive == 0:
                recall = 0.0
            else:
                recall = true_positive / actual_positive
            recall_per_class.append(round(recall, 4))
    
        return recall_per_class

    def Pixel_F1_score(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.Pixel_Recall_Rate()
        Pre = self.Pixel_Precision_Rate()
        F1 = 2 * Rec * Pre / (Rec + Pre)
        return F1


    def calculate_per_class_metrics(self):
        # Adjustments to exclude class 0 in calculations
        TPs = np.diag(self.confusion_matrix)[1:]  # Start from index 1 to exclude class 0
        FNs = np.sum(self.confusion_matrix, axis=1)[1:] - TPs
        FPs = np.sum(self.confusion_matrix, axis=0)[1:] - TPs
        return TPs, FNs, FPs
    
    def Damage_F1_score(self):
        TPs, FNs, FPs = self.calculate_per_class_metrics()
        precisions = TPs / (TPs + FPs + 1e-7)
        recalls = TPs / (TPs + FNs + 1e-7)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)
        return f1_scores
    
    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix) + 1e-7)
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Intersection_over_Union(self):
        IoU = self.confusion_matrix[1, 1] / (
                self.confusion_matrix[0, 1] + self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1])
        return IoU

    def Kappa_coefficient(self):
        # Number of observations (total number of classifications)
        # num_total = np.array(0, dtype=np.long)
        # row_sums = np.array([0, 0], dtype=np.long)
        # col_sums = np.array([0, 0], dtype=np.long)
        # total += np.sum(self.confusion_matrix)
        # # Observed agreement (i.e., sum of diagonal elements)
        # observed_agreement = np.sum(np.diag(self.confusion_matrix))
        # # Compute expected agreement
        # row_sums += np.sum(self.confusion_matrix, axis=0)
        # col_sums += np.sum(self.confusion_matrix, axis=1)
        # expected_agreement = np.sum((row_sums * col_sums) / total)
        num_total = np.sum(self.confusion_matrix)
        observed_accuracy = np.trace(self.confusion_matrix) / num_total
        expected_accuracy = np.sum(
            np.sum(self.confusion_matrix, axis=0) / num_total * np.sum(self.confusion_matrix, axis=1) / num_total)

        # Calculate Cohen's kappa
        kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
        return kappa

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int64') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def uoc_index(self):
        return imbalanced_ordinal_classification_index(self.confusion_matrix)


def imbalanced_ordinal_classification_index(conf_mat, beta=None, missing='zeros', verbose=False):
    # missing: 'zeros', 'uniform', 'diagonal'
   
    N = int(np.sum(conf_mat))
    K = float(conf_mat.shape[0])
    gamma = 1.0
    if beta is None:
        beta_vals = np.linspace(0.0, 1.0, 1000).transpose()
    else:
        beta_vals = [beta]
           
    # Fixing missing classes
    conf_mat_fixed = deepcopy(conf_mat)
    for ii in range(conf_mat.shape[0]):
        if np.sum(conf_mat[ii,:]) == 0:
            if missing == 'zeros':
                K -= 1.0  # Dealt with by 0**Nr[rr]
            elif missing == 'uniform':
                conf_mat_fixed[ii,:] = np.ones((1,conf_mat.shape[1]))
            elif missing == 'diagonal':
                conf_mat_fixed[ii,ii] = 1
            else:
                raise ValueError('Unknown way of dealing with missing classes.')
           
    # Computing number of samples in each class
    Nr = np.sum(conf_mat_fixed, axis=1)
   
    beta_oc = list()
   
    # Computing total dispersion and helper matrices
    helper_mat2 = np.zeros(conf_mat_fixed.shape)
    for rr in range(conf_mat_fixed.shape[0]):
        for cc in range(conf_mat_fixed.shape[1]):
            helper_mat2[rr, cc] = (float(conf_mat_fixed[rr, cc])/(Nr[rr] + 0**Nr[rr]) * ((abs(rr-cc))**gamma))
    total_dispersion = np.sum(helper_mat2)**(1/gamma)
    helper_mat1 = np.zeros(conf_mat_fixed.shape)
    for rr in range(conf_mat_fixed.shape[0]):
        for cc in range(conf_mat_fixed.shape[1]):
            helper_mat1[rr, cc] = float(conf_mat_fixed[rr, cc])/(Nr[rr] + 0**Nr[rr])
    helper_mat1 = np.divide(helper_mat1, total_dispersion + K)
   
    for beta in beta_vals:
       
        beta = beta/K
       
        # Creating error matrix and filling first entry
        error_mat = np.zeros(conf_mat_fixed.shape)
        error_mat[0, 0] = 1 - helper_mat1[0, 0] + beta*helper_mat2[0, 0]

        # Filling column 0
        for rr in range(1, conf_mat_fixed.shape[0]):
            cc = 0
            error_mat[rr, cc] = error_mat[rr-1, cc] - helper_mat1[rr, cc] + beta*helper_mat2[rr, cc]

        # Filling row 0
        for cc in range(1, conf_mat_fixed.shape[1]):
            rr = 0
            error_mat[rr, cc] = error_mat[rr, cc-1] - helper_mat1[rr, cc] + beta*helper_mat2[rr, cc]

        # Filling the rest of the error matrix
        for cc in range(1, conf_mat_fixed.shape[1]):
            for rr in range(1, conf_mat_fixed.shape[0]):
                cost_up = error_mat[rr-1, cc]
                cost_left = error_mat[rr, cc-1]
                cost_lefttop = error_mat[rr-1, cc-1]
                aux = np.min([cost_up, cost_left, cost_lefttop])
                error_mat[rr, cc] = aux - helper_mat1[rr, cc] + beta*helper_mat2[rr, cc]
       
        beta_oc.append(error_mat[-1, -1])
   
    if len(beta_vals) < 2:
        return beta_oc[0]
    else:
        return auc(beta_vals, beta_oc)
    
def wilson_index(Y, Yhat):
    return imbalanced_ordinal_classification_index(confusion_matrix(Y, Yhat))