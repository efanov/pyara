"""
Module to compute metrics of module
"""
import numpy as np


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_det_curve(target_scores, nontarget_scores):
    """Compute det_curve"""
    n_scores = target_scores.size + nontarget_scores.size  # sum of sizes
    all_scores = np.concatenate((target_scores, nontarget_scores))  # vector of scores
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))  # vextor of labels
    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')  # retern indexes of sorted array of zeros and ones
    labels = labels[
        indices]  # and sort labels like (zeros and ones) for all 0 in both arrays and (zeros + ones) of 1 in both arrays

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)  # array with max element N
    # (np.arange(1, n_scores + 1, step = 1) - tar_trial_sums | gives array of element with max el = N
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1, step=1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)  # return index of min element
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def min_tDCF(confusion_matrix):
    # 0 - Real | 1 - Sintesized
    # FAR - поставили 0 вместо 1
    # FRR - поставили 1 вместо 0
    Pfar = confusion_matrix[1][0] / (
                confusion_matrix[1][0] + confusion_matrix[0][1] + confusion_matrix[0][0] + confusion_matrix[1][1])
    Pfrr = confusion_matrix[0][1] / (
                confusion_matrix[1][0] + confusion_matrix[0][1] + confusion_matrix[0][0] + confusion_matrix[1][1])
    mDCF = (0.01 * Pfar + 0.1 * Pfrr) * 100
    return mDCF
