# -*- coding: utf-8 -*-
# @Time   : 2024/08/26 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineContrastiveLoss(nn.Module):
    def __init__(self, margin=0, negative_weight=None):
        """
        :param margin: float, margin in CosineContrastiveLoss
        :param num_negs: int, number of negative samples
        :param negative_weight:, float, the weight set to the negative samples. When negative_weight=None, it
            equals to num_negs
        """
        super(CosineContrastiveLoss, self).__init__()
        self._margin = margin
        self._negative_weight = negative_weight

    def forward(self, y_pred, y_true):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs)
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        pos_logits = y_pred[:, 0]
        pos_loss = torch.relu(1 - pos_logits)
        neg_logits = y_pred[:, 1:]
        neg_loss = torch.relu(neg_logits - self._margin)
        if self._negative_weight:
            loss = pos_loss + neg_loss.mean(dim=-1) * self._negative_weight
        else:
            loss = pos_loss + neg_loss.sum(dim=-1)
        return loss.mean()


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs)
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        pos_logits = y_pred[:, 0]
        pos_loss = torch.pow(pos_logits - 1, 2) / 2
        neg_logits = y_pred[:, 1:]
        neg_loss = torch.pow(neg_logits, 2).sum(dim=-1) / 2
        loss = pos_loss + neg_loss
        return loss.mean()


class PairwiseLogisticLoss(nn.Module):
    def __init__(self):
        super(PairwiseLogisticLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = y_pred[:, 0].unsqueeze(-1)
        neg_logits = y_pred[:, 1:]
        logits_diff = pos_logits - neg_logits
        loss = -torch.log(torch.sigmoid(logits_diff)).mean()
        return loss


class PairwiseMarginLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        :param num_negs: number of negative instances in bpr loss.
        """
        super(PairwiseMarginLoss, self).__init__()
        self._margin = margin

    def forward(self, y_pred, y_true):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = y_pred[:, 0].unsqueeze(-1)
        neg_logits = y_pred[:, 1:]
        loss = torch.relu(self._margin + neg_logits - pos_logits).mean()
        return loss


class SigmoidCrossEntropyLoss(nn.Module):
    def __init__(self):
        """
        :param num_negs: number of negative instances in bpr loss.
        """
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        :param y_true: Labels
        :param y_pred: Predicted result
        """
        logits = y_pred.flatten()
        labels = y_true.flatten()
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
        return loss


class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self):
        """
        :param num_negs: number of negative instances in bpr loss.
        """
        super(SoftmaxCrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        probs = F.softmax(y_pred, dim=1)
        hit_probs = probs[:, 0]
        loss = -torch.log(hit_probs).mean()
        return loss


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, positive_scores, negative_scores):
        """
        Bayesian Personalized Ranking (BPR) Loss.

        Args:
        positive_scores (torch.Tensor): Scores for positive samples.
        negative_scores (torch.Tensor): Scores for negative samples.

        Returns:
        torch.Tensor: BPR loss.
        """
        # 计算正样本和负样本之间的差异
        difference = positive_scores - negative_scores

        # 应用sigmoid函数
        sigmoid_difference = torch.sigmoid(difference)

        # 计算BPR损失
        loss = -torch.log(sigmoid_difference).mean()

        return loss
