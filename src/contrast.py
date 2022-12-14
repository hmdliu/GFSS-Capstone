
"""
Loss functions for base class pre-training
Credits: https://github.com/tfzhou/ContrastiveSeg/blob/main/lib/loss/loss_contrast.py
"""

from abc import ABC
from __future__ import (
    division,
    print_function,
    absolute_import
)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastCELoss(nn.Module):
    def __init__(self, args=None):
        super(ContrastCELoss, self).__init__()
        self.args = args
        self.loss_weight = self.args.get('loss_weight', 0.1)
        self.contrast_criterion = PixelContrastLoss(configer=args)

    def forward(self, logits, target, embedding=None):
        loss = CE_loss(self.args, logits, target, self.args.num_classes_tr)
        if embedding is not None:
            h, w = embedding.size(-2), embedding.size(-1)
            pred = F.interpolate(input=logits, size=(h,w), mode='bilinear', align_corners=True)
            pred = pred.argmax(1)
            loss_contrast = self.contrast_criterion(embedding, target, pred) 
            loss += self.loss_weight * loss_contrast
        return loss

class PixelContrastLoss(nn.Module, ABC):
    def __init__(self, configer):
        super(PixelContrastLoss, self).__init__()

        self.configer = configer
        self.bg_anchor = configer.get('bg_anchor', True)
        self.temperature = self.configer.get('temperature', 0.1)
        self.base_temperature = self.configer.get('base_temperature', 0.07)

        self.max_samples = self.configer.get('max_samples', 1024)
        self.max_views = self.configer.get('max_views', 100)
        self.ignore_label = 255

    def _hard_anchor_sampling(self, X, y_hat, y):       # y: pred, y_hat: GT (weird naming from original repo)
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).to(X.device)
        y_ = torch.zeros(total_classes, dtype=torch.float).to(y.device)

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)    # [T, 100, 256] -> [T100, 256]

        anchor_feature = contrast_feature
        anchor_count = contrast_count                                       # 100 every img every class

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # [400, 1]
        logits = anchor_dot_contrast - logits_max.detach()                  # [400, 400]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().to(labels_.device)
        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask
        logits_mask = torch.ones_like(mask).scatter_(1, torch.arange(anchor_num * anchor_count).view(-1, 1).to(mask.device), 0)
        mask = mask * logits_mask                                           # mask of anchor*pos samples pairs (logits_mask gets rid of self)

        neg_logits = torch.exp(logits) * neg_mask                           # neg pixel samples other than self
        neg_logits = neg_logits.sum(1, keepdim=True)                        # [400, 1], sum of exp(negative_sample) for all 400 anchors

        exp_logits = torch.exp(logits)                                      # [400, 400]
        log_prob = logits - torch.log(exp_logits + neg_logits)              # [400, 400]
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)          # [400]

        if not self.bg_anchor:
            non_bg_mask = (labels_.view(-1)!=0).repeat(anchor_count)
            mean_log_prob_pos = mean_log_prob_pos[non_bg_mask]

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels, (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])    # [B, n, 256]

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_)
        return loss

def CE_loss(args, logits, targets, num_classes):
    """
    inputs:  images  : shape [batch_size, C, h, w]
             logits : shape [batch_size, num_classes, h, w]
             targets : shape [batch_size, h, w]
    returns: loss: shape []
             logits: shape [batch_size]
             logits = model(images)
    """
    batch, h, w = targets.size()
    one_hot_mask = torch.zeros(batch, num_classes, h, w, device=targets.device)
    new_target = targets.clone().unsqueeze(1)
    new_target[new_target == 255] = 0

    one_hot_mask.scatter_(1, new_target, 1).long()
    if args.smoothing:
        eps = 0.1
        one_hot = one_hot_mask * (1 - eps) + (1 - one_hot_mask) * eps / (num_classes - 1)
    else:
        one_hot = one_hot_mask      # [batch_size, num_classes, h, w]

    loss = cross_entropy(logits, one_hot, targets)
    return loss

def cross_entropy(logits: torch.tensor, one_hot: torch.tensor, targets: torch.tensor, mean_reduce: bool = True,
                  ignore_index: int = 255) -> torch.tensor:
    """
    inputs: one_hot  : shape [batch_size, num_classes, h, w]
            logits : shape [batch_size, num_classes, h, w]
            targets : shape [batch_size, h, w]
    returns:loss: shape [batch_size] or [] depending on mean_reduce
    """
    assert logits.size() == one_hot.size()
    log_prb = F.log_softmax(logits, dim=1)
    non_pad_mask = targets.ne(ignore_index)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.masked_select(non_pad_mask)
    if mean_reduce:
        return loss.mean()      # average later
    else:
        return loss
