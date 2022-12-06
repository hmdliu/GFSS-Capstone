
import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.blocks import DecoderSimple
from src.model.utils import LOSS_DICT

class MiBModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.norm_b = args.norm_b
        self.norm_s = args.norm_s
        self.norm_q = args.norm_q
        self.im_size = (args.image_size, args.image_size)

        self.inner_loss = LOSS_DICT[args.inner_loss]
        self.meta_loss = LOSS_DICT[args.meta_loss]
        self.wce_loss = LOSS_DICT['wce']

        assert os.path.isfile(args.resume_weight)
        pretrain_state_dict = torch.load(args.resume_weight)['state_dict']
        self.cls1 = DecoderSimple(n_cls=2, d_encoder=args.encoder_dim)
        self.cls2 = DecoderSimple(n_cls=args.num_classes_tr, d_encoder=args.encoder_dim)
        self.state_dict, base_cls_num = self.cls2.init_base(pretrain_state_dict)
        args.log_func(f"\n==> Base Classifier loaded with {base_cls_num} classes")

        # old model for mib kd loss
        self.cls_old = DecoderSimple(n_cls=args.num_classes_tr-1, d_encoder=args.encoder_dim)
        self.state_dict_old, base_cls_num = self.cls_old.init_base(pretrain_state_dict)
        args.log_func(f"\n==> Base Classifier (OLD) loaded with {base_cls_num} classes")
        self.cls_old.eval()

    def meta_params(self):
        return []

    @staticmethod
    def compute_weight(label, n_cls):
        try:
            count = torch.bincount(label.flatten())
            weight = torch.tensor([count[0]/count[i] for i in range(n_cls)])
        except:
            weight = torch.ones(n_cls, device=label.device)
        return weight

    def inner_loop(self, f_s, label_s, weight_s):

        # reset classifier
        self.cls1.reset_parameters()
        self.cls2.load_state_dict(self.state_dict)
        self.cls1.train()
        self.cls2.train()

        # init optimizer
        optimizer1 = torch.optim.SGD(self.cls1.parameters(), lr=self.args.lr_cls)
        optimizer2 = torch.optim.SGD(self.cls2.parameters(), lr=self.args.lr_cls)

        # adapt the classifier to current task
        for _ in range(self.args.adapt_iter):
            # cls 1: 2-way
            bpred_s = self.cls1(f_s, self.im_size)
            bloss_s = self.wce_loss(bpred_s, label_s, weight=weight_s)
            optimizer1.zero_grad()
            bloss_s.backward()
            optimizer1.step()
            # cls2: (B+2)-way
            pred_s = self.cls2(f_s, self.im_size)
            pred_old = self.cls_old(f_s, self.im_size)
            if self.args.inner_loss == "mib":
                loss_s = self.inner_loss(pred_s, label_s, pred_old, fg_idx=-1, weight=weight_s, kdl_weight=self.args.kdl_weight)
            else:
                loss_s = self.inner_loss(pred_s, label_s, fg_idx=-1, weight=weight_s)
            optimizer2.zero_grad()
            loss_s.backward()
            optimizer2.step()

    def forward(self, backbone, img_s, img_q, label_s, label_q, img_b, label_b, cls_b):

        # extract feats
        with torch.no_grad():
            f_s = backbone.extract_features(img_s)
            f_q = backbone.extract_features(img_q)
            f_b = backbone.extract_features(img_b)

        # init variables
        label, pred, loss = [], [], []
        weight_b = self.compute_weight(label_b, n_cls=2)
        weight_s = self.compute_weight(label_s, n_cls=2)
        weight_q = self.compute_weight(label_q, n_cls=2)

        # normalize feats as needed
        if self.norm_s:
            f_s = F.normalize(f_s, dim=1)
        if self.norm_q:
            f_q = F.normalize(f_q, dim=1)
        if self.norm_b:
            f_b = F.normalize(f_b, dim=1)

        # perform inner loop
        self.inner_loop(f_s, label_s, weight_s)

        # pred0: novel class with a 2-way classifier
        self.cls1.eval()
        bpred_q = self.cls1(f_q, self.im_size)
        label.append(label_q)
        pred.append(bpred_q)
        loss.append(self.wce_loss(bpred_q, label_q, weight=weight_q))

        # pred1: novel class with a (B+2)-way classifier
        self.cls2.eval()
        bpred_q, pred_q = self.cls2.forward_binary(f_q, self.im_size, fg_idx=-1)
        label.append(label_q)
        pred.append(bpred_q)
        loss.append(self.meta_loss(pred_q, label_q, fg_idx=-1, weight=weight_q))

        # pred1: base class with a (B+2)-way classifier
        bpred_b, pred_b = self.cls2.forward_binary(f_b, self.im_size, fg_idx=cls_b)
        label.append(label_b)
        pred.append(bpred_b)
        loss.append(self.meta_loss(pred_b, label_b, fg_idx=cls_b, weight=weight_b))
        
        return label, pred, loss