
import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.blocks import Attention, DecoderSimple
from src.model.utils import interpolate, convert_feats, LOSS_DICT
from src.utils import yield_params

class BaseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.norm_b = args.norm_b
        self.norm_s = args.norm_s
        self.norm_q = args.norm_q
        self.im_size = (args.image_size, args.image_size)

        self.inner_loss = LOSS_DICT[args.inner_loss]
        self.meta_loss = LOSS_DICT[args.meta_loss]

        assert os.path.isfile(args.resume_weight)
        pretrain_state_dict = torch.load(args.resume_weight)['state_dict']
        self.classifier = DecoderSimple(
            n_cls=args.num_classes_tr,
            d_encoder=args.encoder_dim,
            bias=('module.classifier.bias' in pretrain_state_dict)
        )
        self.state_dict = self.classifier.init_base(pretrain_state_dict)

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
        self.classifier.load_state_dict(self.state_dict)
        self.classifier.train()

        # init optimizer
        optimizer = torch.optim.SGD(self.classifier.parameters(), lr=self.args.lr_cls)

        # adapt the classifier to current task
        for _ in range(self.args.adapt_iter):

            # make prediction
            pred_s = self.classifier.forward_novel(f_s, self.im_size)
            
            # compute loss & update classifier weights
            loss_s = self.inner_loss(pred_s, label_s, weight=weight_s)
            optimizer.zero_grad()
            loss_s.backward()
            optimizer.step()

    def forward(self, backbone, img_s, img_q, label_s, label_q, img_b, label_b, cls_b):

        # extract feats
        with torch.no_grad():
            f_s = backbone.extract_features(img_s)
            f_q = backbone.extract_features(img_q)
            f_b = backbone.extract_features(img_b)

        # init variables
        pred, loss = [], []
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
        self.classifier.eval()
        # pred0: novel
        pred.append(self.classifier.forward_novel(f_q, self.im_size))
        loss.append(self.meta_loss(pred[-1], label_q, weight=weight_q))
        # pred1: base
        pred.append(self.classifier.forward_base(cls_b, f_b, self.im_size))
        loss.append(self.meta_loss(pred[-1], label_b, weight=weight_b))
        
        return pred, loss