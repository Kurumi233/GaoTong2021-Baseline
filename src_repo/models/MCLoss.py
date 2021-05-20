import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class SimpleMCLoss(nn.Module):
    def __init__(self, num_classes, per_class=3, p=0.4, alpha=1.5, beta=20):
        super().__init__()
        self.num_classes = num_classes
        self.per_class = per_class
        self.p = p
        self.alpha = alpha
        self.beta = beta
        self.celoss = nn.CrossEntropyLoss()
        self._gen_mask()

    def forward(self, feat, targets):
        n, c, h, w = feat.size()
        # L_div
        features = torch.softmax(feat.view(n, c, -1), dim=2)
        features = F.max_pool2d(features, kernel_size=(self.per_class, 1), stride=(self.per_class, 1))
        L_div = 1.0 - features.sum(dim=2).mean() / self.per_class

        # L_dis
        mask = self._gen_mask()
        if feat.is_cuda: mask = mask.cuda()
        features = (mask * feat).view(n, c, -1)
        features = F.max_pool2d(features, kernel_size=(self.per_class, 1), stride=(self.per_class, 1))
        features = F.avg_pool2d(features, kernel_size=(1, h*w), stride=(1, 1)).view(n, -1)
        L_dis = self.celoss(features, targets)

        return self.alpha * L_dis + self.beta * L_div

    def _gen_mask(self):
        drop_num = int(self.per_class * self.p)
        mask = np.ones(self.num_classes * self.per_class, dtype=np.float32)
        drop_idx = []
        for j in range(self.num_classes):
            drop_idx.append(np.random.choice(np.arange(self.per_class), size=drop_num, replace=False) + j * self.per_class)
        mask[drop_idx] = 0.

        return torch.from_numpy(mask).view(1, -1, 1, 1)