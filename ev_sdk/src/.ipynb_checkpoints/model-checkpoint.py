import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from repvgg import repvgg_model_convert, create_RepVGG_A0, create_RepVGG_A1, create_RepVGG_A2, create_RepVGG_B1


class BaseModel(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=32):
        super().__init__()
        self.model_name = model_name
        
        root = '/project/train/src_repo/models'
        if model_name == 'resnet18':
            net = models.resnet18()
            backbone = nn.Sequential(*list(net.children())[:-2])
            plane = 512
        elif model_name == 'rep-a0':
            net = create_RepVGG_A0(deploy=False)  
            backbone = nn.Sequential(*list(net.children())[:-2])
            plane = 1280
        elif model_name == 'rep-a1':
            net = create_RepVGG_A1(deploy=False)  
            backbone = nn.Sequential(*list(net.children())[:-2])
            plane = 1280
        elif model_name == 'rep-a2':
            net = create_RepVGG_A2(deploy=False) 
            backbone = nn.Sequential(*list(net.children())[:-2])
            plane = 1408
        elif model_name == 'rep-b1':
            net = create_RepVGG_B1(deploy=False) 
            backbone = nn.Sequential(*list(net.children())[:-2])
            plane = 2048
        else:
            raise ValueError('model is not support.')

        self.backbone = backbone
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.metric = nn.Linear(plane, num_classes, bias=False)

    def forward(self, x):
        feat = self.backbone(x)
        # gap = self.gap(feat)
        # gmp = self.gmp(feat)
        # feat = 0.5 * gap + 0.5 * gmp
        feat = self.gap(feat)
        feat_flat = torch.flatten(feat, 1)
        out = self.metric(feat_flat)
        
        return out


if __name__ == '__main__':
    model = BaseModel(model_name='rep-b1').eval()
    x = torch.randn((1, 3, 224, 224))
    out = model(x)
    print(out.size())
    print(model)