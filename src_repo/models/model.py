import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.repvgg import repvgg_model_convert, create_RepVGG_A0, create_RepVGG_A1, create_RepVGG_B0, create_RepVGG_A2
from models.metrics import ArcMarginProduct
from models.OSLayer import OSLayer


class BaseModel(nn.Module):
    def __init__(self, model_name, num_classes=32, pretrained=True, pool_type='avg'):
        super().__init__()
        self.model_name = model_name
        
        root = '/project/train/src_repo/models'
        if model_name == 'resnet18':
            net = models.resnet18()
            if pretrained:
                net.load_state_dict(torch.load(os.path.join(root, 'resnet18.pth')))
            backbone = nn.Sequential(*list(net.children())[:-2])
            plane = 512
        elif model_name == 'rep-a0':
            net = create_RepVGG_A0(deploy=False) 
            if pretrained:
                net.load_state_dict(torch.load(os.path.join(root, 'repvggA0.pth')))
            backbone = nn.Sequential(*list(net.children())[:-2])
            plane = 1280
        elif model_name == 'rep-a1':
            net = create_RepVGG_A1(deploy=False) 
            if pretrained:
                net.load_state_dict(torch.load(os.path.join(root, 'repvggA1.pth')))
            backbone = nn.Sequential(*list(net.children())[:-2])
            plane = 1280
        elif model_name == 'rep-a2':
            net = create_RepVGG_A2(deploy=False) 
            if pretrained:
                net.load_state_dict(torch.load(os.path.join(root, 'repvggA2.pth')))
            backbone = nn.Sequential(*list(net.children())[:-2])
            plane = 1408
        elif model_name == 'rep-b0':
            net = create_RepVGG_B0(deploy=False) 
            if pretrained:
                net.load_state_dict(torch.load(os.path.join(root, 'repvggB0.pth')))
            backbone = nn.Sequential(*list(net.children())[:-2])
            plane = 1280
        else:
            raise ValueError('model is not support.')

        self.backbone = backbone
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        
        self.metric = nn.Linear(plane, num_classes, bias=False)
#         self.metric = OSLayer(plane, num_classes, bias=False)

    def forward(self, x):
        feat = self.backbone(x)
        feat_pool = self.gap(feat)
#         gmp = self.gmp(feat)
        # feat_pool = self.pool(feat)
#         feat_pool = 0.5 * gap + 0.5* gmp
        feat_flat = torch.flatten(feat_pool, 1)
        # feat_flat = self.linear(feat_flat)
        out = self.metric(feat_flat)
        
        if self.training:
            return out, feat, feat_flat
        return out


if __name__ == '__main__':
    model = BaseModel(model_name='resnet18').eval()
    x = torch.randn((1, 3, 224, 224))
    out = model(x)
    print(out.size())
    print(model)