import random
import numpy as np
import os
import torch
from torch.utils.data import dataset
from PIL import Image
from torchvision import transforms, models
from utils import load_obj
import random

DATA_ROOT = '/home/data/7'

size = 256
trans = {
    'train':
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((size, size)),
#             transforms.RandomCrop((size, size), padding=4, padding_mode='edge'),
#             transforms.RandomResizedCrop(size=(size, size), scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
    'test':
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
}

class_dict = load_obj('class_dict.pkl')

class Dataset(dataset.Dataset):
    def __init__(self, mode):
        assert mode in ['train', 'test'], print('Mode - {} is not support.'.format(mode))
        txt = '%s.txt' % mode
        
        fpath = []
        labels = []
        self.mode = mode
        with open(txt, 'r')as f:
            for i in f.readlines():
                fp, l = i.strip().split(',')
                fpath.append(fp)
                labels.append(int(l))
        
        print(len(labels))
        
        self.fpath = fpath
        self.labels = labels
        self.trans = trans[mode]
        
    def __getitem__(self, index):
        img = Image.open(self.fpath[index])
        label = self.labels[index]
        
#         if self.mode is 'train':
#             random_pad = tuple(random.choices(list(range(0, 6)), k=4))
#             img = transforms.Pad(padding=random_pad, fill=(255, 255, 255))(img)
        img = self.trans(img)
        
        return img, label
    
    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    data = Dataset('train')
    print(len(data))