import os
import numpy as np
from utils import save_obj
from sklearn.model_selection import train_test_split
from collections import Counter
import logging

logging.basicConfig(level=logging.DEBUG)

data_root = '/home/data/7'

classes = os.listdir(data_root)

class_dict = {}
for i, j in enumerate(classes):
    class_dict[j] = i

with open('/project/train/log/log.txt', 'w')as f:
    for k, v in class_dict.items():
        f.write('{},{}\n'.format(k, v))
    
    
logging.info(class_dict)

fpath = []
labels = []
for i in classes:
    for j in os.listdir(os.path.join(data_root, i)):
        fpath.append(os.path.join(data_root, i, j))
        labels.append(class_dict[i])

# count
# logging.info(Counter(labels))
with open('/project/train/log/log.txt', 'a+')as f:
    for k, v in Counter(labels).items():
        f.write('{},{}\n'.format(k, v))

train_x, test_x, train_y, test_y = train_test_split(fpath, labels, test_size=0.1, random_state=2021, stratify=labels)

with open('train.txt', 'w')as f:
    for i, j in zip(train_x, train_y):
        f.write('{},{}\n'.format(i, j))
    
with open('test.txt', 'w')as f:
    for i, j in zip(test_x, test_y):
        f.write('{},{}\n'.format(i, j))

