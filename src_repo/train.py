import os
import time
import random
import argparse
import shutil
import numpy as np
import logging

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataload import Dataset

from models.model import BaseModel
from models.Losses import CenterLoss, FocalLoss, LabelSmoothing
from models.OLS import OnlineLabelSmoothing
from models.MCLoss import SimpleMCLoss

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='rep-a2', type=str)
parser.add_argument('--savepath', default='/project/train/models/final', type=str)
parser.add_argument('--loss', default='fl', type=str)
parser.add_argument('--num_classes', default=32, type=int)
parser.add_argument('--pool_type', default='avg', type=str)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--scheduler', default='cos', type=str)
parser.add_argument('--resume', default=None, type=str)
# parser.add_argument('--resume', default='./Test/best.pth', type=str)
parser.add_argument('--lr_step', default=30, type=int)
parser.add_argument('--warm', default=5, type=int)
parser.add_argument('--print_step', default=50, type=int)
parser.add_argument('--lr_gamma', default=0.5, type=float)
parser.add_argument('--total_epoch', default=60, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--multi-gpus', default=1, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--seed', default=2021, type=int)
parser.add_argument('--pretrained', default=1, type=int)
parser.add_argument('--use_mixup', default=1, type=int)

args = parser.parse_args()


def train():
    model.train()
    
    epoch_loss = 0
    correct = 0.
    total = 0.
    t1 = time.time()
    s1 = time.time()
    for idx, (data, labels) in enumerate(trainloader):
        data, labels = data.to(device), labels.long().to(device)
        
        if args.use_mixup and epoch < 30:
            lam = np.random.beta(0.2, 0.2)
            index = torch.randperm(data.size(0)).cuda()
            data = lam * data + (1 - lam) * data[index, :]
            labels_a, labels_b = labels, labels[index]
            
            out, feat, feat_flat = model(data)
            loss = lam * criterion(out, labels_a) + (1 - lam) * criterion(out, labels_b)
        else:
            out, feat, feat_flat = model(data)
            loss = criterion(out, labels)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * data.size(0)
        total += data.size(0)
        _, pred = torch.max(out, 1)
        correct += pred.eq(labels).sum().item()

        if idx % args.print_step == 0:
            s2 = time.time()
            logging.info(f'idx:{idx:>3}/{len(trainloader)}, loss:{epoch_loss / total:.4f}, acc@1:{correct / total:.4f}, time:{s2 - s1:.2f}s')
            s1 = time.time()

    acc = correct / total
    loss = epoch_loss / total

    logging.info(f'loss:{loss:.4f} acc@1:{acc:.4f} time:{time.time() - t1:.2f}s')

    with open('/project/train/log/log.txt', 'a+')as f:
        f.write('loss:{:.4f}, acc:{:.4f}, time:{:.2f}->'.format(loss, acc, time.time() - t1))

    return {'loss': loss, 'acc': acc}


def test(epoch):
    model.eval()

    epoch_loss = 0
    correct = 0.
    total = 0.
    with torch.no_grad():
        for idx, (data, labels) in enumerate(valloader):
            data, labels = data.to(device), labels.long().to(device)
            out = model(data)
            loss = criterion(out, labels)

            epoch_loss += loss.item() * data.size(0)
            total += data.size(0)
            _, pred = torch.max(out, 1)
            correct += pred.eq(labels).sum().item()

        acc = correct / total
        loss = epoch_loss / total

        logging.info(f'test loss:{loss:.4f} acc@1:{acc:.4f}')

    global best_acc, best_epoch
    
    state = {
        'net': model.state_dict(),
        'acc': acc,
        'epoch': epoch
    }

    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch

        torch.save(state, os.path.join(savepath, 'best.pth'))

    torch.save(state, os.path.join(savepath, 'last.pth'))

    with open('/project/train/log/log.txt', 'a+')as f:
        f.write('epoch:{}, loss:{:.4f}, acc:{:.4f}\n'.format(epoch, loss, acc))

    return {'loss': loss, 'acc': acc}


def plot(d, mode='train', best_acc_=None):
    plt.figure(figsize=(10, 4))
    plt.suptitle('%s_curve' % mode)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    epochs = len(d['acc'])

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(epochs), d['loss'], label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(epochs), d['acc'], label='acc')
    if best_acc_ is not None:
        plt.scatter(best_acc_[0], best_acc_[1], c='r')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend(loc='upper left')

    plt.savefig(os.path.join('/project/train/models/final', '%s.jpg' % mode), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    best_epoch = 0
    best_acc = 0.
    use_gpu = False

    if args.seed is not None:
        print('use random seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = False

    if torch.cuda.is_available():
        use_gpu = True
        cudnn.benchmark = True

    # loss
    criterion = nn.CrossEntropyLoss()
#     centerloss = CenterLoss(num_classes=args.num_classes, feat_dim=512)
#     mcloss = SimpleMCLoss(num_classes=args.num_classes, per_class=44, p=0.4, alpha=1.5, beta=20)
#     criterion = FocalLoss()
#     criterion = LabelSmoothing(smoothing=0.1)

    # dataloader
    trainset = Dataset(mode='train')
    valset = Dataset(mode='test')

    trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, \
                             num_workers=args.num_workers, pin_memory=True, drop_last=True)

    valloader = DataLoader(dataset=valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, \
                           pin_memory=True)

    # model
    model = BaseModel(model_name=args.model_name, num_classes=args.num_classes, pretrained=args.pretrained,
                          pool_type=args.pool_type)

    if args.resume:
        state = torch.load(args.resume)
        print('resume from:{}'.format(args.resume))
        print('best_epoch:{}, best_acc:{}'.format(state['epoch'], state['acc']))
        model.load_state_dict(state['net'], strict=False)
        best_acc = state['acc']
    
    device = ('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # optim
    optimizer = torch.optim.SGD(
        [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}],
        weight_decay=args.weight_decay, momentum=args.momentum)

    logging.info('init_lr={}, weight_decay={}, momentum={}'.format(args.lr, args.weight_decay, args.momentum))

    if args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma, last_epoch=-1)
    elif args.scheduler == 'multi':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=args.lr_gamma, last_epoch=-1)
    elif args.scheduler == 'cos':
        warm_up_step = args.warm
        lambda_ = lambda epoch: (epoch + 1) / warm_up_step if epoch < warm_up_step else 0.5 * (
                np.cos((epoch - warm_up_step) / (args.total_epoch - warm_up_step) * np.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_)

    # savepath
    savepath = args.savepath
    logging.info('savepath: %s' % (savepath))

    os.makedirs(savepath, exist_ok=True)
    os.makedirs('/project/train/log', exist_ok=True)
    
    with open(os.path.join(savepath, 'setting.txt'), 'w')as f:
        for k, v in vars(args).items():
            f.write('{}:{}\n'.format(k, v))

#     f = open(os.path.join(savepath, 'log.txt'), 'w')
#     f.close()

    total = args.total_epoch
    start = time.time()

    train_info = {'loss': [], 'acc': []}
    test_info = {'loss': [], 'acc': []}

    for epoch in range(total):
        logging.info('epoch[{:>3}/{:>3}]'.format(epoch, total))
        d_train = train()
        scheduler.step()
        d_test = test(epoch)

        for k in train_info.keys():
            train_info[k].append(d_train[k])
            test_info[k].append(d_test[k])

        plot(train_info, mode='train')
        plot(test_info, mode='test', best_acc_=[best_epoch, best_acc])

    end = time.time()
    logging.info('total time:{}m{:.2f}s'.format((end - start) // 60, (end - start) % 60))
    logging.info('best_epoch:{}'.format(best_epoch))
    logging.info('best_acc:{}'.format(best_acc))
    with open('/project/train/log/log.txt', 'a+')as f:
        f.write('# best_acc:{:.4f}, best_epoch:{}'.format(best_acc, best_epoch))
    
#     shutil.move(os.path.join(savepath, 'log.txt'), '/project/train/log/log.txt')