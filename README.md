# GaoTong2021-Baseline

2021高通AI应用创新大赛-手绘图像识别赛道分享，这只是个baseline。

### 训练

```shell
bash /project/train/src_repo/run.sh
```

### 测试

```shell
# python 测试
# 调用src/ji.py
# 量化
cd */ev_sdk/model
bash convert_model.sh
```

### 结果分享

(`#`号后面为线上成绩，`#`号以前为通用设置，未说明表示条件相同)

- 基线模型

```yaml
backbone: resnet18
img_size: 224x224
epoch: 30
bs: 32
scheduler: cos
loss: focal
transforms/img_aug: 随机水平、垂直翻转，随机角度变换，随机裁剪
# 0.3496
```

- 修改图像尺寸，更换backbone

```yaml
img_size: 112x112
backbone: resnet18 # 0.6529
backbone: repvgg-a0 # 0.6739
```

- 修改预处理

```yaml
backbone: repvgg-a0
transforms/img_aug: 去除水平翻转 # 0.7725
transforms/img_aug: 去除角度变换 # 0.8014
```

- 增加训练轮次和部分参数

```yaml
epoch: 60
bs: 64
backbone: repvgg-a0 # 0.8398
backbone: repvgg-a1 # 0.8538
```

- 换大模型&其他改进

```yaml
backbone: repvgg-a1
transforms/img_aug: 随机裁剪变成resize # 0.9339
transforms/img_aug: 去除水平翻转 # 0.9281
##########
backbone: repvgg-b0
transforms/img_aug: 水平翻转、resize # 0.9381
##########
backbone: repvgg-a2
transforms/img_aug: 水平翻转、resize # 0.9436
```

- 更换Loss

```yaml
backbone: repvgg-a1
loss: ce # 0.9359
loss: focal # 0.9339
loss: label smoothing (alpha=0.2) # 0.9317
```

- 使用MixUp

```yaml
backbone: repvgg-a1
mixup: alpha=0.1 # *
mixup: alpha=0.2 # *
# 记不清了0.2比0.1高
```

- 加大图像尺寸

```yaml
backbone: repvgg-a2
loss: ce
img_size: 112x112 # 0.9436
# 后面结果有点杂分不清了
img_size: 192x192 # 0.944+
img_size: 224x224 # 0.946+
```

- 最终

```yaml
backbone: repvgg-a2
loss: ce
scheduler: cos
mixup: alpha=2
策略: 前30epoch使用mixup，后30epoch正常训练
transforms/img_aug: 水平翻转、resize
# 0.9503
```

### 其他

由于前期一直没有搞通snpe量化，中间闲置了一个月没有继续搞，上周突然通了，抓紧改了一下，模型主干基本定了就没有改，主要还是从Loss和数据入手，从0.8以后都是改数据预处理提升的，很多Trick不是很有效。

中间尝试了CenterLoss、MCLoss，以及魔改的别的Loss，效果不如只使用交叉熵好，同时主干权衡了一下性能分，没有换别的。在训练数据尝试了padding，在验证集提升很大，但是测试集反而掉点。

数据统计下来没有不平衡的问题，也没有去尝试更加discriminative的方法。

毕竟这只是Baseline，等其他大佬们的方案\_(:з」∠)\_

#### 参考链接

[1] https://blog.csdn.net/u013347145/article/details/115592697

[2] https://blog.csdn.net/sinat_38439143/article/details/116101664?spm=1001.2014.3001.5501

[3] https://github.com/DingXiaoH/RepVGG

[4] https://github.com/PRIS-CV/Mutual-Channel-Loss