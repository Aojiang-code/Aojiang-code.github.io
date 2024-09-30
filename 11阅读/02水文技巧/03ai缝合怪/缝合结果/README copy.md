## resnet50 
### resnet50 85.4%

```bash
python -m tensorboard.main --logdir=/work/home/aojiang/06项目复现/08Kansformer/code/DRresults/DR_grading/tensorboard/resnet/
```
[epoch 100] train_loss: 0.100  train_acc: 0.964  val_accuracy: 0.854

### train_resnet_DDR_L2.py
[epoch 100] train_loss: 0.338  train_acc: 0.875  val_accuracy: 0.843
### train_resnet_DDR_L2_RRP.py
[epoch 100] train_loss: 0.805  train_acc: 0.816  val_accuracy: 0.844

### resnet50 SGD
> 使用不同的学习率调整器
#### train_resnet_DDR_SGD_CosineAnnealingLR
[epoch 100] train_loss: 0.186  train_acc: 0.935  val_accuracy: 0.829

```bash
python -m tensorboard.main --logdir=/work/home/aojiang/06项目复现/08Kansformer/code/DRresults/DR_grading/SGD_CosineAnnealingLR/tensorboard/resnet/
```

#### train_resnet_DDR_SGD_lambdaLR
[epoch 100] train_loss: 0.230  train_acc: 0.915  val_accuracy: 0.834
```bash
python -m tensorboard.main --logdir=/work/home/aojiang/06项目复现/08Kansformer/code/DRresults/DR_grading/SGD_lambdaLR/tensorboard/resnet/
```
#### train_resnet_DDR_SGDyh:渐进式的学习率调整策略
[epoch 100] train_loss: 0.102  train_acc: 0.964  val_accuracy: 0.846
> 结合了两种主要的学习率调整技术：预热和余弦退火

```bash
python -m tensorboard.main --logdir=/work/home/aojiang/06项目复现/08Kansformer/code/DRresults/DR_grading/SGByh/tensorboard/resnet/
```

### SENet_01_train_resnet_DDR：SENet 
#### SENet_01_train_resnet_DDR:一般的SGB
优化器SGB，学习策略LambdaLR
[epoch 100] train_loss: 0.838  train_acc: 0.730  val_accuracy: 0.751
### PPA01 80.5

实例56：Adam 优化器，模型未运行

```bash

```

### WFEC
### WFEConv_02_train_resnet_DDR_L2_RRP.py
[epoch 100] train_loss: 0.821  train_acc: 0.809  val_accuracy: 0.830
#### Adam
[epoch 100] train_loss: 0.119  train_acc: 0.957  val_accuracy: 0.847
#### Adam_L2
WFEConv_02_train_resnet_DDR_Adam_L2.py
[epoch 100] train_loss: 0.378  train_acc: 0.862  val_accuracy: 0.834
#### SGB
[epoch 100] train_loss: 0.124  train_acc: 0.954  val_accuracy: 0.852
### HWA
实例51，无法运行


### HWD
#### Adam
HWD_01_train_resnet_DDR_Adam.py， 模型过拟合，已经没有跑下去的必要了
[epoch 80] train_loss: 0.175  train_acc: 0.940  val_accuracy: 0.830
### SCSA
#### Adam
模型已经过拟合了，已经没有跑下去的必要了
[epoch 79] train_loss: 0.124  train_acc: 0.956  val_accuracy: 0.849
#### Adam_L2
SCSA_02_train_resnet_DDR_Adam_L2.py
[epoch 100] train_loss: 0.337  train_acc: 0.876  val_accuracy: 0.842
### SWH: SCSA+ WFECone+ HWD
#### 01并行缝合
实例50
[epoch 100] train_loss: 0.376  train_acc: 0.861  val_accuracy: 0.824
[epoch 200] train_loss: 0.558  train_acc: 0.800  val_accuracy: 0.823
[epoch 200] train_loss: 0.571  train_acc: 0.795  val_accuracy: 0.815
#### 02并行缝合
实例49
[epoch 100] train_loss: 0.427  train_acc: 0.842  val_accuracy: 0.826
[epoch 200] train_loss: 0.559  train_acc: 0.795  val_accuracy: 0.815
## kansformer1

```bash
python -m tensorboard.main --logdir=/work/home/aojiang/06项目复现/08Kansformer/code/DRresults/DR_grading/tensorboard/kansformer1
```


绘图:
```python
path_to_events_file = '/work/home/aojiang/06项目复现/08Kansformer/code/DRresults/DR_grading/tensorboard/kansformer1'
output_folder = '/work/home/aojiang/06项目复现/08Kansformer/code/DRresults/plots/kansformer1'  # 指定输出文件夹
```


### 下图使用的数据集是`DR_grading`

![kansformer_DDR](kansformer1_training_progress_ddr.png)



### 下图使用的数据集是`DR_grading_new`
[epoch 100] train_loss: 1.155  train_acc: 0.555  val_accuracy: 0.550
实例53



## densent   77.5%


```bash
python -m tensorboard.main --logdir=/work/home/aojiang/06项目复现/08Kansformer/code/DRresults/DR_grading/tensorboard/densenet/
```


绘图：
```python
path_to_events_file = '/work/home/aojiang/06项目复现/08Kansformer/code/DRresults/DR_grading/tensorboard/densenet/'
output_folder = '/work/home/aojiang/06项目复现/08Kansformer/code/DRresults/plots/densenet'  # 指定输出文件夹
```



### 下图使用的数据集是`DR_grading`

![densent](densenet_training_progress_ddr.png)



### 下图使用的数据集是`DR_grading_new`

未运行

### SENet_01_train_densenet_DDR     62.1%


```bash
python -m tensorboard.main --logdir=/work/home/aojiang/06项目复现/08Kansformer/code/DRresults/DR_grading/SENet/tensorboard/densenet/
```


绘图：
```python
path_to_events_file = '/work/home/aojiang/06项目复现/08Kansformer/code/DRresults/DR_grading/SENet/tensorboard/densenet/'
output_folder = '/work/home/aojiang/06项目复现/08Kansformer/code/DRresults/plots/densenet_SENet'  # 指定输出文件夹
```



### 下图使用的数据集是`DR_grading`


![densenet_SENet](densenet_SENet_training_progress_ddr.png)

### 下图使用的数据集是`DR_grading_new`

实例54：
[epoch 100] train_loss: 1.363  train_acc: 0.568  val_accuracy: 0.583


## EfficientNet
不使用预训练参数，其准确率很低， train_EfficientNet_DDR_L2.py
[epoch 100] train_loss: 0.748  train_acc: 0.722  val_accuracy: 0.758
## Resnext
train_resnext_DDR_L2.py
[epoch 100] train_loss: 0.245  train_acc: 0.909  val_accuracy: 0.841




## 总结：
1. 所有模型中只有ResNet模型最好
2. 在ResNet模型的基础上进行改进
3. 目前的工作是筛选合适的优化器
4. Adam优化器效果和SGB优化器效果差不多，Adam优化器效果更好一点点
