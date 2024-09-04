# 创建dataloder

## GTP提示词
### 一、
我需要你帮我写一份python代码，接下来我将告诉你一些信息，请你记忆

### 二、
我有一个数据集，名为03Diabetic_Retinopathy_Dataset_new
它的路径为：
/work/home/aojiang/06项目复现/08Kansformer/code/data/03Diabetic_Retinopathy_Dataset_new
03Diabetic_Retinopathy_Dataset_new这个文件夹中包含两个文件夹，分别是train和val
train文件夹和val文件夹中都包含五个文件夹，这五个文件夹的名称分别为Healthy、Mild DR、Moderate DR、Proliferate DR、Severe DR
这五个文件夹中都包含一些图片

请您记忆上述信息


### 三、
对于数据集03Diabetic_Retinopathy_Dataset_new
我使用dataload_five_flower.py加载03Diabetic_Retinopathy_Dataset_new这个数据集
dataload_five_flower.py的路径为/work/home/aojiang/06项目复现/08Kansformer/code/dataload/dataload_five_flower.py

dataload_five_flower.py中的代码如下：
from PIL import Image
from matplotlib.cbook import ls_mapper
import torch
from torch.utils.data import Dataset
import random
import os

class Five_Flowers_Load(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path 
        self.transform = transform

        random.seed(0)  # 保证随机结果可复现
        assert os.path.exists(data_path), "dataset root: {} does not exist.".format(data_path)

        # 遍历文件夹，一个文件夹对应一个类别
        flower_class = [cla for cla in os.listdir(os.path.join(data_path))] 
        self.num_class = len(flower_class)
        # 排序，保证顺序一致
        flower_class.sort()
        # 生成类别名称以及对应的数字索引  {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
        class_indices = dict((cla, idx) for idx, cla in enumerate(flower_class)) 

        self.images_path = []  # 存储训练集的所有图片路径
        self.images_label = []  # 存储训练集图片对应索引信息 
        self.images_num = []  # 存储每个类别的样本总数
        supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
        # 遍历每个文件夹下的文件
        for cla in flower_class:
            cla_path = os.path.join(data_path, cla)
            # 遍历获取supported支持的所有文件路径
            images = [os.path.join(data_path, cla, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
            # 获取该类别对应的索引
            image_class = class_indices[cla]
            # 记录该类别的样本数量
            self.images_num.append(len(images)) 
            # 写入列表
            for img_path in images: 
                self.images_path.append(img_path)
                self.images_label.append(image_class)

        print("{} images were found in the dataset.".format(sum(self.images_num))) 

 

    def __len__(self):
        return sum(self.images_num)
    
    def __getitem__(self, idx):
        img = Image.open(self.images_path[idx])
        label = self.images_label[idx]
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[idx]))
        if self.transform is not None:
            img = self.transform(img)
        else:
            raise ValueError('Image is not preprocessed')
        return img, label
    
    # 非必须实现，torch里有默认实现；该函数的作用是: 决定一个batch的数据以什么形式来返回数据和标签
    # 官方实现的default_collate可以参考
    # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0) 
        labels = torch.as_tensor(labels)  
        return images, labels
 
请你记忆上述信息


### 四、

下述代码应用了dataload_five_flower.py，请你了解和熟悉dataload_five_flower.py的使用方法，以便我命令你再创建新的python代码时，可以借鉴和参考。

# -*- coding: utf-8 -*-
############################################################################################################
# 增添了以下功能：
# 1. 使用argparse类实现可以在训练的启动命令中指定超参数
# 2. 可以通过在启动命令中指定 --seed 来固定网络的初始化方式，以达到结果可复现的效果
# 3. 使用了更高级的学习策略 cosine warm up：在训练的第一轮使用一个较小的lr（warm_up），从第二个epoch开始，随训练轮数逐渐减小lr。 
# 4. 可以通过在启动命令中指定 --model 来选择使用的模型 
# 5. 使用amp包实现半精度训练，在保证准确率的同时尽可能的减小训练成本
# 6. 实现了数据加载类的自定义实现
# 7. 可以通过在启动命令中指定 --tensorboard 来进行tensorboard可视化, 默认不启用。
#    注意，使用tensorboad之前需要使用命令 "tensorboard --logdir= log_path"来启动，结果通过网页 http://localhost:6006/'查看可视化结果,如果不成功,可以试试`python -m tensorboard.main --logdir=/work/home/aojiang/06项目复现/08Kansformer/code/DRresults/04results/tensorboard/vgg_big/`,或`python -m tensorboard.main --logdir=/work/home/aojiang/06项目复现/08Kansformer/code/DRresults/04results/tensorboard/vision_transformer1/`
############################################################################################################
# --model 可选的超参如下：
# alexnet   zfnet   vgg   vgg_tiny   vgg_small   vgg_big   googlenet   xception   resnet_small   resnet   resnet_big   resnext   resnext_big  
# densenet_tiny   densenet_small   densenet   densenet_big   mobilenet_v3   mobilenet_v3_large   shufflenet_small   shufflenet
# efficient_v2_small   efficient_v2   efficient_v2_large   convnext_tiny   convnext_small   convnext   convnext_big   convnext_huge
# vision_transformer_small   vision_transformer   vision_transformer_big   swin_transformer_tiny   swin_transformer_small   swin_transformer 

# 训练命令示例： # python train.py --model alexnet --num_classes 5
############################################################################################################
import os 
import argparse 
import math
import shutil
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler 
import classic_models 
from utils.lr_methods import warmup 
from dataload.dataload_five_flower import Five_Flowers_Load
from utils.train_engin import train_one_epoch, evaluate 

parser = argparse.ArgumentParser()
#parser.add_argument('--num_classes', type=int, default=5, help='the number of classes')
parser.add_argument('--num_classes', type=int, default=2, help='the number of classes')#对于04Diagnosis_of_Diabetic_Retinopathy这个数据集，input为2
parser.add_argument('--epochs', type=int, default=50, help='the number of training epoch')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size for training')
parser.add_argument('--lr', type=float, default=0.0002, help='star learning rate')
parser.add_argument('--lrf', type=float, default=0.0001, help='end learning rate') 
parser.add_argument('--seed', default=21, action='store_true', help='fix the initialization of parameters')
#parser.add_argument('--tensorboard', default=False, action='store_true', help=' use tensorboard for visualization') 
parser.add_argument('--tensorboard', default=True, action='store_true', help=' use tensorboard for visualization') 
parser.add_argument('--use_amp', default=False, action='store_true', help=' training with mixed precision') 
#parser.add_argument('--data_path', type=str, default=r"data/flower")
parser.add_argument('--data_path', type=str, default=r"data/04Diagnosis_of_Diabetic_Retinopathy")
parser.add_argument('--model', type=str, default="vision_transformer1", help=' select a model for training') #这行代码调用的是`vision_transformer1`,即`kit_base_patch16_224`，具体的其他模型可查看`/work/home/aojiang/06项目复现/08Kansformer/code/classic_models/__init__.py`
#parser.add_argument('--model', type=str, default="resnext", help=' select a model for training') 
#下面这行参数无法同时调用2个以上的GPU，也可能是我的调用方法有问题,`python train_transformer_DR.py --device 0 1`
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

# 添加一个命令行参数 --gpuid，此参数指定使用的 GPU 设备ID，以列表形式给出，默认为 [0]，即使用第一个GPU。如果此列表为空，表示使用 CPU 进行训练。在终端进入`code`,并运行`python train_transformer_DR.py --gpuid 0 1`(如果你运行`python check_gpus.py`后显示有两个GPU可用）

#parser.add_argument("--gpuid", default=[0], nargs='+', type=int, help="GPU device IDs to use for training. Empty list implies CPU usage.")
# 添加参数 "--gpuid"，用于指定训练过程中使用的GPU设备ID。
# default=[0]：默认值为 [0]，即使用第0号GPU。
# nargs='+': 表示可以输入多个GPU ID，形式为列表。
# type=int: 将输入的ID解析为整数类型。
# help: 参数说明，空列表表示使用CPU进行训练。

parser.add_argument('--weights', type=str, default=r'model_pth/vit_base_patch16_224.pth', help='initial weights path')
#parser.add_argument('--weights', type=str, default=r'model_pth/vgg19-dcbb9e9d.pth', help='initial weights path')#使用vgg19的预训练权重#会报错
# 添加参数 "--weights"，用于指定初始化权重文件的路径。
# type=str：将输入的路径解析为字符串类型。
# default=r'model_pth/vit_base_patch16_224.pth'：默认值为指定的权重文件路径。
# help: 参数说明，指定初始化权重的文件路径。

#parser.add_argument('--weights', type=str, default=r'model_pth/vgg19-dcbb9e9d.pth', help='initial weights path')#使用vgg19的预训练权重#会报错


opt = parser.parse_args()  

if opt.seed:
    def seed_torch(seed=1):
        random.seed(seed) # Python random module.	
        os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed) # Numpy module.
        torch.manual_seed(seed)  # 为CPU设置随机种子
        torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        # 设置cuDNN：cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率。如果需要保证可重复性，可以使用如下设置:
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        # 实际上这个设置对精度影响不大，仅仅是小数点后几位的差别。所以如果不是对精度要求极高，其实不太建议修改，因为会使计算效率降低。
        print('random seed has been fixed')
    seed_torch() 

'''def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu')  # 强制使用 CPU
    print(args)
'''
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args)

    # 初始化模型
    model = classic_models.find_model_using_name(opt.model, num_classes=opt.num_classes)

    # 如果有多个GPU可用，则使用DataParallel进行并行处理
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
    
    model = model.to(device)

    # 其他代码省略...

    if opt.tensorboard:
        # 这是存放你要使用tensorboard显示的数据的绝对路径
        #log_path = os.path.join('./results/tensorboard' , args.model)
        #log_path = os.path.join('./DRresults/04results/tensorboard' , args.model)#保存DR的日志
        #log_path = os.path.join(os.getcwd(), 'DRresults/04results/tensorboard' , args.model)#保存DR的日志
        log_path = os.path.join(os.getcwd(), 'DRresults/04results/tensorboard' , args.model)#保存DR的日志
        print('Start Tensorboard with "tensorboard --logdir={}"'.format(log_path)) 

        if os.path.exists(log_path) is False:
            os.makedirs(log_path)
            print("tensorboard log save in {}".format(log_path))
        else:
            shutil.rmtree(log_path) #当log文件存在时删除文件夹。记得在代码最开始import shutil 

        # 实例化一个tensorboard
        tb_writer = SummaryWriter(log_path)
    '''
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])} 
    '''
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                    # 随机裁剪图像为224x224大小，并调整其分辨率。
                                    transforms.RandomHorizontalFlip(),
                                    # 随机水平翻转图像，以增加数据的多样性。
                                    transforms.ToTensor(),
                                    # 将图像转换为张量（Tensor），并将其像素值从[0, 255]缩放到[0, 1]。
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                                    # 使用均值[0.485, 0.456, 0.406]和标准差[0.229, 0.224, 0.225]对图像进行标准化，
                                    # 使其符合预训练模型的输入规范。    
        "val": transforms.Compose([transforms.Resize(256),
                                    # 将图像的短边调整到256像素，同时保持长宽比不变。
                                    transforms.CenterCrop(224),
                                    # 从图像中心裁剪出224x224大小的区域。
                                    transforms.ToTensor(),
                                    # 将图像转换为张量（Tensor），并将其像素值从[0, 255]缩放到[0, 1]。
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
                                    # 使用均值[0.485, 0.456, 0.406]和标准差[0.229, 0.224, 0.225]对图像进行标准化，
                                    # 使其符合预训练模型的输入规范。
    


 
    # 对标pytorch封装好的ImageFlolder，我们自己实现了一个数据加载类 Five_Flowers_Load，并使用指定的预处理操作来处理图像，结果会同时返回图像和对应的标签。  
    train_dataset = Five_Flowers_Load(os.path.join(args.data_path , 'train'), transform=data_transform["train"])
    val_dataset = Five_Flowers_Load(os.path.join(args.data_path , 'val'), transform=data_transform["val"]) 
 
    if args.num_classes != train_dataset.num_class:
        raise ValueError("dataset have {} classes, but input {}".format(train_dataset.num_class, args.num_classes))
 
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # 使用 DataLoader 将加载的数据集处理成批量（batch）加载模式
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0, collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,  num_workers=0, collate_fn=val_dataset.collate_fn)
 
    # create model
    model = classic_models.find_model_using_name(opt.model, num_classes=opt.num_classes).to(device) 
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        # del_keys = ['head.weight', 'head.bias'] if model.has_logits \
        #     else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        del_keys = ['head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
    pg = [p for p in model.parameters() if p.requires_grad] 
    optimizer = optim.Adam(pg, lr=args.lr)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc = 0.
    
    # save parameters path
    #save_path = os.path.join(os.getcwd(), 'results/weights', args.model)
    save_path = os.path.join(os.getcwd(), 'DRresults/04results/weights', args.model)#保存DR的模型训练参数
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for epoch in range(args.epochs):
        # train
        mean_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch, use_amp=args.use_amp, lr_method= warmup)
        scheduler.step()
        # validate
        val_acc = evaluate(model=model, data_loader=val_loader, device=device)

 
        print('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' %  (epoch + 1, mean_loss, train_acc, val_acc))   
        with open(os.path.join(save_path, "vision_att_transformer_flower_log.txt"), 'a') as f: 
                f.writelines('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' %  (epoch + 1, mean_loss, train_acc, val_acc) + '\n')

        if opt.tensorboard:
            tags = ["train_loss", "train_acc", "val_accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_acc, epoch)
            tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

        # 判断当前验证集的准确率是否是最大的，如果是，则更新之前保存的权重
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, "vision_att_transformer_flower.pth")) 

        
main(opt)

请你记忆上述信息







### 五、
现在，我有一个新的数据集，名为DR_grading
它的路径为/work/home/aojiang/06项目复现/08Kansformer/code/data/DDR_dataset/DDR-dataset/DR_grading

DR_grading文件夹下包含三个文件夹，它们的名字分别为train、test、valid
这三个文件夹中都包含许多照片

另外我有三个txt文件，它们的名字分别是train.txt、test.txt、valid.txt，这三个txt文件是train、test、valid
这三个文件夹中照片的标签

举例来说，train.txt中前几行的内容如下：
007-0004-000.jpg 0
007-0007-000.jpg 0
007-0008-000.jpg 0
007-0009-000.jpg 0
007-0011-000.jpg 0
007-0013-000.jpg 0
007-0022-000.jpg 0
007-0023-000.jpg 0
007-0024-000.jpg 0
007-0025-000.jpg 0
007-0026-000.jpg 0
007-0028-000.jpg 0
007-0029-000.jpg 0
007-0030-000.jpg 0
007-0031-000.jpg 0
007-0032-000.jpg 0
007-0034-000.jpg 0
007-0035-000.jpg 0
007-0036-000.jpg 0
007-0037-000.jpg 0
007-0038-000.jpg 0

现在我需要你帮我写一份python代码，这份python代码的名字为dataload_DR_grading.py
请你参考dataload_five_flower.py和dataload_five_flower.py的使用方法
帮我写一份加载DR_grading数据集的dataload_DR_grading.py


## 版本一

```python
from PIL import Image
import torch
from torch.utils.data import Dataset
import os

class DR_Grading_Dataset(Dataset):
    def __init__(self, data_path: str, labels_file: str, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.images_path = []
        self.images_label = []

        # 读取标签文件
        label_dict = {}
        with open(labels_file, 'r') as f:
            for line in f:
                image_name, label = line.strip().split()
                label = int(label)
                label_dict[image_name] = label
                self.images_path.append(os.path.join(data_path, image_name))
                self.images_label.append(label)

        # 计算类别总数
        self.num_class = len(set(self.images_label))

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        label = self.images_label[idx]

        # 加载图片
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 应用转换
        if self.transform:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels)
        return images, labels

```

## 版本二

```python
from PIL import Image
import torch
from torch.utils.data import Dataset
import os

class DR_Grading_Dataset(Dataset):
    def __init__(self, data_path: str, labels_file: str, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.images_path = []
        self.images_label = []

        # 读取标签文件
        label_dict = {}
        with open(labels_file, 'r') as f:
            for line in f:
                image_name, label = line.strip().split()
                label = int(label)
                label_dict[image_name] = label
                self.images_path.append(os.path.join(data_path, image_name))
                self.images_label.append(label)

        # 计算类别总数
        self.num_class = len(set(self.images_label))

        # 计算并打印图像总数
        print("{} images were found in the dataset.".format(len(self.images_path)))

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        label = self.images_label[idx]

        # 加载图片
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 应用转换
        if self.transform:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels)
        return images, labels

```





resnet50 84.49
resnext50  80.1
densenet169 77.63
transformer1  75.2




