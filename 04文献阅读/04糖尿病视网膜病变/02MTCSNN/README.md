# MTCSNN: Multi-task Clinical Siamese Neural Network for Diabetic Retinopathy Severity Prediction

> 来源：paperswithcod.com
> 链接：[MTCSNN: Multi-task Clinical Siamese Neural Network for Diabetic Retinopathy Severity Prediction](https://paperswithcode.com/paper/mtcsnn-multi-task-clinical-siamese-neural)


文献获取链接为：[arxiv](https://arxiv.org/pdf/2208.06917v1)

文献开源代码链接为：[github](https://github.com/cfeng16/MTCSNN_for_Diabetic_Retinopathy_Severity_Prediction)


## 数据集
在这份开源项目的README文档中，我注意到，它使用的数据集为开源数据集MedMNIST
开源数据集的链接为：[github](https://github.com/MedMNIST/MedMNIST)
在这个开源数据集的相关文献于2023年发表在了Nature Scientific 上，原文链接为：[nature](https://www.nature.com/articles/s41597-022-01721-8)

这个开源数据集由上海交通大学、复旦大学、中山医院、哈佛大学、剑桥大学创建，包含18个公开数据集，其中12个为二维数据集，6个为三维数据集，包含我们感兴趣的糖尿病视网膜眼底图像数据集。

此外，该开源数据集的作者，还对18个数据集分别使用了七种不同的方法（以残差卷积神经网络为主）进行试验，评价指标为AUC。

## 前期知识
Siamese Neural Network的概念最早是在1991年由Yann LeCun等人提出的，最初是作为用于手写数字识别的一部分，发表在论文《Convolutional Neural Networks for Handwriting Recognition》中。这项工作是LeCun在贝尔实验室进行的，并且是深度学习和卷积神经网络（CNN）早期研究的重要里程碑之一。

后来，Siamese Network的结构在2005年被重新提出并用于计算机视觉任务，特别是用于图像检索和相似性度量。Yann LeCun的学生Chopra等人在2005年的论文[《Learning a Similarity Metric Discriminatively, with Application to Face Verification》](https://link.zhihu.com/?target=http%3A//yann.lecun.com/exdb/publis/pdf/chopra-05.pdf)中进一步发展了这一概念，这篇论文通常与孪生网络的现代应用联系在一起。

> 知乎解读：
> [Siamese Network](https://zhuanlan.zhihu.com/p/639490209)
> [Siamese network 孪生神经网络--一个简单神奇的结构](https://zhuanlan.zhihu.com/p/35040994)

因此，可以说Siamese Neural Network作为深度学习中的一个重要概念，在1991年首次被提出，而在2005年得到了进一步的发展和应用。

### 基于Siamese network的源码实现：

[SiaStegNet](https://github.com/SiaStg/SiaStegNet/tree/master)

[SiamVGG](https://github.com/leeyeehoo/SiamVGG/tree/master)


可以参考的项目：[ResNet50-SNN](https://github.com/profjefer/Facial-Recognition-ResNet50-SNN/tree/main?tab=readme-ov-file)


### ResNet34模型的源代码

ResNet34是一个深度残差网络，它由34层组成，并且使用了残差学习框架。下面是一个使用PyTorch框架的ResNet34的完整实现代码：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet34(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet34, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # Global average pooling
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Example usage:
# model = ResNet34(num_classes=10)
# print(model)
```

这段代码定义了`BasicBlock`类，它是构成ResNet34的基本单元，以及`ResNet34`类，它构建了整个网络。`BasicBlock`类中包含了残差学习的核心，即两个卷积层加上一个恒等映射（如果需要的话）。`ResNet34`类则定义了网络的四个残差层，每层由多个`BasicBlock`组成，并且最后通过一个全连接层输出到指定的类别数。

请注意，这个实现使用了全局平均池化（`F.avg_pool2d`）来替代传统的平均池化层，这是ResNet的一个常见变体。此外，`num_classes`参数允许你指定输出类别的数量，这在处理不同数据集时非常有用。


#### ResNet34的层数是如何构成的
让我们来详细分析ResNet34的层数是如何构成的。

ResNet34由以下主要部分组成：

1. **初始卷积层**：一个卷积层，输出通道数为64。
2. **残差层**：由多个残差块（BasicBlock）组成，每个残差块包含两个卷积层。
3. **全连接层**：最后的分类层。

在ResNet34中，残差层由以下残差块组成：

- 第一层（layer1）：64个通道，有3个残差块。
- 第二层（layer2）：128个通道，有4个残差块，并且这一层的输入输出之间有一个步长为2的卷积层，用于下采样。
- 第三层（layer3）：256个通道，有6个残差块，同样在这一层的输入输出之间有一个步长为2的卷积层。
- 第四层（layer4）：512个通道，有3个残差块，也有一个步长为2的卷积层。

每个残差块包含两个卷积层，所以如果我们将每个残差块视为两个层，那么层数可以这样计算：

- 第一层：3个残差块 * 2个卷积层/块 = 6层
- 第二层：4个残差块 * 2个卷积层/块 = 8层
- 第三层：6个残差块 * 2个卷积层/块 = 12层
- 第四层：3个残差块 * 2个卷积层/块 = 6层

加上初始卷积层和全连接层，总层数为：

\[ 1 (\text{initial conv}) + 6 (\text{layer1}) + 8 (\text{layer2}) + 12 (\text{layer3}) + 6 (\text{layer4}) + 1 (\text{final FC}) = 34 \]

所以，ResNet34总共有34层，包括初始卷积层、残差块中的卷积层和最后的全连接层。

#### 将上述ResNet-34模型转变为孪生网络（Siamese Network）

要将上述ResNet-34模型转变为孪生网络（Siamese Network），我们需要进行一些重要的修改。孪生网络由两个相同的子网络组成，它们共享权重，并且用于比较两个输入样本是否相似。以下是如何修改上述代码以创建一个孪生网络的示例：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseResNet34(nn.Module):
    def __init__(self, block, layers, num_classes, grayscale):
        super(SiameseResNet34, self).__init__()
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3

        # Define the first convolution layer and the first batch norm layer
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Create the residual layers using the same block
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Use adaptive pooling to make the output size (1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Define the fully connected layer for classification
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # The _make_layer method remains the same as in ResNet-34
    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        # Forward pass for one input through the network
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, input1, input2):
        # Forward pass for both inputs through the shared network
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # Compute the difference between the two outputs
        difference = output1 - output2

        # Calculate the Euclidean distance or another similarity metric
        #  similarity = F.pairwise_distance(output1, output2)
        similarity = F.pairwise_distance(difference, torch.zeros_like(difference))


        # Get the final classification output
        logits = self.fc(similarity)

        return logits, similarity

# Define the Siamese ResNet-34 model
def siamese_resnet34(num_classes, grayscale=False):
    model = SiameseResNet34(BasicBlock, [3, 4, 6, 3], num_classes, grayscale)
    return model
```

在这个修改后的代码中，`SiameseResNet34` 类包含了孪生网络的逻辑。`forward_once` 方法用于通过共享网络传递单个输入，并返回输出特征。`forward` 方法接受两个输入（`input1` 和 `input2`），通过共享网络传递它们，并计算它们的相似度。

请注意，这个示例中的相似度是通过简单的欧氏距离来计算的，这可能需要根据你的具体任务进行调整。例如，你可能需要使用余弦相似度或其他相似度度量方法。此外，`self.fc` 层可能需要根据相似度度量和具体任务进行调整。















