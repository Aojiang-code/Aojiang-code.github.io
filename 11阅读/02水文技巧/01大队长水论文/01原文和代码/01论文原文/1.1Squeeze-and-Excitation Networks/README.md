# Squeeze-and-Excitation Networks

## 目录

- [Squeeze-and-Excitation Networks](#squeeze-and-excitation-networks)
  - [目录](#目录)
      - [摘要](#摘要)
      - [1. 引言](#1-引言)
      - [2.相关工作](#2相关工作)
        - [深层架构](#深层架构)
        - [注意力和门控机制。](#注意力和门控机制)
      - [3. 挤压激励块](#3-挤压激励块)
        - [3.1. Squeeze：全局信息嵌入](#31-squeeze全局信息嵌入)
        - [3.2.激励：自适应重新校准](#32激励自适应重新校准)
      - [4. 模型和计算复杂性](#4-模型和计算复杂性)
      - [5. 实施](#5-实施)
      - [6. 实验](#6-实验)
        - [6.1. ImageNet 分类](#61-imagenet-分类)
        - [6.2.场景分类](#62场景分类)


#### 摘要
Convolutional neural networks are built upon the convolution operation, which extracts informative features by fusing spatial and channel-wise information together within local receptive fields.

----
卷积神经网络是在卷积运算的基础上建立起来的，它通过在局部接收域内融合空间信息和通道信息来提取信息特征。

In order to boost the representational power of a network, several recent approaches have shown the benefit of enhancing spatial encoding.

----
为了提高网络的表示能力，最近的几种方法已经显示了增强空间编码的好处。

In this work, we focus on the channel relationship and propose a novel architectural unit, which we term the “Squeezeand-Excitation” (SE) block, that adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.

----
在这项工作中，我们聚焦于通道关系，并提出了一个新的体系结构单元，我们称之为“挤压和激励”(SE)块，通过明确建模通道之间的相互依赖性，自适应地重新校准通道方向的特征响应。

We demonstrate that by stacking these blocks together, we can construct SENet architectures that generalise extremely well across challenging datasets.

----
我们证明，通过堆叠这些块在一起，我们可以构建 SENet 体系结构，通过具有挑战性的数据集非常好地一般化。

Crucially, we find that SE blocks produce significant performance improvements for existing state-of-the-art deep architectures at minimal additional computational cost.

----
至关重要的是，我们发现 SE 块以最小的额外计算成本为现有的最先进的深度体系结构提供了显著的性能改进。

SENets formed the foundation of our ILSVRC 2017 classification submission which won first place and significantly reduced the top-5 error to 2.251%, achieving a ∼25% relative improvement over the winning entry of 2016.

----
SENets 是我们 ILSVRC 2017年分类提交的基础，获得了第一名，并将前5名的错误显着降低到2.251% ，比2016年的获奖作品相对提高了25% 。

Code and models are available at https: //github.com/hujie-frank/SENet.

----
代码和模型可在 https:// github.com/hujie-frank/senet 下载。

#### 1. 引言

Convolutional neural networks (CNNs) have proven to be effective models for tackling a variety of visual tasks [21, 27, 33, 45].

----
卷积神经网络(CNN)已被证明是处理各种视觉任务的有效模型[21,27,33,45]。

[21] A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet classification with deep convolutional neural networks. In NIPS, 2012. 1, 3

----
[21] A · 克里切夫斯基、 I · 苏茨克弗和 G · E · 辛顿。基于深度卷积神经网络的 ImageNet 分类。在 NIPS，2012年。一，三

[27] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015. 1

----
[27] J. Long，E. Shelhamer，and T. Darrell。语义分割的完全卷积网络。 CVPR，2015.1

[33] S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In NIPS, 2015. 1, 7

----
[33] S · 任、 K · 何、 R · 格希克和 J · 孙。更快的 R-CNN: 实现区域提案网络的实时目标检测。在 NIPS，2015年。一，七

[45] A. Toshev and C. Szegedy. DeepPose: Human pose estimation via deep neural networks. In CVPR, 2014. 1

----
[45] A. Toshev and C. Szegedy. DeepPose: 利用深度神经网络进行人体姿态估计。 CVPR，2014.1

For each convolutional layer, a set of filters are learned to express local spatial connectivity patterns along input channels

----
对于每个卷积层，学习一组过滤器来表示沿输入通道的局部空间连接模式


In other words, convolutional filters are expected to be informative combinations by fusing spatial and channel-wise information together within local receptive fields.

----
换句话说，卷积滤波器通过将空间信息和通道信息融合在一起，可望成为信息组合。

By stacking a series of convolutional layers interleaved with non-linearities and downsampling, CNNs are capable of capturing hierarchical patterns with global receptive fields as powerful image descriptions.

----
通过叠加一系列与非线性和下采样交错的卷积层，神经网络能够捕获具有全局接收场的层次模式作为强大的图像描述。


By stacking a series of convolutional layers interleaved with non-linearities and downsampling, CNNs are capable of capturing hierarchical patterns with global receptive fields as powerful image descriptions.

----
通过叠加一系列与非线性和下采样交错的卷积层，神经网络能够捕获具有全局接收场的层次模式作为强大的图像描述。


Recent work has demonstrated that the performance of networks can be improved by explicitly embedding learning mechanisms that help capture spatial correlations without requiring additional supervision.

----
最近的工作表明，通过明确地嵌入学习机制，帮助捕捉空间相关性，而不需要额外的监督，网络的性能可以得到改善。

One such approach was popularised by the Inception architectures [16, 43], which showed that the network can achieve competitive accuracy by embedding multi-scale processes in its modules.

----
一种这样的方法被先启架构推广[16,43] ，这表明网络可以通过在其模块中嵌入多尺度的过程来实现竞争的准确性。


[16] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, 2015. 1, 2, 5, 6

----
[16] S. Ioffe 和 C. Szegedy。批量标准化: 通过减少内部协变量转移来加速深度网络训练。在 ICML，2015年。一，二，五，六


[43] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015. 1, 2, 4

----
[43] C. 塞吉迪，W. 刘，Y. 贾，P. 塞曼内，S. 里德，D. 安圭洛夫，D. 尔汉，V. 万豪克和 A. 拉比诺维奇。更深层次的回旋。在 CVPR，2015年。一，二，四

More recent work has sought to better model spatial dependence [1, 31] and incorporate spatial attention [19].

----
最近的研究试图更好地模拟空间依赖[1,31] ，并将空间注意力纳入其中[19]。

[1] S. Bell, C. L. Zitnick, K. Bala, and R. Girshick. Insideoutside net: Detecting objects in context with skip pooling and recurrent neural networks. In CVPR, 2016. 1

----
[1]贝尔、齐特尼克、巴拉和吉希克。内部外部网络: 利用跳跃池和递归神经网络检测上下文中的对象。在 CVPR，2016年。1


[31] A. Newell, K. Yang, and J. Deng. Stacked hourglass networks for human pose estimation. In ECCV, 2016. 1, 2

----
[31] A. Newell，K. Yang，and J. Deng。用于人体姿态估计的叠加沙漏网络。 ECCV，2016.1,2


[19] M. Jaderberg, K. Simonyan, A. Zisserman, and K. Kavukcuoglu. Spatial transformer networks. In NIPS, 2015. 1, 2

----
[19] M.Jaderberg，K. Simonyan，A. Zisserman，and K. Kavukcuoglu。空间变压器网络。收录于 NIPS，2015.1,2


In this paper, we investigate a different aspect of architectural design - the channel relationship, by introducing a new architectural unit, which we term the “Squeeze-andExcitation” (SE) block.

----
本文通过引入一个新的建筑单元——“挤压-激励”(SE)块，探讨了建筑设计的另一个方面——渠道关系。


Our goal is to improve the representational power of a network by explicitly modelling the interdependencies between the channels of its convolutional features.

----
我们的目标是通过明确地模拟网络卷积特征通道之间的相互依赖关系来提高网络的表征能力。


To achieve this, we propose a mechanism that allows the network to perform feature recalibration, through which it can learn to use global information to selectively emphasise informative features and suppress less useful ones.

----
为了实现这一目标，我们提出了一种允许网络执行特征重校准的机制，通过这种机制，网络可以学习使用全局信息来有选择地强调信息特征，并抑制不太有用的特征。

The basic structure of the SE building block is illustrated in Fig. 1.

----
SE 构件的基本结构如图1所示。


![alt text](image-2.png)


For any given transformation Ftr : X → U, X ∈ RH′×W ′×C′ , U ∈ RH×W ×C , (e.g. a convolution or a set of convolutions), we can construct a corresponding SE block to perform feature recalibration as follows.

----
对于任意给定的变换 Ftr: X → U，X ∈ RH ′ × W ′ × C ′ ，U ∈ RH × W × C，(例如一个卷积或一组卷积) ，我们可以构造一个相应的 SE 块来执行如下特征校正。


The features U are first passed through a squeeze operation, which aggregates the feature maps across spatial dimensions H × W to produce a channel descriptor.

----
特征U首先经过挤压操作，该操作聚合跨空间维度 H x W 的特征地图以产生通道描述符。

This descriptor embeds the global distribution of channel-wise feature responses, enabling information from the global receptive field of the network to be leveraged by its lower layers.

----
该描述符嵌入了通道特征响应的全局分布，使得来自网络的全局感受野的信息能够被其较低层利用。

This is followed by an excitation operation, in which sample-specific activations, learned for each channel by a self-gating mechanism based on channel dependence, govern the excitation of each channel.

----
接下来是激励操作，其中通过基于通道依赖性的自门机制为每个通道学习特定于样本的激活，控制每个通道的激励。


The feature maps U are then reweighted to generate the output of the SE block which can then be fed directly into subsequent layers.

----
然后对特征图 U 进行重新加权以生成 SE 块的输出，然后可以将其直接馈送到后续层中。



An SE network can be generated by simply stacking a collection of SE building blocks.

----
通过简单地堆叠一组 SE 构建块就可以生成 SE 网络。

SE blocks can also be used as a drop-in replacement for the original block at any depth in the architecture.

----
SE 块还可以用作架构中任何深度的原始块的直接替代品。

However, while the template for the building block is generic, as we show in Sec. 6.4, the role it performs at different depths adapts to the needs of the network.

----
然而，虽然构建块的模板是通用的，正如我们在第 6.4节中所示，它在不同深度所扮演的角色适应网络的需要。


In the early layers, it learns to excite informative features in a class agnostic manner, bolstering the quality of the shared lower level representations.

----
在早期层中，它学习以与类无关的方式激发信息特征，从而提高共享的较低级别表示的质量。


In later layers, the SE block becomes increasingly specialised, and responds to different inputs in a highly class-specific manner.

----
在后面的层中，SE 块变得越来越专业，并以高度特定于类的方式响应不同的输入。

Consequently, the benefits of feature recalibration conducted by SE blocks can be accumulated through the entire network.

----
因此，SE 块进行的特征重新校准的好处可以通过整个网络累积。

The development of new CNN architectures is a challenging engineering task, typically involving the selection of many new hyperparameters and layer configurations.

----
新 CNN 架构的开发是一项具有挑战性的工程任务，通常涉及选择许多新的超参数和层配置。


By contrast, the design of the SE block outlined above is simple, and can be used directly with existing state-of-the-art architectures whose modules can be strengthened by direct replacement with their SE counterparts.

----
相比之下，上面概述的 SE 块的设计很简单，并且可以直接与现有的最先进的架构一起使用，其模块可以通过直接替换为 SE 对应项来增强。


Moreover, as shown in Sec. 4, SE blocks are computationally lightweight and impose only a slight increase in model complexity and computational burden.

----
此外，如第 4 节所示，SE块计算量轻，仅略微增加模型复杂性和计算负担。


To support these claims, we develop several SENets and provide an extensive evaluation on the ImageNet 2012 dataset [34].

----
为了支持这些主张，我们开发了多个 SENet，并对 ImageNet 2012 数据集进行了广泛的评估 [34]。


[34] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei. ImageNet large scale visual recognition challenge. IJCV, 2015. 2

----
[34] O. Russakovsky、J. Deng、H. Su、J. Krause、S. Satheesh、S. Ma、Z. Huang、A. Karpathy、A. Khosla、M. Bernstein、A. C. Berg 和 L. Fei -费。 ImageNet 大规模视觉识别挑战。国际JCV，2015。2

To demonstrate their general applicability, we also present results beyond ImageNet, indicating that the proposed approach is not restricted to a specific dataset or a task.

----
为了证明它们的普遍适用性，我们还展示了 ImageNet 之外的结果，表明所提出的方法并不局限于特定的数据集或任务。

Using SENets, we won the first place in the ILSVRC 2017 classification competition.

----
使用 SENets，我们在 ILSVRC 2017 分类竞赛中获得了第一名。

Our top performing model ensemble achieves a 2.251% top-5 error on the test set1.

----
我们表现​​最佳的模型集成在测试集 1 上实现了 2.251% 的 top-5 错误率。

This represents a ∼25% relative improvement in comparison to the winner entry of the previous year (with a top-5 error of 2.991%).

----
与上一年的获奖者相比，这意味着相对改善了约 25%（前 5 名错误率为 2.991%）。

#### 2.相关工作

##### 深层架构
VGGNets [39] and Inception models [43] demonstrated the benefits of increasing depth.

----
VGGNets [39] 和 Inception 模型 [43] 证明了增加深度的好处。

[39] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.

----
[39] K.西蒙扬和A.齐瑟曼。用于大规模图像识别的非常深的卷积网络。 ICLR，2015 年。

[43] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.

----
[43] C. Szegedy、W. Liu、Y. Jia、P. Sermanet、S. Reed、D. Anguelov、D. Erhan、V. Vanhoucke 和 A. Rabinovich。更深入地了解卷积。在 CVPR，2015 年。

Batch normalization (BN) [16] improved gradient propagation by inserting units to regulate layer inputs, stabilising the learning process.

----
批量归一化（BN）[16]通过插入单元来调节层输入来改进梯度传播，从而稳定学习过程。

[16] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, 2015.

----
[16] S. Ioffe 和 C. Szegedy。批量归一化：通过减少内部协变量偏移来加速深度网络训练。在 ICML，2015 年。

ResNets [10, 11] showed the effectiveness of learning deeper networks through the use of identity-based skip connections.

----
ResNets [10, 11] 展示了通过使用基于身份的跳跃连接来学习更深层次网络的有效性。


[10] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.

----
[10] 何坤，张旭，任思，孙静．用于图像识别的深度残差学习。在 CVPR，2016 年。


[11] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.

----
[11] 何坤，张旭，任思，孙静．深度残差网络中的身份映射。在 ECCV，2016 年。


Highway networks [40] employed a gating mechanism to regulate shortcut connections.

----
高速公路网络[40]采用门控机制来调节捷径连接。

[40] R. K. Srivastava, K. Greff, and J. Schmidhuber. Training very deep networks. In NIPS, 2015.

----
[40] R.K. Srivastava、K. Greff 和 J. Schmidhuber。训练非常深的网络。在 NIPS，2015 年。

Reformulations of the connections between network layers [5, 14] have been shown to further improve the learning and representational properties of deep networks.

----
网络层 [5, 14] 之间的连接的重新表述已被证明可以进一步提高深度网络的学习和表示属性。

[5] Y. Chen, J. Li, H. Xiao, X. Jin, S. Yan, and J. Feng. Dual path networks. In NIPS, 2017.

----
[5] Y. Chen，J. Li，H. Xiao，X. Jin，S. Yan，J. Feng。双路径网络。在 NIPS，2017 年。

[14] G. Huang, Z. Liu, K. Q. Weinberger, and L. Maaten. Densely connected convolutional networks. In CVPR, 2017.

----
[14] G. Huang、Z. Liu、K. Q. Weinberger 和 L. Maaten。密集连接的卷积网络。在 CVPR，2017 年。

An alternative line of research has explored ways to tune the functional form of the modular components of a network.

----
另一条研究路线探索了调整网络模块化组件功能形式的方法。

Grouped convolutions can be used to increase cardinality (the size of the set of transformations) [15, 47].

----
分组卷积可用于增加基数（变换集的大小）[15, 47]。

[15] Y. Ioannou, D. Robertson, R. Cipolla, and A. Criminisi. Deep roots: Improving CNN efficiency with hierarchical filter groups. In CVPR, 2017.

----
[15] Y. Ioannou、D. Robertson、R. Cipolla 和 A. Criminisi。深层根源：通过分层过滤器组提高 CNN 效率。在 CVPR，2017 年。

[47] S. Xie, R. Girshick, P. Dollar, Z. Tu, and K. He. Aggregated residual transformations for deep neural networks. In CVPR, 2017.

----
[47] S. Xie，R. Girshick，P. Dollar，Z. Tu，K. He。深度神经网络的聚合残差变换。在 CVPR，2017 年。

Multi-branch convolutions can be interpreted as a generalisation of this concept, enabling more flexible compositions of operators [16, 42, 43, 44].

----
多分支卷积可以解释为这个概念的推广，使得算子的组合更加灵活[16,42,43,44]。

[16] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, 2015.

----
[16] S. Ioffe 和 C. Szegedy。批量归一化：通过减少内部协变量偏移来加速深度网络训练。在 ICML，2015 年。

[42] C. Szegedy, S. Ioffe, V. Vanhoucke, and A. Alemi. Inceptionv4, inception-resnet and the impact of residual connections on learning. In ICLR Workshop, 2016.

----
[42] C. Szegedy、S. Ioffe、V. Vanhoucke 和 A. Alemi。 Inceptionv4、inception-resnet 以及残差连接对学习的影响。 ICLR 研讨会，2016 年。

[43] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.

----
[43] C. Szegedy、W. Liu、Y. Jia、P. Sermanet、S. Reed、D. Anguelov、D. Erhan、V. Vanhoucke 和 A. Rabinovich。更深入地了解卷积。在 CVPR，2015 年。


[44] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. In CVPR, 2016.

----
[44] C. Szegedy、V. Vanhoucke、S. Ioffe、J. Shlens 和 Z. Wojna。重新思考计算机视觉的初始架构。在 CVPR，2016 年。

Recently, compositions which have been learned in an automated manner [26, 54, 55] have shown competitive performance.

----
最近，以自动化方式学习的作品[26,54,55]已经表现出有竞争力的表现。

[26] H. Liu, K. Simonyan, O. Vinyals, C. Fernando, and K. Kavukcuoglu. Hierarchical representations for efficient architecture search. arXiv: 1711.00436, 2017.

----
[26] H. Liu、K. Simonyan、O. Vinyals、C. Fernando 和 K. Kavukcuoglu。用于高效架构搜索的分层表示。 arXiv：1711.00436，2017。


[54] B. Zoph and Q. V. Le. Neural architecture search with reinforcement learning. In ICLR, 2017.

----
[54] B. Zoph 和 Q. V. Le。使用强化学习的神经架构搜索。 ICLR，2017 年。

[55] B. Zoph, V. Vasudevan, J. Shlens, and Q. V. Le. Learning transferable architectures for scalable image recognition. arXiv: 1707.07012, 2017.

----
[55] B. Zoph、V. Vasudevan、J. Shlens 和 Q. V. Le。学习可扩展图像识别的可转移架构。 arXiv：1707.07012，2017。

Cross-channel correlations are typically mapped as new combinations of features, either independently of spatial structure [6, 20] or jointly by using standard convolutional filters [24] with 1×1 convolutions.

----
跨通道相关性通常被映射为新的特征组合，要么独立于空间结构 [6, 20]，要么通过使用具有 1×1 卷积的标准卷积滤波器 [24] 联合映射。

[6] F. Chollet. Xception: Deep learning with depthwise separable convolutions. In CVPR, 2017.

----
[6] F.乔莱。 Xception：具有深度可分离卷积的深度学习。在 CVPR，2017 年。


[20] M. Jaderberg, A. Vedaldi, and A. Zisserman. Speeding up convolutional neural networks with low rank expansions. In BMVC, 2014.

----
[20] M. Jaderberg、A. Vedaldi 和 A. Zisserman。通过低秩扩展加速卷积神经网络。在 BMVC，2014 年。

[24] M. Lin, Q. Chen, and S. Yan. Network in network. arXiv:1312.4400, 2013.

----
[24] 林明，陈强，严胜。网络中的网络。 arXiv：1312.4400，2013。

Much of this work has concentrated on the objective of reducing model and computational complexity, reflecting an assumption that channel relationships can be formulated as a composition of instance-agnostic functions with local receptive fields.

----
这项工作的大部分集中在降低模型和计算复杂性的目标上，反映了一种假设，即通道关系可以表示为具有局部感受野的实例不可知函数的组合。

In contrast, we claim that providing the unit with a mechanism to explicitly model dynamic, non-linear dependencies between channels using global information can ease the learning process, and significantly enhance the representational power of the network.

----
相比之下，我们声称为该单元提供一种使用全局信息显式建模通道之间动态、非线性依赖关系的机制可以简化学习过程，并显着增强网络的表示能力。

##### 注意力和门控机制。

Attention can be viewed, broadly, as a tool to bias the allocation of available processing resources towards the most informative components of an input signal [17, 18, 22, 29, 32].

----
从广义上讲，注意力可以被视为一种工具，将可用处理资源的分配偏向输入信号中信息最丰富的部分[17,18,22,29,32]。

[17] L. Itti and C. Koch. Computational modelling of visual attention. Nature reviews neuroscience, 2001.

----
[17]L.伊蒂和C.科赫。视觉注意力的计算建模。自然评论神经科学，2001。

[18] L. Itti, C. Koch, and E. Niebur. A model of saliency-based visual attention for rapid scene analysis. IEEE TPAMI, 1998.

----
[18] L. Itti、C. Koch 和 E. Niebur。用于快速场景分析的基于显着性的视觉注意模型。 IEEE TPAMI，1998。


[22] H. Larochelle and G. E. Hinton. Learning to combine foveal glimpses with a third-order boltzmann machine. In NIPS,2010

----
[22] H. Larochelle 和 G. E. Hinton。学习将中央凹瞥见与三阶玻尔兹曼机结合起来。在 NIPS 中，2010

[29] V. Mnih, N. Heess, A. Graves, and K. Kavukcuoglu. Recurrent models of visual attention. In NIPS, 2014.

----
[29] V. Mnih、N. Heess、A. Graves 和 K. Kavukcuoglu。视觉注意力的循环模型。在 NIPS，2014 年。

[32] B. A. Olshausen, C. H. Anderson, and D. C. V. Essen. A neurobiological model of visual attention and invariant pattern recognition based on dynamic routing of information. Journal of Neuroscience, 1993.

----
[32] B. A. Olshausen、C. H. Anderson 和 D. C. V. Essen。基于信息动态路由的视觉注意和不变模式识别的神经生物学模型。神经科学杂志，1993。

The benefits of such a mechanism have been shown across a range of tasks, from localisation and understanding in images [3, 19] to sequence-based models [2, 28].

----
这种机制的好处已经在一系列任务中得到了体现，从图像的定位和理解 [3, 19] 到基于序列的模型 [2, 28]。

It is typically implemented in combination with a gating function (e.g. a softmax or sigmoid) and sequential techniques [12, 41].

----
它通常与门函数（例如 softmax 或 sigmoid）和顺序技术结合实现 [12, 41]。

Recent work has shown its applicability to tasks such as image captioning [4, 48] and lip reading [7].

----
最近的工作表明它适用于图像字幕 [4, 48] 和唇读 [7] 等任务。

In these applications, it is often used on top of one or more layers representing higher-level abstractions for adaptation between modalities.

----
在这些应用程序中，它通常用在代表更高级别抽象的一层或多层之上，以便在模态之间进行适应。

Wang et al. [46] introduce a powerful trunkand-mask attention mechanism using an hourglass module [31].

----
王等人。 [46]使用沙漏模块引入了一种强大的躯干和面罩注意力机制[31]。

This high capacity unit is inserted into deep residual networks between intermediate stages.

----
这个高容量单元被插入到中间阶段之间的深度残差网络中。

In contrast, our proposed SE block is a lightweight gating mechanism, specialised to model channel-wise relationships in a computationally efficient manner and designed to enhance the representational power of basic modules throughout the network.

----
相比之下，我们提出的 SE 块是一种轻量级门控机制，专门用于以计算有效的方式对通道关系进行建模，并旨在增强整个网络中基本模块的表示能力。

#### 3. 挤压激励块
The Squeeze-and-Excitation block is a computational unit which can be constructed for any given transformation Ftr : X → U, X ∈ RH′×W ′×C′ , U ∈ RH×W ×C .

----
挤压和激励块是一个计算单元，可以为任何给定的变换 Ftr 构建：X → U，X ε RH′×W ′×C′，U ε RH×W ×C 。

For simplicity, in the notation that follows we take Ftr to be a convolutional operator.

----
为简单起见，在下面的符号中，我们将 Ftr 视为卷积运算符。

Let V = [v1, v2, . . . , vC ] denote the learned set of filter kernels, where vc refers to the parameters of the c-th filter. We can then write the outputs of Ftr as U = [u1, u2, . . . , uC ], where

----
设 V = [v1, v2,... 。 。 , vC ] 表示学习到的滤波器内核集合，其中 vc 指的是第 c 个滤波器的参数。然后我们可以将 Ftr 的输出写为 U = [u1, u2,...。 。 。 , uC ], 其中

![公式1](<../../11阅读/02水文技巧/01大队长水论文/01原文和代码/01论文原文/1.1Squeeze-and-Excitation Networks/01图片/公式1.png>)

Here ∗ denotes convolution, vc = [v1 c , v2 c , . . . , vC′ c ] and X = [x1, x2, . . . , xC′ ] (to simplify the notation, bias terms are omitted), while vs c is a 2D spatial kernel, and therefore represents a single channel of vc which acts on the corresponding channel of X.

----
这里*表示卷积，vc = [v1 c , v2 c , ... 。 。 , vC′ c ] 且 X = [x1, x2, . 。 。 , xC′ ]（为了简化符号，省略了偏差项），而 vs c 是一个 2D 空间核，因此表示 vc 的单个通道，它作用于 X 的相应通道。

Since the output is produced by a summation through all channels, the channel dependencies are implicitly embedded in vc, but these dependencies are entangled with the spatial correlation captured by the filters.

----
由于输出是通过所有通道求和产生的，因此通道依赖性隐式嵌入到 vc 中，但这些依赖性与滤波器捕获的空间相关性纠缠在一起。

Our goal is to ensure that the network is able to increase its sensitivity to informative features so that they can be exploited by subsequent transformations, and to suppress less useful ones.

----
我们的目标是确保网络能够提高对信息特征的敏感性，以便后续转换可以利用它们，并抑制不太有用的特征。

We propose to achieve this by explicitly modelling channel interdependencies to recalibrate filter responses in two steps, squeeze and excitation, before they are fed into next transformation.

----
我们建议通过显式建模通道相互依赖性来实现这一目标，以在将滤波器响应送入下一个转换之前分两个步骤（挤压和激励）重新校准滤波器响应。

A diagram of an SE building block is shown in Fig. 1.

----
SE 构建块的图如图 1 所示。


![图1](image-2.png)

##### 3.1. Squeeze：全局信息嵌入
In order to tackle the issue of exploiting channel dependencies, we first consider the signal to each channel in the output features.

----
为了解决利用通道依赖性的问题，我们首先考虑输出特征中每个通道的信号。

Each of the learned filters operates with a local receptive field and consequently each unit of the transformation output U is unable to exploit contextual information outside of this region.

----
每个学习过滤器都使用局部感受野进行操作，因此变换输出 U 的每个单元都无法利用该区域之外的上下文信息。

This is an issue that becomes more severe in the lower layers of the network whose receptive field sizes are small.

----
在感受野尺寸较小的网络较低层中，这个问题变得更加严重。

To mitigate this problem, we propose to squeeze global spatial information into a channel descriptor.

----
为了缓解这个问题，我们建议将全局空间信息压缩到通道描述符中。

This is achieved by using global average pooling to generate channel-wise statistics.

----
这是通过使用全局平均池化生成通道统计数据来实现的。

Formally, a statistic z ∈ RC is generated by shrinking U through spatial dimensions H × W , where the c-th element of z is calculated by:

----
形式上，统计量 z ∈ RC 是通过通过空间维度 H × W 缩小 U 来生成的，其中 z 的第 c 个元素的计算公式为：


![公式2](99\01日志\00图片\20240710\公式2.png)

Discussion：讨论：

The transformation output U can be interpreted as a collection of the local descriptors whose statistics are expressive for the whole image.

----
变换输出 U 可以解释为局部描述符的集合，其统计数据可以表达整个图像。

Exploiting such information is prevalent in feature engineering work [35, 38, 49].

----
利用此类信息在特征工程工作中很普遍[35,38,49]。

We opt for the simplest, global average pooling, noting that more sophisticated aggregation strategies could be employed here as well.

----
我们选择最简单的全局平均池，并指出这里也可以采用更复杂的聚合策略。


##### 3.2.激励：自适应重新校准
To make use of the information aggregated in the squeeze operation, we follow it with a second operation which aims to fully capture channel-wise dependencies.

----
为了利用挤压操作中聚合的信息，我们接下来进行第二个操作，旨在完全捕获通道方面的依赖关系。

To fulfil this objective, the function must meet two criteria: first, it must be flexible (in particular, it must be capable of learning a nonlinear interaction between channels) and second, it must learn a non-mutually-exclusive relationship since we would like to ensure that multiple channels are allowed to be emphasised opposed to one-hot activation.

----
为了实现这一目标，该函数必须满足两个标准：首先，它必须是灵活的（特别是，它必须能够学习通道之间的非线性交互），其次，它必须学习非互斥关系，因为我们将就像确保允许强调多个通道而不是单热激活。

To meet these criteria, we opt to employ a simple gating mechanism with a sigmoid activation:

----
为了满足这些标准，我们选择采用带有 sigmoid 激活的简单门控机制：

![公式3](00图片/20240710/公式3.png)

where δ refers to the ReLU [30] function, W1 ∈ R C r ×C and W2 ∈ RC× C r . To limit model complexity and aid generalisation, we parameterise the gating mechanism by forming a bottleneck with two fully connected (FC) layers around the non-linearity, i.e. a dimensionality-reduction layer with parameters W1 with reduction ratio r (this parameter choice is discussed in Sec. 6.4), a ReLU and then a dimensionalityincreasing layer with parameters W2.

----
其中 δ 指 ReLU [30] 函数，W1 ∈ R C r ×C 且 W2 ∈ RC× C r 。为了限制模型复杂性并帮助泛化，我们通过在非线性周围形成两个全连接（FC）层的瓶颈来参数化门控机制，即具有参数 W1 和缩减比 r 的降维层（此参数选择已讨论） （第 6.4 节），一个 ReLU，然后是一个参数为 W2 的维数增加层。

The final output of the block is obtained by rescaling the transformation output U with the activations:

----
该块的最终输出是通过使用激活重新调整变换输出 U 来获得的：

![公式4](00图片/20240710/公式4.png)

where  ̃ X = [ ̃ x1,  ̃ x2, . . . ,  ̃ xC ] and Fscale(uc, sc) refers to channel-wise multiplication between the feature map uc ∈ RH×W and the scalar sc.

----
其中 ̃ X = [ ̃ x1, ̃ x2, . 。 。 , ̃ xC ] 和 F Scale(ucsc) 指的是特征图 uc ∈ RH×W 和标量 sc 之间的通道乘法。


Discussion. The activations act as channel weights adapted to the input-specific descriptor z. In this regard, SE blocks intrinsically introduce dynamics conditioned on the input, helping to boost feature discriminability.

----
讨论。激活充当适应输入特定描述符 z 的通道权重。在这方面，SE 块本质上引入了以输入为条件的动态，有助于提高特征辨别力。

It is straightforward to apply the SE block to AlexNet [21] and VGGNet [39].

----
将 SE 块应用于 AlexNet [21] 和 VGGNet [39] 非常简单。

The flexibility of the SE block means that it can be directly applied to transformations beyond standard convolutions.

----
SE 块的灵活性意味着它可以直接应用于标准卷积之外的变换。

To illustrate this point, we develop SENets by integrating SE blocks into modern architectures with sophisticated designs.

----
为了说明这一点，我们通过将 SE 块集成到具有复杂设计的现代架构中来开发 SENet。

For non-residual networks, such as Inception network, SE blocks are constructed for the network by taking the transformation Ftr to be an entire Inception module (see Fig. 2).

----
对于非残差网络，例如Inception网络，通过将变换Ftr作为整个Inception模块来构造SE块（见图2）。


![图2](99\01日志\00图片\20240710\图2.png)

By making this change for each such module in the architecture, we construct an SE-Inception network. Moreover, SE blocks are sufficiently flexible to be used in residual networks.

----
通过对架构中的每个此类模块进行此更改，我们构建了一个 SE-Inception 网络。此外，SE 块足够灵活，可以在残差网络中使用。

Fig. 3 depicts the schema of an SEResNet module.

----
图 3 描述了 SEResNet 模块的架构。

![图3](99\01日志\00图片\20240710\图3.png)


Here, the SE block transformation Ftr is taken to be the non-identity branch of a residual module.

----
这里，SE块变换Ftr被视为残差模块的非恒等分支。

Squeeze and excitation both act before summation with the identity branch.

----
挤压和激励都在与恒等分支求和之前起作用。

More variants that integrate with ResNeXt [47], Inception-ResNet [42], MobileNet [13] and ShuffleNet [52] can be constructed by following the similar schemes. We describe the architecture of SE-ResNet-50 and SE-ResNeXt-50 in Table 1.

----
可以通过遵循类似的方案来构建与 ResNeXt [47]、Inception-ResNet [42]、MobileNet [13] 和 ShuffleNet [52] 集成的更多变体。我们在表 1 中描述了 SE-ResNet-50 和 SE-ResNeXt-50 的架构。


![表1](99\01日志\00图片\20240710\表1.png)

#### 4. 模型和计算复杂性
For the proposed SE block to be viable in practice, it must provide an effective trade-off between model complexity and performance which is important for scalability.

----
为了使所提出的 SE 块在实践中可行，它必须在模型复杂性和性能之间提供有效的权衡，这对于可扩展性非常重要。

We set the reduction ratio r to be 16 in all experiments, except where stated otherwise (more discussion can be found in Sec. 6.4).

----
除非另有说明，我们在所有实验中将减速比 r 设置为 16（更多讨论可在第 6.4 节中找到）。

To illustrate the cost of the module, we take the comparison between ResNet-50 and SE-ResNet-50 as an example, where the accuracy of SE-ResNet-50 is superior to ResNet-50 and approaches that of a deeper ResNet101 network (shown in Table 2).

----
为了说明模块的成本，我们以ResNet-50和SE-ResNet-50之间的比较为例，其中SE-ResNet-50的准确率优于ResNet-50并接近更深的ResNet101网络（如表2所示）。

![表2](00图片/20240710/表2.png)


ResNet-50 requires ∼3.86 GFLOPs in a single forward pass for a 224 × 224 pixel input image.

----
对于 224 × 224 像素输入图像，ResNet-50 在单次前向传递中需要 ∼3.86 GFLOP。

Each SE block makes use of a global average pooling operation in the squeeze phase and two small fully connected layers in the excitation phase, followed by an inexpensive channel-wise scaling operation.

----
每个 SE 块在挤压阶段使用全局平均池化操作，在激励阶段使用两个小型全连接层，然后进行廉价的通道缩放操作。

In aggregate, SE-ResNet-50 requires ∼3.87 GFLOPs, corresponding to a 0.26% relative increase over the original ResNet-50.

----
总的来说，SE-ResNet-50 需要 ∼3.87 GFLOP，相对于原始 ResNet-50 增加了 0.26%。

In practice, with a training mini-batch of 256 images, a single pass forwards and backwards through ResNet-50 takes 190 ms, compared to 209 ms for SE-ResNet-50 (both timings are performed on a server with 8 NVIDIA Titan X GPUs).

----
实际上，对于 256 个图像的小批量训练，通过 ResNet-50 向前和向后单次传递需要 190 毫秒，而 SE-ResNet-50 则需要 209 毫秒（这两个时间均在具有 8 个 NVIDIA Titan X 的服务器上执行） GPU）。

We argue that this represents a reasonable overhead particularly since global pooling and small inner-product operations are less optimised in existing GPU libraries.

----
我们认为，这代表了合理的开销，特别是因为现有 GPU 库中的全局池化和小型内积运算优化程度较低。

Moreover, due to its importance for embedded device applications, we also benchmark CPU inference time for each model: for a 224 × 224 pixel input image, ResNet-50 takes 164 ms, compared to 167 ms for SE-ResNet-50.

----
此外，由于其对嵌入式设备应用的重要性，我们还对每个模型的 CPU 推理时间进行了基准测试：对于 224 × 224 像素输入图像，ResNet-50 需要 164 毫秒，而 SE-ResNet-50 需要 167 毫秒。

The small additional computational overhead required by the SE block is justified by its contribution to model performance.

----
SE 块所需的少量额外计算开销因其对模型性能的贡献而得到证明。

Next, we consider the additional parameters introduced by the proposed block.

----
接下来，我们考虑所提出的块引入的附加参数。

All of them are contained in the two FC layers of the gating mechanism, which constitute a small fraction of the total network capacity.

----
它们全部包含在门控机制的两个 FC 层中，仅占总网络容量的一小部分。

More precisely, the number of additional parameters introduced is given by:

----
更准确地说，引入的附加参数的数量由下式给出：

![公式5](00图片/20240710/公式5.png)

where r denotes the reduction ratio, S refers to the number of stages (where each stage refers to the collection of blocks operating on feature maps of a common spatial dimension), Cs denotes the dimension of the output channels and Ns denotes the repeated block number for stage s. SEResNet-50 introduces ∼2.5 million additional parameters beyond the ∼25 million parameters required by ResNet-50, corresponding to a ∼10% increase.

----
其中r表示缩减率，S表示阶段数（其中每个阶段是指在公共空间维度的特征图上操作的块的集合），Cs表示输出通道的维度，Ns表示重复块数对于 阶段 s.除了 ResNet-50 所需的 2500 万个参数之外，SEResNet-50 还引入了 250 万个额外参数，相当于增加了 10%。

The majority of these parameters come from the last stage of the network, where excitation is performed across the greatest channel dimensions.

----
这些参数大部分来自网络的最后阶段，其中激励是在最大通道维度上执行的。

However, we found that the comparatively expensive final stage of SE blocks could be removed at a marginal cost in performance (<0.1% top-1 error on ImageNet) to reduce the relative parameter increase to ∼4%, which may prove useful in cases where parameter usage is a key consideration (see further discussion in Sec. 6.4).

----
然而，我们发现相对昂贵的 SE 块的最后阶段可以以边际性能成本（ImageNet 上的 top-1 误差<0.1%）去除，以将相对参数增加减少到~4%，这在某些情况下可能有用其中参数的使用是一个关键考虑因素（请参阅第 6.4 节中的进一步讨论）。

#### 5. 实施

Each plain network and its corresponding SE counterpart are trained with identical optimisation schemes.

----
每个普通网络及其相应的 SE 对应部分都使用相同的优化方案进行训练。

During training on ImageNet, we follow standard practice and perform data augmentation with random-size cropping [43] to 224 × 224 pixels (299 × 299 for Inception-ResNet-v2 [42] and SE-Inception-ResNet-v2) and random horizontal flipping.

----
在 ImageNet 训练期间，我们遵循标准实践，并通过随机大小裁剪 [43] 至 224 × 224 像素（Inception-ResNet-v2 [42] 和 SE-Inception-ResNet-v2 为 299 × 299）和随机执行数据增强水平翻转。

Input images are normalised through mean channel subtraction.

----
输入图像通过平均通道减法进行归一化。

In addition, we adopt the data balancing strategy described in [36] for mini-batch sampling. The networks are trained on our distributed learning system “ROCS” which is designed to handle efficient parallel training of large networks.

----
此外，我们采用[36]中描述的数据平衡策略进行小批量采样。这些网络在我们的分布式学习系统“ROCS”上进行训练，该系统旨在处理大型网络的高效并行训练。

Optimisation is performed using synchronous SGD with momentum 0.9 and a mini-batch size of 1024. The initial learning rate is set to 0.6 and decreased by a factor of 10 every 30 epochs.

----
使用动量为 0.9、小批量大小为 1024 的同步 SGD 进行优化。初始学习率设置为 0.6，每 30 个 epoch 降低 10 倍。

All models are trained for 100 epochs from scratch, using the weight initialisation strategy described in [9].

----
所有模型都使用[9]中描述的权重初始化策略从头开始训练 100 个时期。

When testing, we apply a centre crop evaluation on the validation set, where 224×224 pixels are cropped from each image whose shorter edge is first resized to 256 (299 × 299 from each image whose shorter edge is first resized to 352 for Inception-ResNet-v2 and SE-Inception-ResNet-v2).

----
测试时，我们在验证集上应用中心裁剪评估，其中从短边首先调整为 256 的每张图像中裁剪 224×224 像素（对于 Inception，从短边首先调整为 352 的每张图像裁剪 299 × 299 像素） ResNet-v2 和 SE-Inception-ResNet-v2）。

#### 6. 实验

##### 6.1. ImageNet 分类
The ImageNet 2012 dataset is comprised of 1.28 million training images and 50K validation images from 1000 classes. We train networks on the training set and report the top-1 and the top-5 errors.

----
ImageNet 2012 数据集由来自 1000 个类别的 128 万张训练图像和 5 万张验证图像组成。我们在训练集上训练网络并报告 top-1 和 top-5 错误。

Network depth. We first compare the SE-ResNet against ResNet architectures with different depths. The results in Table 2 shows that SE blocks consistently improve performance across different depths with an extremely small increase in computational complexity.

----
网络深度。我们首先将 SE-ResNet 与不同深度的 ResNet 架构进行比较。表 2 中的结果表明，SE 块在不同深度上持续提高性能，而计算复杂度的增加极小。

Remarkably, SE-ResNet-50 achieves a single-crop top-5 validation error of 6.62%, exceeding ResNet-50 (7.48%) by 0.86% and approaching the performance achieved by the much deeper ResNet-101 network (6.52% top-5 error) with only half of the computational overhead (3.87 GFLOPs vs. 7.58 GFLOPs).

----
值得注意的是，SE-ResNet-50 的单作物 top-5 验证误差为 6.62%，比 ResNet-50 (7.48%) 提高了 0.86%，接近更深的 ResNet-101 网络（6.52% top-5）所实现的性能。 5 个错误），而计算开销只有一半（3.87 GFLOPs 与 7.58 GFLOPs）。

This pattern is repeated at greater depth, where SE-ResNet-101 (6.07% top-5 error) not only matches, but outperforms the deeper ResNet-152 network (6.34% top-5 error) by 0.27%.

----
这种模式在更深的深度上重复，其中 SE-ResNet-101（6.07% top-5 错误）不仅匹配，而且比更深的 ResNet-152 网络（6.34% top-5 错误）高出 0.27%。

Fig. 4 depicts the training and validation curves of SE-ResNet-50 and ResNet-50 (the curves of more networks are shown in supplementary material).

----
图4描绘了SE-ResNet-50和ResNet-50的训练和验证曲线（更多网络的曲线显示在补充材料中）。

While it should be noted that the SE blocks themselves add depth, they do so in an extremely computationally efficient manner and yield good returns even at the point at which extending the depth of the base architecture achieves diminishing returns.

----
虽然应该指出的是，SE 块本身增加了深度，但它们以极其计算高效的方式实现了这一点，并且即使在扩展基础架构的深度实现收益递减的情况下，也能产生良好的回报。


![图4](99\01日志\00图片\20240710\图4.png)

Moreover, we see that the performance improvements are consistent through training across a range of different depths, suggesting that the improvements induced by SE blocks can be used in combination with increasing the depth of the base architecture.

----
此外，我们发现通过一系列不同深度的训练，性能改进是一致的，这表明 SE 块带来的改进可以与增加基础架构的深度结合使用。

Integration with modern architectures. We next investigate the effect of combining SE blocks with another two state-of-the-art architectures, Inception-ResNet-v2 [42] and ResNeXt (using the setting of 32 × 4d) [47], which both introduce prior structures in modules.

----
与现代建筑的融合。接下来，我们研究将 SE 块与另外两种最先进的架构 Inception-ResNet-v2 [42] 和 ResNeXt（使用 32 × 4d 的设置）[47] 相结合的效果，这两种架构都在模块。

We construct SENet equivalents of these networks, SEInception-ResNet-v2 and SE-ResNeXt (the configuration of SE-ResNeXt-50 is given in Table 1). The results in Table 2 illustrate the significant performance improvement induced by SE blocks when introduced into both architectures.

----
我们构建了这些网络的 SENet 等效项：SEInception-ResNet-v2 和 SE-ResNeXt（SE-ResNeXt-50 的配置在表 1 中给出）。表 2 中的结果说明了将 SE 块引入两种架构时带来的显着性能改进。

In particular, SE-ResNeXt-50 has a top-5 error of 5.49% which is superior to both its direct counterpart ResNeXt-50 (5.90% top-5 error) as well as the deeper ResNeXt-101 (5.57% top-5 error), a model which has almost double the number of parameters and computational overhead.

----
特别是，SE-ResNeXt-50 的 top-5 错误率为 5.49%，优于其直接对应的 ResNeXt-50（5.90% top-5 错误）以及更深的 ResNeXt-101（5.57% top-5）错误），该模型的参数数量和计算开销几乎翻倍。

As for the experiments of Inception-ResNetv2, we conjecture the difference of cropping strategy might lead to the gap between their reported result and our reimplemented one, as their original image size has not been clarified in [42] while we crop the 299 × 299 region from a relatively larger image (where the shorter edge is resized to 352).

----
对于 Inception-ResNetv2 的实验，我们推测裁剪策略的差异可能会导致他们报告的结果与我们重新实现的结果之间的差距，因为它们的原始图像尺寸在[42]中尚未明确，而我们裁剪了 299 × 299来自相对较大图像的区域（其中较短边缘调整为 352）。


SE-Inception-ResNet-v2 (4.79% top-5 error) outperforms our reimplemented Inception-ResNet-v2 (5.21% top-5 error) by 0.42% (a relative improvement of 8.1%) as well as the reported result in [42].

----
SE-Inception-ResNet-v2（4.79% top-5 错误）比我们重新实现的 Inception-ResNet-v2（5.21% top-5 错误）高出 0.42%（相对改进 8.1%），以及 [ 42]。

We also assess the effect of SE blocks when operating on non-residual networks by conducting experiments with the VGG-16 [39] and BN-Inception architecture [16]. As a deep network is tricky to optimise [16, 39], to facilitate the training of VGG-16 from scratch, we add a Batch Normalization layer after each convolution.

----
我们还通过使用 VGG-16 [39] 和 BN-Inception 架构 [16] 进行实验来评估 SE 块在非残差网络上运行时的效果。由于深度网络很难优化[16, 39]，为了便于从头开始训练 VGG-16，我们在每个卷积之后添加了一个 Batch Normalization 层。

We apply the identical scheme for training SE-VGG-16. The results of the comparison are shown in Table 2, exhibiting the same phenomena that emerged in the residual architectures.

----
我们应用相同的方案来训练 SE-VGG-16。比较结果如表 2 所示，表现出与剩余架构中出现的相同现象。

Finally, we evaluate on two representative efficient architectures, MobileNet [13] and ShuffleNet [52] in Table 3, showing that SE blocks can consistently improve the accuracy by a large margin at minimal increases in computational cost.

----
最后，我们对表 3 中的两种代表性高效架构 MobileNet [13] 和 ShuffleNet [52] 进行了评估，结果表明 SE 块可以在计算成本增加最小的情况下持续大幅度提高准确性。

These experiments demonstrate that improvements induced by SE blocks can be used in combination with a wide range of architectures. Moreover, this result holds for both residual and non-residual foundations.

----
这些实验表明，SE 模块带来的改进可以与各种架构结合使用。此外，该结果对于残余地基和非残余地基均成立。

Results on ILSVRC 2017 Classification Competition. SENets formed the foundation of our submission to the competition where we won first place. Our winning entry comprised a small ensemble of SENets that employed a standard multi-scale and multi-crop fusion strategy to obtain a 2.251% top-5 error on the test set. One of our highperforming networks, which we term SENet-154, was constructed by integrating SE blocks with a modified ResNeXt [47] (details are provided in supplemental material), the goal of which is to reach the best possible accuracy with less emphasis on model complexity. We compare it with the top-performing published models on the ImageNet validation set in Table 4. Our model achieved a top-1 error of 18.68% and a top-5 error of 4.47% using a 224 × 224 centre crop evaluation. To enable a fair comparison, we provide a 320 × 320 centre crop evaluation, showing a significant performance improvement on prior work. After the competition, we train an SENet-154 with a larger input size 320 × 320, achieving the lower error rate under both the top-1 (16.88%) and the top-5 (3.58%) error metrics.

----
ILSVRC 2017 分类赛结果。 SENets 为我们提交竞赛奠定了基础，并赢得了第一名。我们的获奖作品由一小部分 SENet 组成，它采用标准的多尺度和多作物融合策略，在测试集上获得了 2.251% 的 top-5 错误率。我们的高性能网络之一，我们称之为 SENet-154，是通过将 SE 块与修改后的 ResNeXt [47] 集成而构建的（补充材料中提供了详细信息），其目标是在不那么强调的情况下达到尽可能最佳的准确度模型复杂度。我们将其与表 4 中 ImageNet 验证集上表现最好的已发布模型进行比较。使用 224 × 224 中心裁剪评估，我们的模型实现了 18.68% 的 top-1 误差和 4.47% 的 top-5 误差。为了进行公平的比较，我们提供了 320 × 320 中心裁剪评估，显示了之前工作的显着性能改进。比赛结束后，我们训练了具有更大输入尺寸 320 × 320 的 SENet-154，在 top-1 (16.88%) 和 top-5 (3.58%) 错误指标下实现了较低的错误率。

##### 6.2.场景分类
We conduct experiments on the Places365-Challenge dataset [53] for scene classification. It comprises 8 million training images and 36, 500 validation images across 365 categories. Relative to classification, the task of scene understanding can provide a better assessment of the ability of a model to generalise well and handle abstraction, since it requires the capture of more complex data associations and robustness to a greater level of appearance variation.

----
我们在 Places365-Challenge 数据集 [53] 上进行场景分类实验。它包含 800 万张训练图像和 365 个类别的 36、500 张验证图像。相对于分类，场景理解任务可以更好地评估模型的泛化能力和处理抽象的能力，因为它需要捕获更复杂的数据关联以及对更高水平的外观变化的鲁棒性。

We use ResNet-152 as a strong baseline to assess the effectiveness of SE blocks and follow the training and evaluation protocols in [37]. Table 5 shows the results of ResNet-152 and SE-ResNet-152. Specifically, SE-ResNet152 (11.01% top-5 error) achieves a lower validation error than ResNet-152 (11.61% top-5 error), providing evidence that SE blocks can perform well on different datasets. This SENet surpasses the previous state-of-the-art model Places365-CNN [37] which has a top-5 error of 11.48% on this task.

----
我们使用 ResNet-152 作为强大的基线来评估 SE 块的有效性，并遵循[37]中的训练和评估协议。表5显示了ResNet-152和SE-ResNet-152的结果。具体来说，SE-ResNet152（11.01% top-5 错误）实现了比 ResNet-152（11.61% top-5 错误）更低的验证错误，这证明 SE 块可以在不同数据集上表现良好。该 SENet 超越了之前最先进的模型 Places365-CNN [37]，该模型在此任务上的 top-5 错误率为 11.48%。



<br>[返回标题](#squeeze-and-excitation-networks)


<br>[返回目录](#目录)
