# 基于深度学习的听诊分析用于筛查需要血管成形术的血液透析的天然动静脉瘘显著狭窄的可行性

## 基本信息
- 标题：
- 分区：
- 期刊：
- 发表时间：年
- 阅读时间：2024年1月
- 关键词：
- 作者：
- 国家：韩国
- 方法：
- 创新点：
- 不足之处：
- 可借鉴之处：
- DOI:

## [ChatPaper](https://chatpaper.org/)分析结果

### Basic Information:
Title: Feasibility of Deep Learning-Based Analysis of Auscultation for Screening Significant Stenosis of Native Arteriovenous Fistula for Hemodialysis Requiring Angioplasty (血液透析所需动脉-静脉瘘听诊深度学习分析筛查重要狭窄的可行性研究)
Authors: Jae Hyon Park, Insun Park, Kichang Han, Jongjin Yoon, Yongsik Sim, Soo Jin Kim, Jong Yun Won, Shina Lee, Joon Ho Kwon, Sungmo Moon, Gyoung Min Kim, Man-deuk Kim
Affiliation: Department of Radiology, Yonsei University College of Medicine, Seoul, Korea (韩国首尔延世大学医学院放射科)
Keywords: Angioplasty, Deep learning, Arteriovenous fistula, Auscultation, Renal dialysis
URLs: [Paper](https://doi.org/10.3348/kjr.2022.0364), [GitHub: None]
### 论文简要 :
本研究旨在探讨使用基于深度学习的听诊数据分析来预测血液透析所需经皮血管成形术的动脉-静脉瘘重要狭窄的可行性。

### 背景信息:
#### 论文背景: 
血液透析是终末期肾脏疾病患者的主要肾脏替代治疗方法，需要一个正常运作的动脉-静脉瘘或动脉-静脉移植物。然而，随着时间的推移，血管血栓或狭窄会发生，动脉-静脉瘘往往变得功能失常。因此，准确诊断重要狭窄的动脉-静脉瘘并及时干预对于保持透析通路至关重要。
#### 过去方案: 
根据过去的方法，通过定期体格检查，包括触诊和听诊，由具有中等证据质量的医务人员对动脉-静脉瘘狭窄进行筛查。然而，基于声音的听诊诊断可能是主观的，并且依赖于医务人员的临床经验。此外，即使是经过训练的医务人员也无法仅凭听诊来量化狭窄的严重程度。
#### 论文的Motivation: 
鉴于经皮血管成形术的主要适应症是重要狭窄（管腔直径≥50%）或阻塞，仅凭听诊很难准确评估患者是否需要经皮血管成形术。因此，使用深度学习对听诊进行量化和特征提取，以检测需要经皮血管成形术的重要狭窄的存在，可以帮助医务人员筛查需要血液透析的ESRD患者。
### 方法:
#### a. 理论背景:

本研究旨在探讨使用基于深度学习的听诊数据分析来预测血液透析患者动静脉瘘（AVF）的严重狭窄，以及及时干预的可行性。血液透析是终末期肾脏疾病（ESRD）患者的主要肾脏替代疗法，需要一个功能正常的动静脉瘘或动静脉移植。然而，由于血管血栓或狭窄，动静脉瘘往往会随着时间的推移而失去功能。因此，准确诊断严重的动静脉瘘狭窄并及时干预对于保持透析通路至关重要。
#### b. 技术路线:

本研究招募了40名有功能失常的原发性动静脉瘘（AVF）的ESRD患者。在PTA之前和之后，使用无线电子听诊器记录了AVF分流的数字声音，并将音频文件转换为mel频谱图。构建了多种深度卷积神经网络（DCNN）模型，并评估和比较了它们对诊断≥50% AVF狭窄的性能。
### 结果:
#### a. 详细的实验设置:

研究人群包括40名ESRD患者，他们有功能失常的原发性动静脉瘘（AVF）。使用无线电子听诊器记录了AVF分流的数字声音，在经皮经血管成形术（PTA）之前和之后。将音频文件转换为mel频谱图，用于构建深度卷积神经网络（DCNN）模型。
#### b. 详细的实验结果:

ResNet50和EfficientNetB5模型在预测≥50% AVF狭窄的优化时期分别获得了0.99和0.98的AUC值。然而，Grad-CAM热图显示，只有ResNet50在mel频谱图中突出显示与AVF狭窄相关的区域。

### 讨论(方法、创新点、不足之处)
#### 方法：

该研究使用了深度学习方法，通过分析听诊数据来预测血管内瘘（AVF）的狭窄程度。
研究中使用了无线电子听诊器记录了AVF分流的数字声音，并将其转换为mel频谱图。
使用了三种深度卷积神经网络（DenseNet201、EfficientNetB5和ResNet50）构建了模型，并评估了它们对诊断≥ 50% AVF狭窄的性能。
#### 创新：

该研究首次尝试使用深度学习方法分析听诊数据来预测AVF狭窄。
通过使用mel频谱图作为输入，研究人员成功地构建了DCNN模型，并使用Grad-CAM生成了与AVF狭窄相关的可视化解释。
#### 缺点：

该研究的样本量较小，可能存在选择偏倚。
由于AVF分流声音的稀缺性，研究中使用了合成的mel频谱图进行数据增强。
#### 参考领域：

该研究为使用深度学习方法预测AVF狭窄提供了参考。
该研究的方法和模型可以在AVF监测和筛查中有潜在应用。
请注意，以上总结是根据文献中的相关内容进行的，可能不包含所有细节。 Pages: [8, 1, 9]

### 模型
根据文献中的相关内容，本研究使用了以下模型：

DenseNet201

EfficientNetB5

ResNet50

这些模型是常用的卷积神经网络结构，用于构建用于预测血流动力学显著性动静脉瘘狭窄的DCNN模型。在Conv-pool层之后添加了两个全连接层，使用修正线性单元作为激活函数，其中第一个全连接层有2048个神经元，第二个全连接层有2048个神经元。为了正则化和避免模型过拟合，在第一个和第二个全连接层之后添加了两个dropout层（dropout率为0.5）。对于二分类任务，使用具有一个神经元的最后一层，并使用softmax激活函数。DCNN模型使用ImageNet权重进行初始化，并使用分类交叉熵作为损失函数，使用学习率为0.0001的均方根传播优化器进行编译。模型使用批量大小为10和50个epochs进行训练。每个数据集随机划分为训练集、验证集和测试集，划分比例分别为70%、10%和20%。使用梯度加权类激活映射（Grad-CAM）生成DCNN模型决策的可视化解释。

这些模型在预测血流动力学显著性狭窄方面表现出良好的性能。在优化的epochs（DenseNet201≥40个epochs，EfficientNetB5≥12个epochs，ResNet50≥19个epochs）时，DenseNet201、EfficientNetB5和ResNet50模型的AUROC分别为0.70、0.98和0.99。然而，DenseNet201和EfficientNetB5的Grad-CAM热图显示了与AVF狭窄无关的mel频谱中的区域。相反，ResNet50的Grad-CAM热图突出显示了与pre-PTA和post-PTA分流声音之间的高频和中频边界以及中频和低频边界有关的区域。

这些模型的实现使用了Google Colab平台，使用了Python 3.10.4版本以及NumPy、pandas、scikit-learn、TensorFlow和Keras等Python库。

模型的性能评估使用了接收者操作特征曲线下面积（AUROC）以及混淆矩阵、精确度、准确度、召回率和F-1分数等指标。模型的性能评估基于≥ 0.5的诊断阈值对最终模型输出进行分类。≥ 50% AVF狭窄的真实情况基于DSA的结果。统计分析使用了Google Colab或SAS软件版本9.4进行。

综上所述，本文献中使用了DenseNet201、EfficientNetB5和ResNet50等模型来构建DCNN模型，用于预测血流动力学显著性动静脉瘘狭窄。这些模型在预测狭窄方面表现出良好的性能，并且通过Grad-CAM热图提供了对模型决策的可视化解释。

### Note:
本总结源自于LLM的总结，请注意数据判别. Power by ChatPaper. End.

## Abstract

