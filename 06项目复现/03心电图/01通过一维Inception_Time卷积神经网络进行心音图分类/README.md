# Phonocardiogram Classification Using 1-Dimensional Inception TimeConvolutional Neural Networks

通过一维Inception Time卷积神经网络进行心音图分类

> 最开始是在[paperwithcode](https://paperswithcode.com/paper/phonocardiogram-classification-using-1)上看到的这篇文章 。

> 使用[CirCor DigiScope](https://paperswithcode.com/dataset/circor-digiscope)数据库，这个数据库相关的文献还有不少，大概有十篇，建议全部看一下，还挺有趣的。

## Abstract

心脏杂音是由湍流血流引起的声音，通常是患者结构性心脏疾病的首个征兆。通过听诊器或最近的心音图（PCG）可以检测到这些声音。我们旨在利用机器学习从PCG记录中识别心脏杂音的存在、不存在或不确定情况，以及预测临床结果是否正常或异常。

我们对一个由1568个儿童患者组成的PCG数据集进行了两个一维卷积神经网络（CNN）的训练和测试。其中一个模型用于预测心脏杂音，而另一个模型用于预测临床结果。这两个模型都是按照记录方式进行训练的，但最终的预测是针对每个患者的（患者为单位的预测）。

本论文描述了我们参与了2022年George B. Moody PhysioNet Challenge。该挑战的目标是从心音图记录中识别心脏杂音和临床结果。我们的团队Simulab训练了一个临床结果分类器，在挑战成本得分中获得了8720分（在305个提交中排名第1），而杂音分类器在验证集上实现了加权准确率为0.585（在305个提交中排名第182）。

## 1.Introduction
d

本论文描述了我们在George B. Moody PhysioNet Challenge 2022中的方法。该挑战的目标是根据来自一个或多个听诊位置的PCG记录，检测杂音的存在、不存在或不确定，并将患者的临床结果分类为正常或异常。



## 2.Method
我们采用了一种监督式机器学习方法，使用单个PCG信号和卷积神经网络（CNN）来检测杂音并对患者的临床结果进行分类。我们使用Python（3.8.9）和Tensorflow（2.8.2）实现了这些模型。该代码将被开源，并发布在GitHub上[开源代码]（https://github.com/Bsingstad/Heart-murmur-detection-2022-Simulab）。



### 2.1.Data
这项工作使用的数据集包含来自1568名儿童患者的5272个PCG记录[12, 13]。其中3163个PCG记录来自942名患者，用于训练。剩下的2109个PCG记录来自149个和477名患者，仅提供给挑战的组织者用于验证和测试。每个患者可能有一个或多个PCG记录，记录位置可以靠近主动脉瓣、肺动脉瓣、三尖瓣、二尖瓣，或者在某些情况下未知。

每个患者都被标记为临床结果（异常/正常）和心脏杂音（存在/未知/不存在），由临床专家进行注释[4]。在出现心脏杂音的情况下，在训练集中提供了记录杂音的位置信息。



### 2.2.Pre-processing




#### 2.2.1. Signal processing

在训练数据中，PCG信号的采样频率为4000Hz。我们将所有信号进行了降采样到100Hz的处理。此外，我们对训练数据中的所有信号进行了零填充，使得所有信号的长度都等于最长信号的长度（6451个样本）。在验证和测试数据中，也使用了6451个样本作为阈值。长度小于6451的信号会用长度为6451-l的零填充尾部，长度大于6451的信号会被截断。



#### 2.2.2. Label processing

数据集从以患者为单位的标签转换为以记录为单位的标签。这是通过将同一患者的所有PCG记录与原始的整体标签相匹配来完成的。然而，杂音的以记录为单位的重新标记过程显示在算法1中。




### 2.3.Models

我们训练了两个分类模型：一个用于分类杂音，另一个用于分类临床结果。这两个模型都是一维CNN，采用Inception Time架构[14]。其中，杂音模型是多类别分类器，用于分类心音记录中的杂音是否存在/不存在/未知。临床结果模型是二元分类器，用于分类患者的临床结果是否正常或异常。杂音分类器使用加权分类交叉熵进行训练，而临床结果分类器使用加权二元交叉熵进行训练。两个模型中的权重与类别的患病率呈反比关系确定。


### 2.4.Post-processing
模型的记录级别预测最终被转换回患者级别的预测。杂音的转换过程显示在算法2中，临床结果的转换过程显示在算法3中。

#### 算法2
该算法是一个心脏杂音检测算法。它接受以下输入参数：p（患者）、r（心音图记录）、t（总体群体）、l（标签）。

算法的输出是pnl（每个患者的标签）。

算法主要步骤如下：

1. 对于每个n属于t中的患者：
2. 如果对于任何一个rl在pn中为Absent（即杂音不存在），则将pnl设置为Absent（表示杂音缺失）。
3. 否则，如果对于任何一个rl在pn中为Present（即杂音存在），则将pnl设置为Present（表示杂音存在）。
4. 否则，如果对于任何一个rl在pn中为Unknown（即杂音状态未知），则将pnl设置为Unknown（表示杂音状态未知）。

换句话说，该算法基于心音图记录中的杂音线索（rl）来确定每个患者（pn）的标签（pnl）。如果杂音被确定为不存在，则标签被设置为Absent；如果杂音被确定为存在，则标签被设置为Present；如果无法确定杂音的存在或缺失，则标签被设置为Unknown。通过遍历整个患者总体来判断每个患者的标签，并将最终结果存储在pnl中。

#### 算法3

该算法是一个结局（心脏状态）判断算法。它接受以下输入参数：p（患者）、r（心音图记录）、t（总体群体）、l（标签）。

算法的输出是pnl（每个患者的标签）。

算法主要步骤如下：

1. 对于每个n属于t中的患者：
2. 如果对于任何一个rl在pn中为Abnormal（即心脏状态异常），则将pnl设置为Abnormal（表示心脏状态异常）。
3. 否则，如果对于任何一个rl在pn中为Normal（即心脏状态正常），则将pnl设置为Normal（表示心脏状态正常）。

换句话说，该算法基于心音图记录中的异常线索（rl）来确定每个患者（pn）的标签（pnl）。如果存在任何一个异常线索，则最终标签被设置为Abnormal；否则，如果所有线索都指示正常，标签则被设置为Normal。通过遍历整个患者总体来判断每个患者的标签，并将最终结果存储在pnl中。

### 2.5.Model selection (local development)
为了评估模型的性能，并找到最佳的模型架构和超参数，在将最终模型提交给挑战组织者之前，我们在训练集上进行了本地开发。使用5折交叉验证（CV）对患者级别进行分层，将训练数据划分为本地训练集和验证集，并在每一轮迭代中训练和验证新模型。通过这种方式，我们寻找到了最优的模型架构和超参数组合。





### 2.6.Submitted model
本地开发中找到的最佳模型和超参数被用于通过向组织者提交我们的代码使用Docker镜像来训练最终的模型。这些模型在整个训练集上进行了训练，然后应用于隐藏的验证集。



## 3.Results
杂音分类器经过30个epoch的训练，而临床结果分类器经过20个epoch的训练。这两个模型使用批量大小为20进行训练，并使用学习率为0.001的Adam优化器。表1显示了在训练数据集上的交叉验证结果，包括加权准确率、挑战成本、准确率和F-measure，而验证集上的结果仅给出了杂音模型的加权准确率和临床结果模型的挑战成本。

Note: The table mentioned in your question is missing, so I cannot provide the specific values of the results.


## 4.Discussion and conclusion
在隐藏的验证集上取得的结果比训练集上的交叉验证结果显著好。临床结果分类器甚至在隐藏的验证集上的挑战成本得分上超过了所有其他George B Moody Challenge 2022竞争者的分类器。然而，这些令人惊喜的好结果可能是偶然发生的，但在临床结果排名中我们得到了5个属于前16名的得分，平均挑战成本得分为9079±254。与验证集相比，训练集上的性能差异可能是由于两个数据集中类别分布的不同，但这只是推测，因为验证集对挑战参与者是隐藏的。

同时还尝试使用2016年PhysioNet挑战数据来进行模型的预训练[15, 16]。我们尝试了不同的方法来继续对预训练模型进行训练，如冻结除最后一层之外的所有层、少量/大量的epoch和高/低的学习率。然而，在训练集上的交叉验证期间没有显著的改进，并且验证集上的性能相对于无预训练的情况下有所下降。

杂音分类器和临床结果分类器都是使用单个PCG记录进行训练的，其中未考虑听诊位置。然而，在挑战的初期阶段，我们还尝试了多通道PCG分类器，但它们被单通道分类器性能更好地超越。这个观察结果以及我们的分类器与验证集上其他竞争者的分类器相比的性能支持了一个假设，即CNN可以在不知道听诊位置的情况下从PCG记录中检测异常。这一发现可能对基于CNN的PCG分类器的进一步发展产生影响。然而，还需要进一步的研究来深入解释这些CNN如何解读PCG。对这些模型可解释性的更多关注可能会得出有临床意义的有趣发现。

## References

[1] 世界卫生组织心血管疾病（CVDs）. 链接：https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds). 访问日期：2022年8月21日。

[2] Tavel ME. 心脏听诊。《循环》2006年3月；113（9）：1255-1259，DOI：https://doi.org/10.1161/CIRCULATIONAHA.105.591149。

[3] Montinari MR, Minelli S. 心脏听诊的前200年和未来展望。《多学科医疗保健杂志》2019年3月；12：183-189，DOI：https://doi.org/10.2147/JMDH.S193904。

[4] Reyna MA, Kiarashi Y, Elola A, Oliveira J, Renna F, Gu A等. 来自心音图记录的心脏杂音检测：George B. Moody PhysioNet Challenge 2022. 预印本，Health Informatics，2022年8月。DOI：https://doi.org/10.1101/2022.08.11.22278688。

[5] Vukanovic-Criley JM, Criley S, Warde CM, Boker JR, Guevara-Matheus L, Churchill WH等. 医学生、实习生、医师和教员心脏检查技能能力。《内科档案》2006年3月；166（6）：610-616，DOI：https://doi.org/10.1001/archinte.166.6.610。

[6] Roy D, Sargeant J, Gray J, Hoyt B, Allen M, Fleming M. 帮助家庭医生通过互动式光盘-ROM提高心脏听诊技能。《继续医学教育杂志》2002年；22（3）：152-159，DOI：https://doi.org/10.1002/chp.1340220304。

[7] Barrett MJ, Mackie AS, Finley JP. 现代化中的心脏听诊：夭折还是重生？《心脏病学评论》2017年10月；25（5）：205-210，DOI：https://doi.org/10.1097/CRD.0000000000000145。

[8] Rangayyan RM, Lehner RJ. 心音图信号分析：综述《生物医学工程评论》1987年；15（3）：211-236。

[9] Emmanuel BS. 临床诊断中心脏声音分析的信号处理技术综述。《医学工程技术杂志》2012年8月；36（6）：303-307，DOI：https://doi.org/10.3109/03091902.2012.684831。

[10] Debbal ￥￥, Bereksi-Reguig F. 计算机化的心脏声音分析。《计算生物学与医学》2008年2月；38（2）：263-280，DOI：https://doi.org/10.1016/j.compbiomed.2007.09.006。

[11] Chen W, Sun Q, Chen X, Xie G, Wu H, Xu C. 基于深度学习的心音分类方法：系统综述。《熵》2021年5月；23（6）：667，DOI：https://doi.org/10.3390/e23060667。

[12] Goldberger AL等. PhysioBank、PhysioToolkit和PhysioNet：复杂生理信号的新研究资源组成部分。《循环》2000年6月；101（23），DOI：https://doi.org/10.1161/01.CIR.101.23.e215。

[13] Oliveira J, Renna F, Costa PD, Nogueira M, Oliveira C, Ferreira C等. CirCor DigiScope数据集：从杂音检测到杂音分类。IEEE生物医学与健康信息学杂志2022年6月；26（6）：2524-2535，DOI：https://doi.org/10.1109/JBHI.2021.3137048。

[14] Ismail Fawaz等. InceptionTime：时间序列分类的AlexNet。《数据挖掘与知识发现》2020年11月；34（6）：1936-1962，DOI：https://doi.org/10.1007/s10618-020-00710-y。

[15] Clifford G等. 正常/异常心音记录的分类：PhysioNet/计算心脏病学挑战2016年。《Circulation》2016年9月；URL http://www.cinc.org/archives/2016/pdf/179-154.pdf。

[16] Clifford GD, Liu C, Moody B, Millet J, Schmidt S, Li Q等. 心脏声音分析的最新进展。《生理测量学》2017年8月；38（8）：E10，DOI：https://doi.org/10.1088/1361-6579/aa7ec8。




01通过一维Inception_Time卷积神经网络进行心音图分类 copy通讯地址：
Bjørn-Jostein Singstad
挪威奥斯陆0164号Kristian Augustus Gate 23
电子邮件：bjornjs@simula.no


[<<返回心音图目录](06项目复现\03心电图/)
