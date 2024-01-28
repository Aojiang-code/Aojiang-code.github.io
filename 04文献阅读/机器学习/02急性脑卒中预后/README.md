# 儿童重症监护中早期预测急性肾损伤（AKI）的机器学习模型

## 基本信息
- 标题：Machine Learning–Based Model for Prediction of  Outcomes in Acute Stroke
- 分区：
- 期刊：Stroke
- 发表时间：2019年
- 阅读时间：2024年1月
- 关键词：
- 作者：JoonNyung Heo, MD*; Jihoon G. Yoon, MD*; Hyungjong Park, MD; Young Dae Kim, MD, PhD;  Hyo Suk Nam, MD, PhD; Ji Hoe Heo, MD, PhD
- 国家：韩国
- 方法：
- 创新点：
- 不足之处：
- 可借鉴之处：
- DOI:10.1161/STROKEAHA.118.024293


## [ChatPaper](https://chatpaper.org/)分析结果

### Basic Information:
- Title: Machine Learning–Based Model for Prediction of Outcomes in Acute Stroke (中风急性期预测结果的机器学习模型)
- Authors: JoonNyung Heo, Jihoon G. Yoon, Hyungjong Park, Young Dae Kim, Hyo Suk Nam, Ji Hoe Heo
- Affiliation: Department of Neurology, Yonsei University College of Medicine, Seoul, Korea (韩国延世大学医学院神经内科)
- Keywords: cerebral infarction, machine learning, medical decision making, neural networks, stroke
- URLs: [Paper](https://www.ahajournals.org/doi/10.1161/STROKEAHA.118.024293), [GitHub: None]
### 论文简要 :
本研究开发了基于机器学习的模型，用于预测急性期中风患者的结果，其中深度神经网络模型表现出更高的准确性，有助于改善中风患者的长期预后预测。(This study developed a machine learning-based model for predicting outcomes in acute stroke patients, with the deep neural network model demonstrating higher accuracy and improving long-term prognosis prediction in stroke patients.)
### 背景信息:
#### 论文背景: 
中风患者的长期预后预测对治疗决策和管理预后期望具有重要意义。然而，传统的预测模型在描述人体生理复杂性方面存在局限性，而机器学习算法则能更好地描述这种复杂性。(Background: Predicting long-term outcomes in stroke patients is crucial for treatment decisions and managing prognostic expectations. However, traditional predictive models have limitations in describing the complexity of human physiology, while machine learning algorithms can better capture this complexity.)

#### 过去方案: 
为了解决中风患者预后预测的问题，已经开发了多种预测评分系统。然而，随着机器学习技术的进步，将其应用于医学领域已经取得了令人期待的结果。(Past methods: To address the issue of stroke prognosis prediction, several prognostic scoring systems have been developed. However, with the advancements in machine learning techniques, applying these techniques in the medical field has shown promising results.)

#### 论文的Motivation: 
鉴于机器学习技术对中风管理的预期影响，本研究开发了使用机器学习技术预测中风长期结果的模型，并将其与已知的预后模型进行了比较。(Motivation: Considering the expected impact of machine learning on stroke management, this study developed models using machine learning techniques to predict long-term stroke outcomes and compared their predictability to a well-known prognostic model.)

### 方法:
#### a. 理论背景:

本研究旨在探究机器学习技术在缺血性中风患者长期预后预测中的适用性。研究人员使用急性缺血性中风患者的前瞻性队列进行了一项回顾性研究。他们开发了三种机器学习模型（深度神经网络、随机森林和逻辑回归），并将其预测能力与Acute Stroke Registry and Analysis of Lausanne（ASTRAL）评分进行了比较。研究包括2604名患者，有利的预后定义为3个月时的修正Rankin量表得分为0、1或2。
#### b. 技术路线:

研究人员使用了三种机器学习模型：深度神经网络、随机森林和逻辑回归。
深度神经网络模型在预测能力方面明显优于ASTRAL评分，曲线下面积为0.888，而ASTRAL评分为0.839。
随机森林和逻辑回归模型与ASTRAL评分在性能上没有显著差异。
当仅使用ASTRAL评分的变量时，机器学习模型的性能与ASTRAL评分的性能没有显著差异。
### 结果:
#### a. 详细的实验设置:

本研究证明了机器学习算法，特别是深度神经网络，可以改善缺血性中风患者的长期预后预测。深度神经网络模型在包含较不重要变量的情况下表现明显优于ASTRAL评分，而在仅使用ASTRAL评分的变量作为输入时与ASTRAL评分表现相似。
与ASTRAL评分相比，机器学习模型的预测能力提高不大，尤其考虑到输入机器学习模型的许多变量的负担增加。然而，随着电子健康记录系统的改进，自动计算已内置于系统中，从而减少了模型简化的需求。
本研究还强调了一些限制，包括单中心研究和需要使用其他数据源进行验证。
#### b. 详细的实验结果:

深度神经网络模型的预测能力明显优于ASTRAL评分，曲线下面积为0.888，而ASTRAL评分为0.839。
随机森林和逻辑回归模型与ASTRAL评分在性能上没有显著差异。
当仅使用ASTRAL评分的变量时，机器学习模型的性能与ASTRAL评分的性能没有显著差异。

### 讨论(方法、创新点、不足之处)
#### 方法
这篇文献研究了使用机器学习技术预测缺血性中风患者的长期预后。研究采用了回顾性研究方法，纳入了2010年1月至2014年12月期间入院的所有缺血性中风患者。研究使用了38个变量，包括患者的人口统计学特征、中风临床评分、发病到入院时间等。研究开发了三个机器学习模型（深度神经网络、随机森林和逻辑回归），并将其与Acute Stroke Registry and Analysis of Lausanne (ASTRAL)评分进行了比较。研究结果显示，深度神经网络模型的曲线下面积显著高于ASTRAL评分（0.888 vs 0.839），而随机森林模型（0.857）和逻辑回归模型（0.849）的曲线下面积与ASTRAL评分没有显著差异。使用ASTRAL评分所使用的6个变量，机器学习模型的性能与ASTRAL评分的性能没有显著差异。研究表明，机器学习算法，特别是深度神经网络，可以改善缺血性中风患者的长期预后预测能力。
#### 创新点
该研究的创新点在于将机器学习技术应用于缺血性中风患者的长期预后预测，并与传统的ASTRAL评分进行比较。研究结果表明，机器学习模型在预测缺血性中风患者长期预后方面具有较高的准确性。
#### 不足之处
然而，该研究也存在一些局限性。首先，该研究是一项回顾性研究，可能存在信息偏倚。其次，研究样本来自于单个医疗中心，可能存在地域性差异。此外，机器学习模型的预测性能可能受到变量的可用性和其他中心数据的影响。
#### 启示
从这篇文献中我们可以得到的启示是，机器学习技术在医学领域中的应用具有潜力，可以改善缺血性中风患者的长期预后预测能力。然而，我们需要进一步研究和验证这些模型的可靠性和适用性，以便将其应用于临床实践中。 Pages: [1, 3]

### Note:
本总结源自于LLM的总结，请注意数据判别. Power by ChatPaper. End.

## Abstract
### Background and Purpose


### Methods



### Results



### Conclusions



### Key Words
cerebral infarction ◼ machine learning ◼ medical decision making ◼ neural networks ◼ stroke
脑梗死 ◼ 机器学习 ◼ 医学决策 ◼ 神经网络 ◼ 中风

## Background
在缺血性中风患者中预测长期结局可能对治疗决策和处理预后期望具有用处。已经开发了几个用于这个目的的预后评分系统[1]。在机器学习的最新进展下，该技术在医学领域的应用已经取得了令人期待的结果[2]。人体生理的复杂和不可预测性往往更适合通过机器学习算法进行描述。与传统的预测模型使用选定变量进行计算不同，机器学习技术可以轻松地整合大量变量，因为所有计算都是由计算机完成的[3]。这些特点使得机器学习技术非常适用于医学领域。在中风中，机器学习技术在各个领域中的应用日益增多，包括内腔治疗后的结局预测[4,5]。
考虑到其对缺血性中风管理的预期影响，我们利用机器学习技术开发了预测长期中风结局的模型。然后，我们将其与众所周知的预后模型 Acute Stroke Registry and Analysis of Lausanne (ASTRAL) 评分进行了比较[1]。

## Methods
支持本研究结果的数据可根据合理请求从对应作者处获取。这是一项回顾性研究，使用了一个纳入在症状发作后7天内入院的缺血性中风患者的前瞻性队列注册研究。对于本研究，我们包括了2010年1月至2014年12月间入院的所有患者。我们排除了具有大于2分的中风前修改Rankin量表（mRS）评分、3个月时缺失mRS评分或接受再通治疗的患者。功能性结局在3个月时确定（在线附加资料），并且将mRS评分为0、1或2定义为有利结局。该研究获得了延世大学医疗体系伦理审查委员会的批准，并因研究的回顾性性质而获得豁免知情同意书。


### Data and Machine Learning Algorithms
为了开发机器学习模型，我们选择了38个变量，包括患者的人口统计学数据、初始美国国立卫生研究院中风量表评分、发病到入院时间、基于急性中风治疗的ORG 10472试验分类系统的中风亚型、以往疾病和药物治疗史、实验室检查结果以及3个月时的mRS评分（在线附加资料中的表格I）。

我们使用了三种机器学习算法：深度神经网络、随机森林和逻辑回归[3,6]。深度神经网络由若干层相互连接的人工神经元组￥￥￥工神经元根据生物神经元自身进行设计，接收多个输入并与权重相乘后输出输入的总和。随机森林算法由多个决策树组成，每个决策树根据输入变量进行多个真假条件判断。决策树所做的决策总和用于最终分类。

我们使用所有变量作为输入来训练机器学习模型，以分类可能具有良好结局的患者。对于深度神经网络模型，我们使用了3个包含15个人工神经网络单元的隐藏层。对于随机森林模型，我们使用了300个决策树。为了评估机器学习模型的准确性，我们计算了ASTRAL评分作为参考，该评分是急性中风的一种已建立的预后评分系统之一。

我们还研究了当使用ASTRAL评分的6个用于计算的变量（年龄、美国国立卫生研究院中风量表评分、发病到入院延迟、视野缺陷、血糖和意识水平下降）作为输入时，机器学习模型如何预测结果。对于这个分析，机器学习模型使用了ASTRAL评分的6个变量进行训练。深度神经网络模型使用了一个包含4个人工神经网络单元的隐藏层，而随机森林模型使用了150个决策树。

在研究人群中，随机选择了67％（n=1744）作为训练集，剩下的33％（n=858）则作为测试集，以防止模型出现过拟合。TensorFlow 1.1.0版本（Google）和scikit-learn工具包0.18.1版本（Google）用于训练机器学习模型[7]。


### Statistical Analyses
统计分析使用R软件包版本3.3.2进行。使用pROC软件包计算接受者操作特征曲线分析和曲线下面积，以比较每个模型的有效性。P值小于0.05的变量被认为是统计学上显著的，所有的P值均为双侧检验。


## Results
在研究期间，共有3522名患者被纳入该队列。排除了453名3个月mRS评分不可用的患者、60名中风前mRS评分大于2的患者、87名丢失实验室检查或临床数据的患者以及318名接受溶栓治疗的患者，最终包括了2604名患者（图1）。这2604名患者的平均年龄为66.2±12.6岁，其中61.7%为男性。入选和排除患者之间人口统计学变量的比较详见在线附加资料（在线附加资料中的表II）。


### Comparison of the Models for the Prediction of  Favorable Outcomes
在2604名患者中，有2043名（78%）患者具有良好的结局。深度神经网络模型的表现明显优于ASTRAL评分（曲线下面积为0.888 [95% CI，0.873-0.903] 对比0.839 [0.822-0.855]；P<0.001）。然而，随机森林模型（0.857 [0.840-0.874]）和逻辑回归模型（0.849 [0.831-0.867]）的表现与ASTRAL评分相似（分别为P=0.136，P=0.413；图2）。

当仅使用ASTRAL评分的变量时，机器学习模型的表现与ASTRAL评分相似（深度神经网络模型的为0.853 [95% CI, 0.835-0.871]，P=0.255；随机森林模型为0.828 [95% CI, 0.808-0.847]，P=0.396；逻辑回归模型为0.846 [95% CI, 0.828-0.865]，P=0.541；在线附加资料中的表III）。


## Discussion
本研究表明，机器学习模型的使用可以准确预测急性中风患者的长期结局。根据统计分析计算的预后评分由于其简单性而仅使用了一些关键变量，并且尽可能地进行了系数舍入。然而，许多因素影响中风的结局，这些变量可能会对预测产生甚至轻微的影响。事实上，我们的研究表明，在包含了较不关键的变量的情况下，深度神经网络模型的表现比ASTRAL评分显著更好，而当仅使用ASTRAL评分的变量作为输入时，两者的表现相似。

本研究证明了深度神经网络模型优于其他模型。深度神经网络模型本身可能更适合于预测结果。复杂网络的多层结构可以有效地表示中风患者结局的复杂性质。然而，改进后的性能背后的理论基础尚不清楚。

与ASTRAL评分相比，预测能力的改善很小，特别是考虑到为机器学习模型输入许多变量的增加负担。然而，随着电子健康记录系统的改进，自动计算已经内置于系统中，因此减少了模型简化的需求。此外，考虑到机器学习模型可以通过额外数据进行自我学习，上述结果有待改善但具有潜力。

本研究存在一些限制。这是一项单中心研究，需要使用其他来源的数据进行验证。作为机器学习算法输入的变量通常适用于大多数情况下可获得或评估的变量。然而，预测可能会根据变量稍微受到影响，并且当结合其他中心的数据时，可能需要根据可获取性进行调整。接受再通治疗的患者被排除在外，原因是这些患者的预后主要受治疗本身的变量影响，而这些变量仅适用于接受该治疗的患者亚组。


## Conclusions
本研究证明了机器学习算法，特别是深度神经网络，可以提高对缺血性中风患者的长期结果预测能力。


## Sources of Funding
本研究得到了韩国￥￥￥资助的国家研究基金会（NRF）基础科学研究计划（NRF-2018R1A2A3074996）的支持。


## Disclosures



## 


