# 儿童重症监护中早期预测急性肾损伤（AKI）的机器学习模型

## 基本信息
- 标题：Machine learning model for early prediction of acute kidney injury (AKI) in pediatric critical care
- 分区：
- 期刊：Critical Care
- 发表时间：2021年
- 阅读时间：2024年1月
- 关键词：Acute kidney injury, AKI, Pediatric critical care, Machine learning, Predictive model
- 作者：Junzi Dong1* , Ting Feng1, Binod Thapa‑Chhetry1, Byung Gu Cho1
- 国家：美国
- 方法：
- 创新点：
- 不足之处：
- 可借鉴之处：
- DOI:https://doi.org/10.1186/s13054-021-03724-0


## [ChatPaper](https://chatpaper.org/)分析结果

### Basic Information:
Title: Machine learning model for early prediction of acute kidney injury (AKI) in pediatric critical care (机器学习模型用于儿科重症监护中早期预测急性肾损伤)
Authors: Junzi Dong, Ting Feng, Binod Thapa-Chhetry, Byung Gu Cho, Tunu Shum, David P. Inwald, Christopher J. L. Newth, Vinay U. Vaidya
Affiliation: Connected Care and Personal Health Team, Philips Research North America, 222 Jacobs Street, Cambridge, MA 02141, USA (美国飞利浦研究北美公司)
Keywords: Acute kidney injury, AKI, Pediatric critical care, Machine learning, Predictive model
URLs: [Paper](https://doi.org/10.1186/s13054-021-03724-0), [GitHub: None]
### 论文简要 :
本研究开发了一个机器学习模型，通过学习生理测量的预病模式，能够在儿科重症监护中提前48小时预测急性肾损伤（AKI），以及提供相关信息和建议的预警，有望通过早期干预措施改善儿科AKI的预后。
### 背景信息:
#### 论文背景: 
儿科重症监护患者中有四分之一会发生急性肾损伤（AKI），这与更高的死亡率、更长的住院时间和随后发展为慢性肾脏疾病有关。目前，AKI的诊断依赖于血清肌酐和尿量，但肾功能损伤通常在肌酐升高之前发生，因此目前的诊断指南只能在肾损伤或功能障碍已经出现后检测到AKI。早期预测AKI对于识别患者的风险并及早干预以改善预后非常重要。
#### 过去方案: 
目前已有许多研究团队利用电子健康记录（EHR）数据进行AKI的早期预测，但迄今为止没有一个模型能够解释特定预测的原因，尽管明确需要可解释和可操作的预测。在儿科患者中，由于生理特点随年龄差异较大，开发一个能够学习儿科早期AKI的年龄适宜特征的预测模型仍然是一个额外的挑战。
#### 论文的Motivation: 
本研究旨在开发一个能够实时运行的儿科AKI预测模型，能够检测患者生理学的微小变化并提醒护理人员高风险的AKI患者，并提供可解释的背景信息和建议措施。主要目标是在AKI发生前的6到48小时内预测中度到重度AKI的发生。此外，该模型还对包括任何AKI（1/2/3期）的发生和需要肾脏替代治疗（RRT）的预测进行了评估。据我们所知，这是第一个能够解释每个预测的AKI预测模型，并且是儿科重症监护AKI预测的第一个经过多中心验证的模型。
### 方法:
#### a. 理论背景:

本研究开发了一个机器学习模型，可以在儿科重症监护患者中提前48小时预测急性肾损伤（AKI），比目前已建立的诊断指南更早。该模型使用生理测量和智能工程预测因子来评估实时的AKI风险，并生成警报以便快速评估和降低AKI风险。
#### b. 技术路线:

本研究使用来自16,863名儿科重症监护患者的电子健康记录（EHR）数据，开发了一个机器学习模型，用于早期预测AKI。数据被分为推导、验证和保留测试数据集。该模型使用智能工程预测因子（如肌酐变化率）进行训练，并在保留测试数据上进行验证。主要结果是在发生中度到重度AKI之前，能够预测其发生的6到48小时。
结果:
#### a. 详细的实验设置:

该机器学习模型成功地在常规标准检测之前预测了2/3期AKI，中位提前时间为30小时。该模型还预测了相当比例的肾脏替代治疗（RRT）和任何AKI发作。在预测的患者中，有很高比例的患者在被模型识别出来后但在发生AKI之前接受了潜在的肾毒性药物。
#### b. 详细的实验结果:

本研究中描述的机器学习模型能够准确预测中度到重度AKI的发生，提前48小时。通过提供早期警报和可操作的反馈，它可能通过实施早期措施（如药物调整）来改善结果，从而预防或减少AKI的发生。这是首个为所有儿科重症监护患者验证的多中心AKI预测模型。
### Note:
本总结源自于LLM的总结，请注意数据判别. Power by ChatPaper. End.



## Abstract
### Background:



### Methods:





