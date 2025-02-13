## Overview  概述
In this competition, you’ll develop models to improve the prediction of transplant survival rates for patients undergoing allogeneic Hematopoietic Cell Transplantation (HCT) — an important step in ensuring that every patient has a fair chance at a successful outcome, regardless of their background.
在本次竞赛中，您将开发模型来改进对接受同种异体造血细胞移植 （HCT） 的患者移植存活率的预测，这是确保每位患者都有公平机会获得成功结果的重要一步，无论其背景如何。
## Description  描述
Improving survival predictions for allogeneic HCT patients is a vital healthcare challenge. Current predictive models often fall short in addressing disparities related to socioeconomic status, race, and geography. Addressing these gaps is crucial for enhancing patient care, optimizing resource utilization, and rebuilding trust in the healthcare system.
改善同种异体 HCT 患者的生存预测是一项重要的医疗保健挑战。当前的预测模型往往无法解决与社会经济地位、种族和地理相关的差异。解决这些差距对于加强患者护理、优化资源利用和重建对医疗保健系统的信任至关重要。

This competition aims to encourage participants to advance predictive modeling by ensuring that survival predictions are both precise and fair for patients across diverse groups. By using synthetic data—which mirrors real-world situations while protecting patient privacy—participants can build and improve models that more effectively consider diverse backgrounds and conditions.
该竞赛旨在通过确保对不同群体的患者的生存预测既精确又公平，来鼓励参与者推进预测建模。通过使用合成数据（在保护患者隐私的同时反映真实世界的情况），参与者可以构建和改进模型，从而更有效地考虑不同的背景和条件。

You’re challenged to develop advanced predictive models for allogeneic HCT that enhance both accuracy and fairness in survival predictions. The goal is to address disparities by bridging diverse data sources, refining algorithms, and reducing biases to ensure equitable outcomes for patients across diverse race groups. Your work will help create a more just and effective healthcare environment, ensuring every patient receives the care they deserve.
您面临的挑战是开发用于同种异体 HCT 的高级预测模型，以提高生存预测的准确性和公平性。目标是通过弥合不同的数据源、改进算法和减少偏差来解决差异，以确保不同种族群体的患者获得公平的结果。您的工作将有助于创造更加公正和有效的医疗保健环境，确保每位患者都能获得应有的护理。

## Evaluation  评估
### Evaluation Criteria  评估标准
The evaluation of prediction accuracy in the competition will involve a specialized metric known as the Stratified Concordance Index (C-index), adapted to consider different racial groups independently. This method allows us to gauge the predictive performance of models in a way that emphasizes equitability across diverse patient populations, particularly focusing on racial disparities in transplant outcomes.
比赛中预测准确性的评估将涉及一个称为分层一致性指数 （C-index） 的专门指标，适用于独立考虑不同的种族群体。这种方法使我们能够以强调不同患者群体的公平性的方式衡量模型的预测性能，特别是关注移植结果的种族差异。

### Concordance index  一致性索引
It represents the global assessment of the model discrimination power: this is the model’s ability to correctly provide a reliable ranking of the survival times based on the individual risk scores. 
它代表了对模型鉴别能力的全局评估：这是模型根据个体风险评分正确提供可靠生存时间排名的能力。

The concordance index is a value between 0 and 1 where:
一致性索引是介于 0 和 1 之间的值，其中：

0.5 is the expected result from random predictions,
0.5 是随机预测的预期结果，
1.0 is a perfect concordance and,
1.0 是一个完美的索引，并且
0.0 is perfect anti-concordance (multiply predictions with -1 to get 1.0)
0.0 是完美的 anti-concordance（将预测乘以 -1 得到 1.0）


### Stratified Concordance Index
分层一致性索引
For this competition, we adjust the standard C-index to account for racial stratification, thus ensuring that each racial group's outcomes are weighed equally in the model evaluation. The stratified c-index is calculated as the mean minus the standard deviation of the c-index scores calculated within the recipient race categories, i.e., the score will be better if the mean c-index over the different race categories is large and the standard deviation of the c-indices over the race categories is small. This value will range from 0 to 1, 1 is the theoretical perfect score, but this value will practically be lower due to censored outcomes.
对于本次比赛，我们调整标准 C 指数以考虑种族分层，从而确保每个种族群体的结果在模型评估中得到同等的权重。分层的 c 指数计算为平均值减去在接受者种族类别中计算的 c 指数分数的标准差，即，如果不同种族类别的平均 c 指数较大，而种族类别的 c 指数的标准差较小，则分数会更好。该值的范围从 0 到 1,1 是理论满分，但由于删失结果，该值实际上会更低。

The submitted risk scores will be evaluated using the score function. This evaluation process involves comparing the submitted risk scores against actual observed values (i.e., survival times and event occurrences) from a test dataset. The function specifically calculates the stratified concordance index across different racial groups, ensuring that the predictions are not only accurate overall but also equitable across diverse patient demographics.
提交的风险评分将使用评分函数进行评估。该评估过程包括将提交的风险评分与测试数据集中的实际观察值（即生存时间和事件发生率）进行比较。该函数专门计算不同种族群体的分层一致性指数，确保预测不仅总体准确，而且在不同的患者人口统计数据中也是公平的。

## Submission File  提交文件
Participants must submit their predictions for the test dataset as real-valued risk scores. These scores represent the model's assessment of each patient's risk following transplantation. A higher risk score typically indicates a higher likelihood of the target event occurrence.
参与者必须将他们对测试数据集的预测作为实际价值的风险评分提交。这些分数代表模型对每位患者移植后风险的评估。较高的风险评分通常表示目标事件发生的可能性较高。

The submission file must include a header and follow this format:
提交文件必须包含标头并遵循以下格式：

ID,prediction
28800,0.5
28801,1.2
28802,0.8
etc.
where:  哪里：

ID refers to the identifier for each patient in the test dataset.
ID 是指测试数据集中每个患者的标识符。
prediction is the corresponding risk score generated by your model.
prediction 是您的模型生成的相应风险评分。

## Timeline  时间线
December 4, 2024 - Start Date.
2024 年 12 月 4 日 - 开始日期。

February 26, 2025 - Entry Deadline. You must accept the competition rules before this date in order to compete.
2025 年 2 月 26 日 - 报名截止日期。您必须在此日期之前接受比赛规则才能参加比赛。

February 26, 2025 - Team Merger Deadline. This is the last day participants may join or merge teams.
2025 年 2 月 26 日 - 团队合并截止日期。这是参与者可以加入或合并团队的最后一天。

March 5, 2025 - Final Submission Deadline.
2025 年 3 月 5 日 - 最终提交截止日期。

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.
除非另有说明，否则所有截止日期均为相应日期的 11：59 PM UTC。比赛组织者保留在他们认为必要时更新比赛时间表的权利。

## Code Requirements  代码要求
This is a Code Competition
这是一场代码竞赛
Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:
必须通过 Notebooks 提交本次比赛。为了在提交后激活“提交”按钮，必须满足以下条件：

CPU Notebook <= 9 hours run-time
CPU 笔记本 <= 9 小时运行时间
GPU Notebook <= 9 hours run-time
GPU 笔记本 <= 9 小时运行时间
Internet access disabled  已禁用 Internet 访问
Freely & publicly available external data is allowed, including pre-trained models
允许自由和公开的外部数据，包括预训练模型
Submission file must be named submission.csv
提交文件必须命名为 submission.csv
Submission runtimes are slightly obfuscated.
提交运行时略微模糊处理。
Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you are encountering submission errors.
有关如何提交的更多信息，请参阅 Code Competition 常见问题解答。如果您遇到提交错误，请查看代码调试文档。

## Background Information  背景信息

### What is an allogeneic HCT?
什么是同种异体 HCT？
The human immune system comprises cells that develop from hematopoietic stem cells, a special type of cells that reside in the bone marrow. These stem cells are responsible for generating all blood cells, including red blood cells, platelet-producing cells, and immune system cells such as T cells, B cells, neutrophils, and natural killer (NK) cells. Allogeneic hematopoietic cell transplantation (HCT) can be used to replace an individual's faulty hematopoietic stem cells with stem cells that can produce normal immune system cells. In other words, a successful HCT can help fix a person's immune system by introducing healthy stem cells into their body. When hematopoietic stem cells are transferred from one person to another, the recipient is referred to as the HCT recipient. The term "allogeneic" indicates that the stem cells being used come from someone else, the hematopoietic stem cell donor. If the HCT is successful, the donor's hematopoietic stem cells will replace the recipient's cells, producing blood and immune system cells that work correctly.
人体免疫系统由造血干细胞发育而来的细胞组成，造血干细胞是存在于骨髓中的一种特殊类型的细胞。这些干细胞负责产生所有血细胞，包括红细胞、血小板生成细胞和免疫系统细胞，如 T 细胞、B 细胞、中性粒细胞和自然杀伤 （NK） 细胞。同种异体造血细胞移植 （HCT） 可用于用可以产生正常免疫系统细胞的干细胞替换个体有缺陷的造血干细胞。换句话说，成功的 HCT 可以通过将健康的干细胞引入体内来帮助修复一个人的免疫系统。当造血干细胞从一个人转移到另一个人时，受体被称为 HCT 受体。术语“同种异体”表示所使用的干细胞来自其他人，即造血干细胞供体。如果 HCT 成功，供体的造血干细胞将取代受体的细胞，产生正常工作的血液和免疫系统细胞。

The source of hematopoietic stem cells can be bone marrow, peripheral blood, or umbilical cord blood. Depending on the source of the stem cells, HCT procedures may be called bone marrow transplants (BMT), peripheral blood stem cell transplants, or cord blood transplants.
造血干细胞的来源可以是骨髓、外周血或脐带血。根据干细胞的来源，HCT 手术可能称为骨髓移植 （BMT）、外周血干细胞移植或脐带血移植。

More information on how blood stem cell transplants work.
有关造血干细胞移植工作原理的更多信息。

The competition hosts CIBMTR and NMDP have saved over 130,000 lives through cell therapy.
比赛主办方 CIBMTR 和 NMDP 通过细胞疗法挽救了超过 130,000 人的生命。



