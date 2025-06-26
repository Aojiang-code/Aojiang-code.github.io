### 同行评审文件

**文章信息**：https://dx.doi.org/10.21037/jtd-2025-264

#### 评审意见
##### **Comment 1**
**意见 1**：研究是在单一中心进行的，这可能限制了研究结果对其他人群或医疗环境的普适性。这可能会影响该模型在不同人口分布或临床环境中的表现。

**回复 1**：在“局限性”小节中，进一步阐述单中心研究设计对研究结果普适性的影响，增加以下内容：

“首先，本研究为单中心研究，固有地受到我们中心特定的患者人口统计特征、医疗资源和临床实践模式的限制。因此，研究结果可能无法直接推广到其他中心。不同中心的患者人群在遗传背景、合并症特征和手术协议方面存在差异，这可能导致该模型在应用于不同队列时预测性能的波动。为了增强我们研究结果的普适性，后续研究将积极建立多中心合作，纳入更广泛的患者数据集，重新评估模型的有效性和稳定性，并加强模型的普遍适用性。”

**文本更改**：第294-303行

**Comment 1**: The study is conducted at a single center, which may limit the generalizability of the results to other populations or healthcare settings. This could affect the model's performance in different demographic distributions or clinical environments.

**Reply 1**: In the "Limitations" subsection, further elaborate on the impact of single-center study design on the generalizability of findings by adding the following content:

"First, this study was conducted as a single-center investigation, inherently limited by the demographic characteristics of patients, medical resources, and clinical practice patterns specific to our center. Consequently, the research findings may not be directly generalizable to other centers. Variability exists among patient populations across different centers in terms of genetic backgrounds, comorbidity profiles, and surgical protocols, which may lead to fluctuations in the model's predictive performance when applied to diverse cohorts. To enhance the universality of our findings, subsequent studies will actively establish multi-center collaborations to incorporate broader patient datasets, re-evaluate the model's validity and stability, and strengthen the general applicability of the model."

**Changes in the text: Line 294-303**
##### **Comment 2**
**意见 2**：研究的回顾性性质可能会引入偏差，例如选择偏差或信息偏差，这可能会影响研究结果的可靠性。前瞻性研究通常在建立因果关系方面更具说服力。

**回复 2**：在“局限性”小节中，进一步强调回顾性研究设计固有的潜在偏差问题，增加以下内容：

“其次，回顾性研究设计不可避免地存在选择偏差和信息偏差的风险。在本研究中，尽管我们为患者选择实施了严格的纳入和排除标准，并仔细验证数据以尽量减少信息偏差，但这些措施无法完全消除残余偏差的影响。因此，我们计划开展前瞻性研究，以验证本研究中的模型和结果，从而更准确地评估术前/术中危险因素与术后AKI发生率之间的关联。”

**文本更改**：第304-310行

**Comment 2**: The retrospective nature of the study might introduce biases, such as selection bias or information bias, which could impact the reliability of the findings. Prospective studies are generally more robust in establishing cause-and-effect relationships.

**Reply 2**: In the "Limitations" subsection, further emphasize the potential bias issues inherent in retrospective study designs by adding:

"Second, retrospective study designs inevitably carry risks of selection bias and information bias. In this study, despite implementing stringent inclusion and exclusion criteria for patient selection and meticulously verifying data to minimize information bias, these measures could not entirely eliminate the impact of residual biases. Therefore, we plan to conduct a prospective study to validate the models and findings from this research, enabling more precise evaluation of the associations between preoperative/intraoperative risk factors and postoperative AKI incidence."

**Changes in the text: Line 304-310**
##### **Comment 3**

**意见 3**：研究排除了缺失数据超过20%的变量，这可能会遗漏重要的预测因子。尽管使用了均值填补方法处理缺失值，但这种方法可能无法准确反映真实数据分布，从而可能影响模型性能。

**回复 3**：除了目前使用的均值填补方法外，后续研究将探索更先进的处理技术，例如多重填补方法和基于机器学习的填补算法（例如K最近邻算法、随机森林填补算法），以优化模型性能。我们将重新评估之前因高缺失率而从模型中排除的变量的影响，以确保不会遗漏重要的预测因子。

**文本更改**：第314-316行

**Comment 3**: The study excludes variables with more than 20% missing data, which could omit important predictors. While mean imputation is used for missing values, this method may not always accurately reflect the true data distribution, potentially affecting model performance.

**Reply 3**: Regarding missing data, in addition to the currently employed mean imputation method, subsequent studies will explore more advanced processing techniques such as multiple imputation methods and machine learning-based imputation algorithms (e.g., K-Nearest Neighbors algorithm, Random Forest imputation algorithm) to optimize model performance. We will re-evaluate the impact of high-missingness variables previously excluded from the model to ensure that important predictive factors are not omitted.

**Changes in the text: Line 314-316**
##### **Comment 4**

**意见 4**：尽管研究使用了验证集，但未报告外部验证。在不同数据集或环境中进行外部验证对于确认模型在更广泛背景下的预测能力至关重要。

**回复 4**：在“局限性”小节中，增加以下内容关于外部验证：

“第三，本研究仅使用内部验证集进行模型评估，未进行外部验证。我们计划积极寻找合适的外部数据集进行外部验证。通过比较模型在不同数据集上的性能指标，我们旨在进一步完善模型评估框架。”

**文本更改**：第310-314行

**Comment 4**: Although the study uses a validation set, it does not report external validation. External validation in different datasets or settings is crucial to confirm the model's predictive ability in broader contexts.

**Reply 4**: In the "Limitations" subsection, add the following content regarding external validation:

"Third, this study only utilized an internal validation set for model evaluation without conducting external validation. We plan to actively seek appropriate external datasets for external validation. By comparing the model's performance metrics across diverse datasets, we aim to further refine the model evaluation framework."

**Changes in the text: Line 310-314**
##### **Comment 5**

**意见 5**：尽管研究提出了潜在的临床应用，但未提供该模型如何整合到临床实践或其对患者结果的影响的证据。未来的研究应专注于在现实环境中实施该模型，以评估其有效性。

**回复 5**：尽管该模型具有临床应用潜力，但缺乏具体的临床应用证据。我们的未来研究将专注于在实际临床环境中实施该模型，以观察其对临床决策的影响。将在讨论部分增加关于模型临床意义的讨论，如下所述：

被归为低风险的患者将继续接受标准的围手术期护理方案，并进行常规监测。中风险患者应加强肾功能监测，尤其是避免使用肾毒性药物。高风险病例将触发多学科团队会诊，以制定个体化护理计划。这些计划将整合术前手术优化、术中肾脏保护策略以及术后早期肾脏替代治疗的应急计划，从而能够更早地实施有效的干预策略，以降低CABG手术后AKI的发生率和术后死亡率。

**文本更改**：第285-293行

**Comment 5**: While the study suggests potential clinical applications, it does not provide evidence of how the model would be integrated into clinical practice or its impact on patient outcomes. Future studies should focus on implementing the model in real-world settings to assess its effectiveness.

**Reply 5**: Although the model has clinical application potential, it lacks specific clinical application evidence. Our future research will focus on implementing this model in actual clinical settings to observe its impact on clinical decision-making. Discussion on the clinical significance of the model will be added in the discussion section, as follows:

Patients classified as low-risk would continue standard perioperative care regimen with routine monitoring. Moderate-risk patients should strengthen renal function monitoring, particularly to avoid the use of nephrotoxic drugs. High-risk cases would trigger multidisciplinary team consultations to develop individualized care plans. These plans would integrate preoperative surgical optimization, intraoperative renal protective strategies, and contingency plans for early postoperative renal replacement therapy, allowing for the earlier implementation of effective intervention strategies to reduce the incidence of AKI and postoperative mortality after CABG surgery.

**Changes in the text: Line 285-293**
##### **Comment 6**

**意见 6**：研究采用了多种机器学习算法，允许对它们的预测性能进行全面比较。这种方法有助于识别在CABG手术背景下预测AKI的最有效模型。

**回复 6**：我们感谢您对我们对多种机器学习算法进行全面比较的认可。这种方法使我们能够识别出预测CABG术后AKI的最优模型，为后续研究和临床转化提供了有力支持。我们将继续探索其他新兴算法，以进一步优化模型性能。

**文本更改**：无

**Comment 6**: The study employs a diverse range of machine learning algorithms, allowing for a comprehensive comparison of their predictive performances. This approach helps identify the most effective model for AKI prediction in the context of CABG surgery.

**Reply 6**: We appreciate your recognition of our comprehensive comparison of multiple machine learning algorithms. This approach enables the identification of the optimal model for predicting AKI following CABG, providing a robust support for subsequent research and clinical translation. We will continue to explore other emerging algorithms to further optimize model performance.

**Changes in the text: None**
##### **Comment 7**

**意见 7**：在测试的算法中，随机森林模型表现出最佳的预测性能，AUC为0.737。这表明其在临床环境中识别高风险患者的潜在应用价值。

**回复 7**：在本研究中，随机森林模型表现出色，AUC为0.737，显示出在识别高风险患者方面的潜在临床价值。我们的后续研究将进一步通过优化参数配置来增强预测准确性，并整合其他临床变量以研究将该模型转化为可操作的临床应用的有效策略。

**文本更改**：无

**Comment 7**: The Random Forest model shows the best predictive performance among the tested algorithms, with an AUC of 0.737. This suggests its potential utility in clinical settings for identifying high-risk patients.

**Reply 7**: The random forest model demonstrated strong performance in this study with an AUC of 0.737 and showed potential clinical value in identifying high-risk patients. Our subsequent research will further explore the model's strengths through optimized parameter configurations to enhance predictive accuracy. Additionally, we will integrate other clinical variables to investigate effective strategies for translating this model into actionable clinical applications.

---
