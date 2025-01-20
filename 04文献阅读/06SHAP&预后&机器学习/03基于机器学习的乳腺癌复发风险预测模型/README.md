# 基于机器学习的乳腺癌复发风险预测模型
Machine learning-based models for the prediction of breast cancer recurrence risk

## 思路
这篇文献的总体思路是利用机器学习（ML）方法开发一个预测乳腺癌复发风险的模型，并通过SHapley Additive exPlanation（SHAP）方法解释模型，以提高临床可解释性和实用性。以下是研究的总体思路和具体方法：
### 总体思路
#### 背景与目标：
乳腺癌是女性中最常见的恶性肿瘤之一，复发风险高，尤其是激素受体阳性（HR+）乳腺癌患者。
研究目标是开发一个基于机器学习的预测模型，利用临床信息和常规实验室指标预测乳腺癌复发风险，为临床决策提供支持。
#### 数据来源与处理：
数据来自2011年至2018年天津医科大学肿瘤研究所和医院的342例乳腺癌患者。
纳入标准包括病理确诊的乳腺癌患者，排除标准包括合并其他疾病或既往乳腺肿瘤切除的患者。
临床特征包括病理信息、肿瘤大小、淋巴结分期、治疗策略等，共25个特征。
#### 模型开发与比较：
使用11种机器学习算法（包括AdaBoost、XGBoost、随机森林、逻辑回归等）开发预测模型。
通过AUC、准确率、敏感性、特异性、PPV、NPV和F1分数评估模型性能。
使用SHAP方法解释最佳模型（AdaBoost），并分析特征重要性。
#### 模型解释与临床应用：
通过SHAP值分析特征对预测结果的贡献，识别最重要的特征（如CA125、CEA、纤维蛋白原和肿瘤直径）。
使用决策曲线分析（DCA）评估模型的临床价值，确定模型在不同阈值概率下的净效益。
#### 结论：
AdaBoost模型在预测乳腺癌复发方面表现最佳，AUC为0.987，准确率为97.1%。
SHAP方法揭示了模型中最重要的特征，为临床医生提供了可解释的预测结果。
该模型可以作为临床决策支持工具，帮助识别高复发风险的乳腺癌患者。
### 具体方法
#### 数据收集与预处理：
收集患者的临床特征、实验室指标和病理信息。
将数据分为训练集（70%）和测试集（30%）。
对缺失数据采用随机森林和多重插补方法处理。
#### 机器学习模型开发：
使用11种机器学习算法开发预测模型：
- AdaBoost：通过组合多个弱学习器提升模型性能。
- XGBoost、GBDT：基于梯度提升的集成学习算法。
- 随机森林：基于决策树的集成学习算法。
- 逻辑回归：线性分类模型。
- 支持向量分类（SVC）：基于支持向量机的分类算法。
- 多层感知器（MLP）：神经网络模型。
- 线性判别分析（LDA）：基于线性判别分析的分类算法。
- 高斯朴素贝叶斯（GaussianNB）：基于贝叶斯定理的分类算法。
- LightGBM：基于梯度提升的高效算法。
使用3折交叉验证评估模型性能。
#### 模型性能评估：
使用AUC、准确率、敏感性、特异性、PPV、NPV和F1分数评估模型性能。
AdaBoost模型表现最佳，AUC为0.987，准确率为97.1%，敏感性为94.7%，特异性为97.6%。
#### 模型解释与特征重要性分析：
使用SHAP方法解释AdaBoost模型，分析特征对预测结果的贡献。
识别最重要的特征：CA125、CEA、纤维蛋白原和肿瘤直径。
通过SHAP值的分布图展示特征值与预测结果的关系。
#### 临床应用评估：
使用决策曲线分析（DCA）评估模型在不同阈值概率下的净效益。
当阈值概率大于1%时，AdaBoost模型的净效益高于其他算法。
### 总结
这篇研究通过开发和解释机器学习模型，为乳腺癌复发风险的预测提供了一种新的方法。AdaBoost模型在预测乳腺癌复发方面表现出色，SHAP方法则增强了模型的可解释性，使其更适合临床应用。研究结果为临床医生提供了一个可解释的预测工具，有助于识别高复发风险的患者并优化治疗策略。


## Abstract  
Breast cancer is the most common malignancy diagnosed in women worldwide. The prevalence and incidence of breast cancer is increasing every year; therefore, early diagnosis along with suitable relapse detection is an important strategy for prognosis improvement. This study aimed to compare different machine algorithms to select the best model for predicting breast cancer recurrence. The prediction model was developed by using eleven different machine learning (ML) algorithms, including logistic regression (LR), random forest (RF), support vector classification (SVC), extreme gradient boosting (XGBoost), gradient boosting decision tree (GBDT), decision tree, multilayer perceptron (MLP), linear discriminant analysis (LDA), adaptive boosting (AdaBoost), Gaussian naive Bayes (GaussianNB), and light gradient boosting machine (LightGBM), to predict breast cancer recurrence. The area under the curve (AUC), accuracy, sensitivity, specificity, positive predictive value (PPV), negative predictive value (NPV) and F1 score were used to evaluate the performance of the prognostic model. Based on performance, the optimal ML was selected, and feature importance was ranked by Shapley Additive Explanation (SHAP) values. Compared to the other 10 algorithms, the results showed that the AdaBoost algorithm had the best prediction performance for successfully predicting breast cancer recurrence and was adopted in the establishment of the prediction model. Moreover, CA125, CEA, Fbg, and tumor diameter were found to be the most important features in our dataset to predict breast cancer recurrence. More importantly, our study is the first to use the SHAP method to improve the interpretability of clinicians to predict the recurrence model of breast cancer based on the AdaBoost algorithm. The AdaBoost algorithm offers a clinical decision support model and successfully identifies the recurrence of breast cancer.
乳腺癌是全球女性诊断出的最常见的恶性肿瘤。乳腺癌的患病率和发病率每年都在增加;因此，早期诊断和适当的复发检测是改善预后的重要策略。本研究旨在比较不同的机器算法，以选择预测乳腺癌复发的最佳模型。预测模型是使用 11 种不同的机器学习 （ML） 算法开发的，包括逻辑回归 （LR）、随机森林 （RF）、支持向量分类 （SVC）、极端梯度提升 （XGBoost）、梯度提升决策树 （GBDT）、决策树、多层感知器 （MLP）、线性判别分析 （LDA）、自适应提升 （AdaBoost）、高斯朴素贝叶斯 （GaussianNB） 和光梯度提升机 （LightGBM），以预测乳腺癌复发。曲线下面积 （AUC） 、准确性、敏感性、特异性、阳性预测值 （PPV） 、阴性预测值 （NPV） 和 F1 评分用于评估预后模型的性能。根据性能，选择最佳 ML，并通过 Shapley 加法解释 （SHAP） 值对特征重要性进行排序。与其他 10 种算法相比，结果表明 AdaBoost 算法在成功预测乳腺癌复发方面具有最佳预测性能，并被用于预测模型的建立。此外，CA125、CEA、Fbg 和肿瘤直径被发现是我们的数据集中预测乳腺癌复发的最重要特征。更重要的是，我们的研究是第一个使用 SHAP 方法提高临床医生基于 AdaBoost 算法预测乳腺癌复发模型的可解释性的研究。 AdaBoost 算法提供了一个临床决策支持模型，并成功识别了乳腺癌的复发。

## Keywords: 
Breast cancer, Machine learning, Artificial intelligence, Disease recurrence, Prediction model
关键字： 乳腺癌， 机器学习， 人工智能， 疾病复发， 预测模型
















































































































