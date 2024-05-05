# 06Parralel_Recurrent_Convolutional_Neural_Network_for_Abnormal_Heart_Sound_Classification

## 梗概

这篇论文的核心内容是关于一种用于异常心音分类的并行循环卷积神经网络（Parallel Recurrent Convolutional Neural Network，PCNN）的研究。以下是该研究的关键点：

1. **研究目的**：开发并评估一种并行卷积神经网络（PCNN），用于从心音信号中检测心脏异常。

2. **PCNN原理**：PCNN通过结合循环神经网络（RNN）和卷积神经网络（CNN）来保留信号的动态内容。

3. **性能评估**：PCNN的性能与串联形式的卷积神经网络（Serial Convolutional Neural Network，SCNN）、长短期记忆网络（Long- and Short-Term Memory，LSTM）和常规CNN（Conventional CNN，CCNN）进行了比较。

4. **数据集**：使用了知名的公共心音信号数据集，即Physionet心音数据集。

5. **分类方法**：PCNN将输入信号首先分为5秒的非重叠段，并使用SMOTE方法来克服训练数据中正常信号和异常信号之间的类别不平衡问题。

6. **网络结构**：PCNN和CCNN都独立地使用Adam优化器进行训练和优化，超参数设置相同。

7. **实验结果**：PCNN的准确率估计为87.2%，分别比SCNN、LSTM和CCNN高出12%、7%和0.5%。

8. **讨论**：PCNN不仅提高了常规CNN和LSTM的性能，而且与被认为是最新技术的级联组合相比，还提供了更好的准确率。然而，并行组合并没有提高灵敏度。

9. **结论**：PCNN在准确性方面显著提高了使用心音区分正常和异常心脏状况的性能，并且具有高且稳定的准确率，可以与电子听诊器集成，作为初级保健中心的决策支持系统。

10. **关键词**：心音、深度学习、并行卷积神经网络、卷积神经网络、智能听诊。

11. **作者信息**：Arash Gharehbaghi是本文的通讯作者，隶属于瑞典哈姆斯塔德大学信息技术学院。

12. **发表信息**：该文发表在《Technology and Health Care》杂志上，由IOS Press出版，并在线上以开放获取的形式发布。

13. **版权声明**：文章在Creative Commons Attribution Non-Commercial License 4.0 (CC BY-NC 4.0)条款下发布。

这篇研究展示了PCNN在心音信号分析中的潜力，特别是在提高心脏异常检测的准确性方面，为智能听诊器的开发提供了新的研究方向。






