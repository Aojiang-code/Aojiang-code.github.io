# 文献阅读

## [Q2](10项目申报\02心脏杂音\01文献\02PubMed中与LSTM相关的文献\02文献阅读\02Q2\README.md)

## [Q4](10项目申报\02心脏杂音\01文献\02PubMed中与LSTM相关的文献\02文献阅读\04Q4\README.md)

## [无分区](10项目申报\02心脏杂音\01文献\02PubMed中与LSTM相关的文献\02文献阅读\05无分区\README.md)


## 可获取的8篇文献总结
### 核心内容总结


1. **"Automatic heart sound classification from segmented/unsegmented phonocardiogram signals using time and frequency features" by Faiq Ahmad Khan et al.**
   - **核心内容**: 研究了使用分割和未分割的心脏声音图（PCG）信号进行自动心脏声音分类的方法，通过分析时间和频率域特征，并使用不同的分类算法进行性能评估。

2. **"Design of ear‑contactless stethoscope and improvement in the performance of deep learning based on CNN to classify the heart sound" by Tanmay Sinha Roy et al.**
   - **核心内容**: 提出了一种非接触式听诊器的设计，并研究了如何通过调整深度学习算法（特别是CNN）的超参数来提高心脏声音分类的性能。

3. **"The Effect of Signal Duration on the Classification of Heart Sounds: A Deep Learning Approach" by Xinqi Bao et al.**
   - **核心内容**: 分析了信号持续时间对使用深度学习方法进行心脏声音分类的影响，发现较短的信号持续时间可能会减弱某些网络的性能。

4. **"HBNET: A blended ensemble model for the detection of cardiovascular anomalies using phonocardiogram" by Ann Nita Netto et al.**
   - **核心内容**: 开发了一个混合集成模型（HbNet），使用混合深度学习模型和softmax回归来分类成人和儿童的心脏声音，以检测心血管异常。

5. **"Detection of Snore from OSAHS Patients Based on Deep Learning" by Fanlin Shen et al.**
   - **核心内容**: 提出了一种基于深度学习的方法，用于从阻塞性睡眠呼吸暂停低通气综合征（OSAHS）患者中检测鼾声，并通过特征提取和神经网络模型进行分类。

6. **"Parralel Recurrent Convolutional Neural Network for Abnormal Heart Sound Classification" by Arash GHAREHBAGHI et al.**
   - **核心内容**: 介绍了一种并行卷积神经网络（PCNN）用于异常心脏声音分类，并比较了其性能与串联卷积神经网络（SCNN）和其他基线方法。

7. **"Recurrent vs Non-Recurrent Convolutional Neural Networks for Heart Sound Classification" by Arash GHAREHBAGHI et al.**
   - **核心内容**: 比较了传统CNN与结合了递归神经网络（如GRN和LSTM）的CNN在心脏声音分类任务中的性能，探讨了不同架构的优缺点。

8. **"Segmentation of Radar-Recorded Heart Sound Signals Using Bidirectional LSTM" by Kilin Shi et al.**
   - **核心内容**: 研究了使用双向长短期记忆（biLSTM）网络对雷达记录的心脏声音信号进行分割的方法，以实现自动化分类和分析。

### 文献比较
上述文献主要集中在心脏声音信号的自动分类、异常检测以及深度学习在心脏声音分析中的应用。以下是对这些文献的比较：

1. **研究焦点**:
   - Khan等人专注于使用分割和未分割的心脏声音图（PCG）信号进行自动分类。
   - Roy等人设计了一种非接触式听诊器，并通过深度学习算法对心脏声音进行分类。
   - Bao等人研究了信号持续时间对心脏声音分类性能的影响。
   - Netto等人开发了一种混合集成模型（HbNet），用于检测心血管异常。
   - Shen等人提出了一种基于深度学习的方法，用于从OSAHS患者中检测鼾声。
   - Gharehbaghi等人探索了并行和非并行卷积神经网络在心脏声音分类中的表现。
   - Shi等人使用双向LSTM网络对雷达记录的心脏声音信号进行分割。

2. **使用的方法和技术**:
   - 多数研究使用了深度学习算法，特别是卷积神经网络（CNN）和长短期记忆网络（LSTM）。
   - Khan等人和Gharehbaghi等人比较了不同的分类算法，包括SVM和kNN。
   - Netto等人提出了一个结合CNN-BiLSTM和CNN-LSTM的混合模型，并使用softmax回归作为元学习器。
   - Shen等人使用了MFCC、LPCC和LPMFCC三种特征提取方法，并采用了CNN和LSTM进行分类。
   - Gharehbaghi等人比较了传统CNN与结合了GRN和LSTM的CNN架构。
   - Shi等人使用了雷达技术记录心脏声音，并应用了双向LSTM网络进行信号分割。

3. **数据集和特征**:
   - 多数研究使用了公开的心脏声音数据集，如Physionet挑战数据集。
   - 特征提取方法包括MFCC、LPCC、LPMFCC以及Mel频率特征，这些特征被用于训练和优化深度学习模型。

4. **性能评估**:
   - 研究通常报告了准确率、敏感性、特异性、F1分数和其他相关指标来评估模型性能。
   - Netto等人的HbNet模型在成人数据集上显示出99.72%的平均准确率，在儿童数据集上为86.5%。
   - Shen等人的模型在二分类鼾声数据时准确率最高，达到87%。

5. **创新点和贡献**:
   - Roy等人的工作在COVID-19背景下提供了一种无接触的心脏声音听诊方法。
   - Netto等人的研究在早期检测心脏疾病方面具有潜在的临床应用价值，并为医疗从业者提供了决策支持系统。
   - Gharehbaghi等人的研究提供了对CNN和递归神经网络结合使用的深入理解，并探讨了不同架构的性能。

6. **应用和潜在影响**:
   - 这些研究在心血管疾病的早期诊断、监测和分类方面具有潜在的临床应用。
   - 非接触式听诊器和自动分类算法可能对远程医疗和个人健康监测产生积极影响。

7. **局限性和未来工作**:
   - 一些研究提到了数据集的局限性，如数据不平衡和需要更大样本量的多样性。
   - 未来的工作可能包括改进模型的泛化能力、减少计算复杂性、以及在更广泛的患者群体上验证模型性能。

综上所述，这些文献展示了心脏声音分析领域中深度学习技术的应用和发展，以及在提高心脏疾病诊断准确性方面的潜力。

### 方法介绍
> 按文献标题，一一总结这些文献的核心内容，此外重点介绍每篇文献使用的方法

以下是根据您提供的文献标题，对每篇文献的核心内容及使用的方法进行的总结：

1. **"Automatic heart sound classification from segmented/unsegmented phonocardiogram signals using time and frequency features" by Faiq Ahmad Khan et al.**
   - **核心内容**: 该研究提出了一种自动心脏声音分类方法，使用分割和未分割的心脏声音图（PCG）信号，并结合了时间和频率特征。研究比较了不同的分类算法，包括SVM、kNN、决策树、集成分类器、ANN和LSTM网络。
   - **使用的方法**: 使用了多种机器学习和深度学习算法，并对Mel Frequency Cepstral Coefficients (MFCC) 特征和其他时间-频率特征进行了分类性能比较。

2. **"Design of ear‑contactless stethoscope and improvement in the performance of deep learning based on CNN to classify the heart sound" by Tanmay Sinha Roy et al.**
   - **核心内容**: 论文设计了一种非接触式听诊器，并通过深度学习算法（特别是CNN）的超参数调整来提高心脏声音分类的性能。研究目的是在COVID-19期间提供一种无需直接接触的心脏声音听诊方法。
   - **使用的方法**: 利用CNN和RNN对心脏声音进行分类，并通过调整学习率、dropout率和隐藏层等超参数来优化模型性能。

3. **"The Effect of Signal Duration on the Classification of Heart Sounds: A Deep Learning Approach" by Xinqi Bao et al.**
   - **核心内容**: 该研究分析了信号持续时间对使用深度学习方法进行心脏声音分类的影响。研究结果表明，较短的心脏声音信号持续时间会减弱RNN的性能，而CNN模型的性能没有明显下降。
   - **使用的方法**: 采用了CNN和RNN（包括LSTM、BiLSTM、GRU和BiGRU）作为分类模型，并考虑了使用Mel-frequency cepstrum coefficients (MFCCs) 作为特征。

4. **"HBNET: A blended ensemble model for the detection of cardiovascular anomalies using phonocardiogram" by Ann Nita Netto et al.**
   - **核心内容**: 该研究旨在开发一种新型混合集成模型（HbNet），使用混合深度学习模型和softmax回归来分类成人和儿童的心脏声音。研究还包括创建一个全面的5类儿童心脏声音数据集。
   - **使用的方法**: 提出了一个名为HbNet的混合集成模型，该模型结合了CNN-BiLSTM和CNN-LSTM作为基模型，以及softmax回归作为元学习器，使用Mel Frequency Cepstral Coefficients (MFCC) 捕捉与分类相关的音频信号特征。

5. **"Detection of Snore from OSAHS Patients Based on Deep Learning" by Fanlin Shen et al.**
   - **核心内容**: 该研究提出了一种基于深度学习的方法，用于从阻塞性睡眠呼吸暂停低通气综合征（OSAHS）患者中检测鼾声。研究中使用深度学习对与呼吸暂停事件相关的鼾声和非呼吸暂停事件相关的鼾声进行分类，并识别OSAHS症状的严重程度。
   - **使用的方法**: 通过三种特征提取方法（MFCC、LPCC和LPMFCC）提取鼾声数据特征，并采用CNN和LSTM进行分类。

6. **"Parralel Recurrent Convolutional Neural Network for Abnormal Heart Sound Classification" by Arash GHAREHBAGHI et al.**
   - **核心内容**: 论文介绍了一种并行卷积神经网络（PCNN）用于从心脏声音信号中检测心脏异常的研究结果。PCNN通过并行组合递归神经网络和CNN来保留信号的动态内容，并与串联形式的CNN（SCNN）以及其他基线研究进行了性能比较。
   - **使用的方法**: PCNN结合了CNN和LSTM，并通过使用公开的心脏声音信号数据集Physionet进行了性能评估和比较。

7. **"Recurrent vs Non-Recurrent Convolutional Neural Networks for Heart Sound Classification" by Arash GHAREHBAGHI et al.**
   - **核心内容**: 该研究比较了传统的CNN与结合了不同架构的递归神经网络（如GRN和LSTM）的CNN在心脏声音分类任务中的性能。研究发现，尽管并行LSTM-CNN架构的准确率达到了98.0%，但传统CNN在复杂性较低的情况下提供了适当的性能。
   - **使用的方法**: 研究考虑了与CNN并行和串联集成的GRN和LSTM的不同组合，并使用Physionet心脏声音数据集来评估每种集成的准确性和敏感性。

8. **"Segmentation of Radar-Recorded Heart Sound Signals Using Bidirectional LSTM" by Kilin Shi et al.**
   - **核心内容**: 该研究探讨了使用不同的长短期记忆（LSTM）架构对雷达记录的心脏声音信号进行分割的效果。研究目的是实现自动化分类，其中第一步是将心脏声音信号分割成生理阶段。双向LSTM网络实现了93.4%的样本准确率和95.8%的第一次心脏声音的F1分数。
   - **使用的方法**: 使用了基于大型雷达记录心脏声音数据集的LSTM架构，特别是双向LSTM（biLSTM），以实现无需任何预定义参数的心脏声音分割。












