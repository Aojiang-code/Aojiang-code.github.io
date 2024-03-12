# Voice2Series-重编程

## Voice2Series: 对抗性重编程声学模型以进行时间序列分类

### 论文 | Colab 演示 | 视频 | 幻灯片

我们提供了一种端到端的方法（重编程层）来在原始波形上对时间序列数据进行重编程，使用来自 kapre 的微分梅尔频谱图层。

无需离线声学特征提取，所有层都是可微分的。

Pytorch 版本的重编程层可以在 ICASSP 23 音乐重编程中找到。

更新：如果您在这段代码中使用了 ECG 200 数据集，请执行 git pull 并参考问题中报告的标签加载错误。（已修复）

## 环境

Tensorflow 2.2（CUDA=10.0）和 Kapre 0.2.0。

PyTorch 注意：鉴于社区的许多兴趣，我们还将提供 Pytorch V2S 层和框架，整合新的 torch audio 层。有兴趣进一步重编程合作的，请通过电子邮件联系作者。

### 选项 1（来自 yml）

shell
```bash
conda env create -f V2S.yml
```

### 选项 2（来自干净的 Python 3.6）

shell
```bash
pip install tensorflow-gpu==2.1.0
pip install kapre==0.2.0
pip install h5py==2.10.0
pip install pyts
```

## 训练

### 随机映射

请同时查看论文以获取实际验证细节。非常感谢！

python
```python
python v2s_main.py --dataset 0 --eps 20 --mod 2 --seg 18 --mapping 1
```

### 结果

```
Epoch 14/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.4493 - accuracy: 0.9239 - val_loss: 0.4571 - val_accuracy: 0.9106
Epoch 15/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.4297 - accuracy: 0.9306 - val_loss: 0.4381 - val_accuracy: 0.9265
Epoch 16/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.4182 - accuracy: 0.9247 - val_loss: 0.4204 - val_accuracy: 0.9205
Epoch 17/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.3972 - accuracy: 0.9320 - val_loss: 0.4072 - val_accuracy: 0.9242
Epoch 18/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.3905 - accuracy: 0.9303 - val_loss: 0.4099 - val_accuracy: 0.9242
Epoch 19/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.3765 - accuracy: 0.9320 - val_loss: 0.3924 - val_accuracy: 0.9258
Epoch 20/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.3704 - accuracy: 0.9300 - val_loss: 0.3816 - val_accuracy: 0.9250
--- Train loss: 0.36046191089949786
- Train accuracy: 0.93113023
--- Test loss: 0.38329164963780027
- Test accuracy: 0.925
=== Best Val. Acc:  0.92651516  At Epoch of  14
```

### 多对一标签映射

python
```python
python v2s_main.py --dataset 0 --eps 20 --mod 2 --seg 18 --mapping 18
```

### 结果

shell
```
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.8762 - accuracy: 0.9231 - val_loss: 0.8479 - val_accuracy: 0.9182
Epoch 12/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.8360 - accuracy: 0.9236 - val_loss: 0.8191 - val_accuracy: 0.9152
Epoch 13/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.7920 - accuracy: 0.9242 - val_loss: 0.7693 - val_accuracy: 0.9273
Epoch 14/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.7586 - accuracy: 0.9228 - val_loss: 0.7358 - val_accuracy: 0.9235
Epoch 15/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.7265 - accuracy: 0.9270 - val_loss: 0.7076 - val_accuracy: 0.9205
Epoch 16/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.6980 - accuracy: 0.9247 - val_loss: 0.6707 - val_accuracy: 0.9295
Epoch 17/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.6650 - accuracy: 0.9281 - val_loss: 0.6473 - val_accuracy: 0.9250
Epoch 18/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.6444 - accuracy: 0.9286 - val_loss: 0.6270 - val_accuracy: 0.9303
Epoch 19/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.6194 - accuracy: 0.9286 - val_loss: 0.6020 - val_accuracy: 0.9318
Epoch 20/20
3601/3601 [==============================] - 4s 1ms/sample - loss: 0.5964 - accuracy: 0.9275 - val_loss: 0.5813 - val_accuracy: 0.9227
--- Train loss: 0.5795955053139845
- Train accuracy: 0.93113023
--- Test loss: 0.5856682072986256
- Test accuracy: 0.92651516
=== Best Val. Acc:  0.9318182  At Epoch of  18
```

### 类激活映射

python
```python
python cam_v2s.py --dataset 5 --weight wNo5_map6-88-0.7662.h5 --mapping 6 --layer conv2d_1
```

## 理论讨论

对于切片Wasserstein距离映射和理论分析，我们使用 POT 包（JMLR 2021）。

通过重编程 K 路源神经网络分类器的目标任务的总体风险被上面的方程所上界。

## 常见问题解答

### 调整模型的技巧？

目标序列的掩蔽重要吗？

V2S 掩蔽被提供为一个选项，但训练脚本没有在前向传递中使用掩蔽。根据我们的实验，使用或不使用掩蔽只对性能有小的变化。这与提出的关于学习目标领域适应的理论分析并不冲突。

### 我们可以使用 Voice2Series 用于其他领域或与团队合作吗？

是的，欢迎。请通过电子邮件联系作者以探讨潜在的合作。

## 预训练模型和训练

### VGGish AudioSet

```bash
cd weight
pip install gdown
gdown https://drive.google.com/uc?id=1mhqXZ8CANgHyepum7N4yrjiyIg6qaMe6
```

## 额外问题

请在这里打开一个问题进行讨论。谢谢！
