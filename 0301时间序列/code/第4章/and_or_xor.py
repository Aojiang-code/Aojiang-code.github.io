import numpy as np


def sigmoid(x):
    # 返回sigmoid函数的计算结果
    return 1 / (1 + np.exp(-x))


def judge(v):
    # 如果v小于0.5返回0，否则返回1
    return 0 if v < 0.5 else 1


# AND 计算
def and_function(num1, num2):
    # 计算sigmoid(num1 + num2 - 1.5)并判断结果
    return judge(sigmoid(num1 + num2 - 1.5))


# OR 计算
def or_function(num1, num2):
    # 计算sigmoid(num1 + num2 - 0.5)并判断结果
    return judge(sigmoid(num1 + num2 - 0.5))


# XOR 计算
def xor_function(num1, num2):
    # 计算两个中间值
    v1 = sigmoid(num1 + num2 - 1.5)
    v2 = sigmoid(num1 + num2 - 0.5)
    # 计算sigmoid(-v1 + v2 - 0.2)并判断结果
    return judge(sigmoid(- v1 + v2 - 0.2))


# 定义输入数据
data = [[0, 0], [0, 1], [1, 0], [1, 1]]

# 遍历每组输入数据
for ele in data:
    # 计算AND、OR、XOR的结果
    and_label = and_function(ele[0], ele[1])
    or_label = or_function(ele[0], ele[1])
    xor_label = xor_function(ele[0], ele[1])

    # 打印输入和计算结果
    print('输入元素:', ele[0], ele[1], '  ', 'and:', and_label,
          '  ', 'or:', or_label, '  ', 'xor:', xor_label)






# 在这段代码中，定义了三个函数 `and_function`、`or_function` 和 `xor_function`，分别用于计算逻辑与、或、异或操作。通过使用 `sigmoid` 函数和 `judge` 函数，这些逻辑操作被模拟为神经网络的激活函数。代码最后遍历输入数据，计算并打印每组输入的逻辑操作结果。