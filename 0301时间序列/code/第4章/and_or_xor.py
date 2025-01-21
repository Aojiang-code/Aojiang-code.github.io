import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def judge(v):
    return 0 if v < 0.5 else 1


# AND 计算
def and_function(num1, num2):
    return judge(sigmoid(num1 + num2 - 1.5))


# OR 计算
def or_function(num1, num2):
    return judge(sigmoid(num1 + num2 - 0.5))


# XOR 计算
def xor_function(num1, num2):
    v1 = sigmoid(num1 + num2 - 1.5)
    v2 = sigmoid(num1 + num2 - 0.5)
    return judge(sigmoid(- v1 + v2 - 0.2))


data = [[0, 0], [0, 1], [1, 0], [1, 1]]

for ele in data:
    and_label = and_function(ele[0], ele[1])
    or_label = or_function(ele[0], ele[1])
    xor_label = xor_function(ele[0], ele[1])

    print('输入元素:', ele[0], ele[1], '  ', 'and:', and_label,
          '  ', 'or:', or_label, '  ', 'xor:', xor_label)
