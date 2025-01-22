import math
import numpy as np
import matplotlib.pyplot as plt


# 构建位置编码的序列向量矩阵
def bulid_matrixs(vals):
    matrixs = []
    for val in vals:
        matrix = []
        for pos in range(256):
            row = []
            for i in range(int(512 / 2)):
                row.append(math.sin((pos + 1) / val ** (2 * i / 512)))
                row.append(math.cos((pos + 1) / val ** (2 * i / 512)))

            matrix.append(row)

        matrixs.append(np.array(matrix))

    return matrixs


# 从序列中采样元素, 计算两两元素向量之间的余弦相似性
def compute_corr(matrixs, step):
    corrs = []
    for matrix in matrixs:
        corr_matrix = np.corrcoef(matrix[::step])
        corr_vals = []
        for i in range(len(corr_matrix) - 1):
            corr_vals.append(round(corr_matrix[i][i + 1], 2))

        corrs.append(corr_vals)

    return corrs


# 显示位置编码矩阵
def show(vals, matrixs):
    for i in range(len(vals)):
        val = vals[i]
        matrix = matrixs[i]
        ax = plt.matshow(matrix, fignum=0)
        plt.colorbar(ax.colorbar, fraction=0.025)
        plt.title("Val: " + str(val))
        plt.show()


# 比如step=50, 采样序列中位置0, 50, 100, 150...的元素
step = 50

# 公式中底数的取值, 采用多个底数做对比测试
vals = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]

matrixs = bulid_matrixs(vals)
corrs = compute_corr(matrixs, step)

print('step={0}'.format(step))

for i in range(len(vals)):
    val = vals[i]
    corr = corrs[i]

    print('val={0}'.format(val), corr)

show(vals, matrixs)
