import numpy as np
import librosa


def add_noise1(x, w=0.004):  # 加噪
    # w：噪声因子
    output = x + w * np.random.normal(loc=0, scale=1, size=len(x))
    return output


def add_noise2(x, snr=50):  # 控制信噪比
    # snr：生成的语音信噪比
    P_signal = np.sum(abs(x) ** 2) / len(x)  # 信号功率
    P_noise = P_signal / 10 ** (snr / 10.0)  # 噪声功率
    return x + np.random.randn(len(x)) * np.sqrt(P_noise)


def time_shift(x, shift):  # 波形移动
    # shift：移动的长度
    return np.roll(x, int(shift))


def time_stretch(x, rate):  # 波形拉伸
    # rate：拉伸的尺寸，
    # rate > 1 加快速度
    # rate < 1 放慢速度
    return librosa.effects.time_stretch(x, rate)