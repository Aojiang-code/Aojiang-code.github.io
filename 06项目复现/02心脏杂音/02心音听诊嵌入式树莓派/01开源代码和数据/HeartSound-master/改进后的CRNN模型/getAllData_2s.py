import numpy as np
import librosa
from python_speech_features import mfcc
from scipy.signal import butter, lfilter
import os
import csv
from Augmentation import add_noise1, add_noise2, time_shift, time_stretch


def load_csv(filename):
    file=open(filename,'r')
    dataset=[]
    csv_reader=csv.reader(file)
    for line in csv_reader:
        if not line:
            continue
        dataset.append(line)
    return dataset


def butterworth_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs##使用的都是实际频率的1/2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butterworth_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butterworth_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)##将滤波器应用于信号
    return y


def read_Physio(letter):  # 读取Physio每个文件夹中的数据和label
    Physio_all = []
    dict={'a':409,'b':490,'c':31,'d':55,'e':2141,'f':114}
    path_o='./Physio/training-'+letter+'/'
    label=[]
    csvpath = './Physio/training-' + letter + '/REFERENCE.csv'
    data = load_csv(csvpath)
    if letter=='e':
        indic='0'
    else:
        indic=''
    for i in range(dict[letter]):
        if int(data[i][1])==1:
            labelx=0
        elif int(data[i][1])==-1:
            labelx=1
        if i<9:
            filepath=path_o+letter+indic+'000'+str(i+1)+'.wav'
        elif i>=9 and i<99:
            filepath=path_o+letter+indic+'00'+str(i+1)+'.wav'
        elif i>=99 and i<999:
            filepath=path_o+letter+indic+'0'+str(i+1)+'.wav'
        else:
            filepath=path_o+letter+indic+str(i+1)+'.wav'
        signal,f=librosa.load(filepath, sr=None)
        butterworth_bandpass_filter(signal, 25, 400, f, 5)  # butterworth滤波
        iii=1
        while iii*4000<len(signal):
            phy=signal[(iii-1)*4000:iii*4000]  # 保留信号的2秒切片（注意频率是2k）
            Physio_all.append(phy)
            label.append(labelx)
            iii+=1
    return label, Physio_all


def readDataset_Physio():
    all_sig=[]
    all_labe=[]
    for cha in ['a','b','c','d','e','f']:
        print(cha)
        all_sig.append(read_Physio(cha)[1])
        all_labe.append(read_Physio(cha)[0])
    all_signal=sum(all_sig,[])  # 这个数组里面是PhysioNet所有数据。
    all_label=sum(all_labe,[])
    return all_label, all_signal


def readDataset_Five():  # 每一类有500条数据
    classes=['AS','MR','MS','MVP','N']
    all_label=[]
    all_signal=[]
    # 分正常、不正常两类版本
    for ch in classes:
        if ch=='N':
            label=1
        else:
            label=0
        folderpath = './five_class/' + ch
        for i in range(1, 201):
            if i >= 1 and i <= 9:
                num = '00' + str(i)
            elif i >= 10 and i <= 99:
                num = '0' + str(i)
            elif i >= 100:
                num = str(i)
            filepath = folderpath + '/New_' + ch + '_' + num + '.wav'
            signal, fre = librosa.load(filepath, sr=None)
            butterworth_bandpass_filter(signal, 25, 400, fre, 5)
            signal_2k = librosa.resample(signal, fre, 2000)
            signal_5s = np.hstack((signal_2k, signal_2k,signal_2k,signal_2k))[0:4000]
            all_label.append(label)
            all_signal.append(signal_5s)
    '''
    # 分5类版本
    for label, ch in enumerate(classes):
        folderpath='./five_class/'+ch
        for i in range(1,201):
            if i>=1 and i<=9:
                num='00'+str(i)
            elif i>=10 and i<=99:
                num='0'+str(i)
            elif i>=100:
                num=str(i)
            filepath=folderpath+'/New_'+ch+'_'+num+'.wav'
            signal,fre=librosa.load(filepath,sr=None)
            butterworth_bandpass_filter(signal, 25, 400, f, 5)
            signal_2k=librosa.resample(signal,fre,2000)
            signal_5s=np.hstack((signal_2k,signal_2k))[0:10000]
            all_label.append(label)
            all_signal.append(signal_5s)
    '''
    return all_label, all_signal


def getMFCCMap(y, sr=2000):
    mfcc0 = mfcc(y, sr, numcep=13)
    mf1 = librosa.feature.delta(mfcc0)
    mf2 = librosa.feature.delta(mfcc0, order=2)
    mfcc_all=np.hstack((mfcc0,mf1,mf2))
    mfcc_all1=mfcc_all.reshape(199,39,1)
    return mfcc_all1


def getMFCC0(y, sr):
    mfcc0 = mfcc(y, sr, numcep=13)
    return mfcc0


def readDataset_Kaggle():
    all_label = []
    all_signal = []
    datapath='./kaggle/set_b/'
    xml_list = [os.path.join(datapath, i) for i in os.listdir(datapath) if os.path.splitext(i)[-1] == '.wav'] # kaggle/set_b下所有文件的完整路径
    for path in xml_list:
        sig,f=librosa.load(path,sr=None)
        butterworth_bandpass_filter(sig, 25, 400, f, 5)
        signal_2k = librosa.resample(sig, f, 2000)
        signal_5s=np.hstack((signal_2k, signal_2k, signal_2k, signal_2k, signal_2k, signal_2k, signal_2k, signal_2k,))[0:4000]
        endpath=path.split('/')[-1]
        if endpath[:6] == 'murmur' and endpath[7:12] != 'noisy':
            all_label.append(0)
            all_signal.append(signal_5s)
        elif endpath[:6] == 'normal' and endpath[7:12] != 'noisy':
            all_label.append(1)
            all_signal.append(signal_5s)
        elif endpath[:8]=='artifact':
            # 由于噪声样本太少，对它信号增强
            all_signal.append(signal_5s)
            all_signal.append(add_noise1(signal_5s,0.004))
            all_signal.append(add_noise2(signal_5s,50))
            all_signal.append(time_shift(signal_5s,f//2))
            all_signal.append(time_stretch(signal_5s,1.1))
            for _ in range(5):
                all_label.append(2)
    return all_label, all_signal


labels1,signals1=readDataset_Physio()
labels2,signals2=readDataset_Five()
labels3,signals3=readDataset_Kaggle()
maps=[]
labels=[]
for i,signal in enumerate(signals1):
    if len(signal)==4000:
        maps.append(getMFCCMap(signal, 2000))
        labels.append(labels1[i])
for i,signal in enumerate(signals2):
    maps.append(getMFCCMap(signal, 2000))
    labels.append(labels2[i])
for i,signal in enumerate(signals3):
    if len(signal)==4000:
        maps.append(getMFCCMap(signal, 2000))
        labels.append(labels3[i])
maps=np.array(maps)
np.save('./Jan9_2s_features_huge4k.npy', maps)
np.save('./Jan9_2s_label_huge4k.npy',labels)

