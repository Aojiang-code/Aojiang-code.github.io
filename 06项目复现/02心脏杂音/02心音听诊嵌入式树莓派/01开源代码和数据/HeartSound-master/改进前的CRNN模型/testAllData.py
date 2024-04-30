import tensorflow.keras as keras
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib.font_manager import FontProperties
#font1=FontProperties(fname='./simhei.ttf')
# myfont=FontProperties(fname='./Songti.ttc')
# sns.set(font=myfont.get_name())



def calulate_acc(label, prediction):
    num = 0
    N = len(label)
    for i in range(len(label)):
        pred = np.argmax(prediction[i])
        if label[i] ==  pred:
            num += 1
    return num / N


if __name__ == '__main__':
    acc_list = []
    test_features=np.load("./test3_features.npy")
    test_label=np.load("./test3_label.npy")
    model = keras.models.load_model('./model_combine3.h5')
    test_predictions = model.predict(test_features)
    prediction=[]
    for i in range(len(test_predictions)):
        prediction.append(np.argmax(test_predictions[i]))
    predictions=np.array(prediction)
    print(predictions)
    print(test_label)
    C2=confusion_matrix(test_label,predictions)
    print(C2)
    #sns.set(font=font1.get_name())
    f, ax = plt.subplots()
    sns.heatmap(C2, fmt='g',annot=True,cmap='YlGnBu',cbar=False, ax=ax,
                xticklabels = ('abnormal','normal','noisy'),
                yticklabels = ('abnormal','normal','noisy')
    )  # 画热力图
    ax.set_title('Confusion matrix')  # 标题
    ax.set_xlabel('True')  # x轴
    ax.set_ylabel('Prediction')  # y轴
    plt.show()
    acc = calulate_acc(test_label, test_predictions)
    acc_list.append(acc)
    for accs in acc_list:
        print('测试集准确率：',accs)