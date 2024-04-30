from CRNN_2s import build_model
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import datetime


MFCC_conca=[]
np.random.seed(0)
allfeatures=np.load("./Jan9_2s_features_huge4k.npy")
alllabels =np.load("./Jan9_2s_label_huge4k.npy")
print(allfeatures.shape)
print(alllabels.shape)

train_features, test_features, train_label, test_label = train_test_split(allfeatures, alllabels, test_size=0.2, random_state=2019)
np.save('./Jan9_2s_test_features_cpu.npy',test_features)
np.save('./Jan9_2s_test_label_cpu.npy',test_label)
num_epochs = 10
model=build_model()
print(train_features.shape)
print(train_label.shape)

log_dir= "./TFBoard2/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)  # 定义TensorBoard对象

model.fit(train_features, train_label, epochs=num_epochs, batch_size=16, verbose=1, validation_split=0.2,callbacks=[tensorboard_callback])
#model.save('./model_combine3.pb')
#模型保存
#model.fit_generator(train_features, train_label, epochs=num_epochs, batch_size=16, verbose=1, validation_split=0.2, callbacks=callbacks_list)
model.save("./TMP/J92s.h5")#保存模型权重  保存模型是model.save 注意后缀是h5
