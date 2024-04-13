from CRNNmodel import build_model
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import datetime


MFCC_conca=[]
np.random.seed(0)
allfeatures=np.load("./Combine3_MFCC.npy")
alllabels =np.load("./Combine3_label.npy")
print(allfeatures.shape)
print(alllabels.shape)

train_features, test_features, train_label, test_label = train_test_split(allfeatures, alllabels, test_size=0.2, random_state=2019)
np.save('./test3_features.npy',test_features)
np.save('./test3_label.npy',test_label)
num_epochs = 25
model=build_model()
print(train_features.shape)
print(train_label.shape)

log_dir= "./TFBoard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)  # 定义TensorBoard对象
history = model.fit(train_features, train_label, epochs=num_epochs, batch_size=16, verbose=1, validation_split=0.2,callbacks=[tensorboard_callback])
#model.save('./model_combine3.pb')

#定义模型检查点
checkpoint = tf.keras.callbacks.ModelCheckpoint("./TMP/md.h5", monitor='val_metric_precision', verbose=1,
                                            save_best_only=True, mode='max')
callbacks_list = [checkpoint]
#模型保存
#model.fit_generator(train_features, train_label, epochs=num_epochs, batch_size=16, verbose=1, validation_split=0.2, callbacks=callbacks_list)
model.save("./TMP/md.h5")#保存模型权重  保存模型是model.save 注意后缀是h5