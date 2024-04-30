import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint


def build_model():
    inputdata = keras.Input(shape=(199, 39, 1))

    final = keras.layers.Conv2D(32, (3, 3), padding="valid", kernel_initializer='random_uniform')(inputdata)
    final = keras.layers.PReLU()(final)
    final = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(final)
    final = keras.layers.BatchNormalization()(final)

    final = keras.layers.Conv2D(32, (3, 3), padding="valid")(final)
    final = keras.layers.PReLU()(final)
    final = keras.layers.MaxPooling2D((4, 4))(final)
    final = keras.layers.BatchNormalization()(final)

    final = keras.layers.Conv2D(64, (3, 3), padding="valid")(final)
    final = keras.layers.PReLU()(final)
    final = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(final)
    final = keras.layers.BatchNormalization()(final)

    final = keras.layers.Reshape((11, 64))(final)
    final = keras.layers.GRU(64)(final)
    final = keras.layers.Flatten()(final)
    final = keras.layers.Dropout(0.5)(final)
    final = keras.layers.Dense(32)(final)
    final = keras.layers.Dense(3)(final)
    final = keras.layers.Softmax()(final)

    model = keras.Model(inputs=inputdata, outputs=final)
    optimizer = keras.optimizers.Adam(
        lr=0.001, decay=1e-6, epsilon=None)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print(model.summary())
    return model