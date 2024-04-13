import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('./TMP/saved_model') # path to the SavedModel directory
tflite_model = converter.convert()
# Save the model.
with open('./Lite/tfl.tflite', 'wb') as f:
  f.write(tflite_model)
f.close()