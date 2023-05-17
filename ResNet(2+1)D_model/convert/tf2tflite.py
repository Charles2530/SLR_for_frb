import tensorflow as tf

TF_PATH = "F:/中国手语数据集科大/SLR/tf_model slr_cnn3d_epoch005"
TFLITE_PATH = "slr_cnn3d_epoch005.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tf_lite_model = converter.convert()
with open(TFLITE_PATH, 'wb') as f:
    f.write(tf_lite_model)