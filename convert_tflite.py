import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("anomaly_model.h5")
mean = np.load("norm_mean.npy")
std = np.load("norm_std.npy")

def representative_dataset():
    for _ in range(100):
        sample = mean + std * np.random.randn(5)
        yield [sample.reshape(1, 5).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open("anomaly_model.tflite", "wb") as f:
    f.write(tflite_model)
print("Saved anomaly_model.tflite,", len(tflite_model), "bytes")
