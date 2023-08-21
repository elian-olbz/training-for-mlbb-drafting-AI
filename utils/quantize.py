import tensorflow as tf
import numpy as np

# Load the saved model
saved_model_h5_path = 'C:/Users/Marlon/Desktop/dataset/models/meta/meta_ld_512_x5h.h5'
loaded_model = tf.keras.models.load_model(saved_model_h5_path)

# Apply quantization to the model
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False  # Disable lowering tensor list ops
quantized_tflite_model = converter.convert()

# Save the quantized model
quantized_model_path = 'C:/Users/Marlon/Desktop/dataset/quantized_model/meta_ld_512_x5h.tflite'
with open(quantized_model_path, 'wb') as file:
    file.write(quantized_tflite_model)
