import tensorflow as tf

# Load the model
model_path = ''
model = tf.saved_model.load(model_path)

# Print TensorFlow and Keras versions
print(f"TensorFlow version used to save the model: {model.tensorflow_version}")
print(f"Keras version used to save the model: {model.keras_version}")
