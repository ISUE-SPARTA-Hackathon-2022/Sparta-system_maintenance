import tensorflow as tf
import helper as helper
import os

_helper = helper.helper()

execution_path = os.getcwd()
try:
  model = tf.keras.models.load_model('./static/Model 2_2022_10_31 - 12.h5')

  predictions, probabilities = model.predict()
except Exception as e:
  _helper.notifications.error()