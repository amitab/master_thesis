import tensorflow as tf
import numpy as np
import sys

model1_dir = sys.argv[1]
model2_dir = sys.argv[2]

model1 = tf.keras.models.load_model(model1_dir)
model2 = tf.keras.models.load_model(model2_dir)

model_diff = []
layer_sizes = []
for i in range(len(model1.layers)):
  if len(model1.layers[i].get_weights()) > 0:
    print("Layer " + str(i + 1) + ":")
    layer_diff = model1.layers[i].get_weights()[0] - model2.layers[i].get_weights()[0]
    model_diff.append(layer_diff)
    print(layer_diff)
for i in range(len(model_diff)):
  current_layer_size = 0
  total_nonzero = 0
  max = 0
  for cell in np.nditer(model_diff[i]):
    current_layer_size += 1
    if abs(cell) > 0.01:
      total_nonzero += 1
      if abs(cell) > max:
        max = cell
  percentage_diff = ((total_nonzero * 1.) / current_layer_size) * 100
  print("Amount of different weights in Layer " + str(i + 1) + ": " + str(total_nonzero)
        + " / " + str(current_layer_size) + " (" + str(percentage_diff) + "%)")
  print("Maximum Difference in Layer " + str(i+1) + ": " + str(max))
  layer_sizes.append(current_layer_size)
