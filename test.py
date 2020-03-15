import numpy as np
import tensorflow as tf

const4 = tf.constant(np.arange(0, 48, 1).reshape(2, 3, 4, 2))
sum1 = tf.reduce_sum(const4, axis=0, keepdims=True)
print(const4, sum1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(1000, 64, input_length=10))
# the model will take as input an integer matrix of size (batch,
# input_length).
# the largest integer (i.e. word index) in the input should be no larger
# than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch
# dimension.

input_array = np.random.randint(1000, size=(32, 10))
print(input_array.shape)
model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)
print()
user_combine_layer_flat = tf.constant([[1],[2,2]])
movie_combine_layer_flat = tf.constant([[1],[4,4]])
inference = tf.keras.layers.Lambda(lambda layer: tf.reduce_sum(layer[0] * layer[1], axis=1), name="inference")((user_combine_layer_flat, movie_combine_layer_flat))
print(inference)
inference = tf.keras.layers.Lambda(lambda layer: tf.expand_dims(layer, axis=1))(inference)
print(inference)