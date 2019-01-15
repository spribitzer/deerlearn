import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

print(tf.VERSION)
print(tf.keras.__version__)

model = tf.keras.Sequential()
model.add(layers.Dense(256, activation=tf.nn.tanh)) #hyperbolic tangent activation
model.add(layers.Dense(256, activation=tf.nn.tanh)) #hyperbolic tangent activation
model.add(layers.Dense(256, activation=tf.nn.tanh)) #hyperbolic tangent activation
model.add(layers.Dense(256, activation=tf.nn.sigmoid)) #logistic sigmoid

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])

data = np.random.random((1000, 256))
labels = np.random.random((1000, 256))

model.fit(data, labels, epochs=10, batch_size=32)