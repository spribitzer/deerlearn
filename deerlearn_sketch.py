import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


print(tf.VERSION)
print(tf.keras.__version__)


PRraw = genfromtxt('PR.csv', delimiter=',')
TD = genfromtxt('TD.csv', delimiter=',')



[nTraces,nTimePoints] = TD.shape
[df,nRPoints] = PRraw.shape

PR = np.zeros(PRraw.shape)

for Sample in range(nTraces):
    PofR = PRraw[Sample,:]
    PR[Sample,:] = PofR/PofR.max()

frac = 0.8
ind = int(np.floor(nTraces*frac))

learnTD = TD[:ind,:]
learnPR = PR[:ind,:]

evaluateTD = TD[ind+1:,:]
evaluatePR = PR[ind+1:,:]


model = tf.keras.Sequential()
model.add(layers.Dense(256, activation=tf.nn.tanh, input_dim = nTimePoints))  #hyperbolic tangent activation
model.add(layers.Dense(256, activation=tf.nn.tanh)) #hyperbolic tangent activation
model.add(layers.Dense(256, activation=tf.nn.tanh)) #hyperbolic tangent activation
model.add(layers.Dense(256, activation=tf.nn.sigmoid)) #logistic sigmoid

# #  how to use time axis and r axis for the labels

model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])


model.fit(learnTD, learnPR, epochs=20)

model.evaluate(evaluateTD, evaluatePR, batch_size=nTraces-ind-1)

model.summary()

sample = -2

example_batch = evaluateTD[-3:-1,:]
example_batch.shape
example_result = model.predict(example_batch)

plt.plot(example_result[0,:],'r-',evaluatePR[-3,:],'b--')
plt.show()