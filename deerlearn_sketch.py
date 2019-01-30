import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

print(tf.VERSION)
print(tf.keras.__version__)


PRraw = genfromtxt('./Edwards_Stoll_JMR_2018_testset/PR.csv', delimiter=',')
TD = genfromtxt('./Edwards_Stoll_JMR_2018_testset/TD.csv', delimiter=',')
Rs = genfromtxt('./Edwards_Stoll_JMR_2018_testset/Rs.csv', delimiter=',')
Tref = genfromtxt('./Edwards_Stoll_JMR_2018_testset/Tref.csv', delimiter=',')
Tmax = genfromtxt('./Edwards_Stoll_JMR_2018_testset/TmaxVec.csv', delimiter=',')

def calculateRaxis(tmax, tref, nPoints = 256, rrefmin = 0.5, rrefmax = 7.2):
    stretch = (tmax/tref)**(1/3)

    rmax = rrefmax*stretch
    rmin = rrefmin*stretch

    r = np.linspace(rmin,rmax,nPoints)

    return r

[nTraces,nTimePoints] = TD.shape
[df,nRPoints] = PRraw.shape

PR = np.zeros(PRraw.shape)

for Sample in range(nTraces):
    PofR = PRraw[Sample,:]
    PR[Sample,:] = PofR/PofR.max()

frac = 0.8
ind = int(np.floor(nTraces*frac))


noise = np.random.normal(0,0.2,[nTraces, nTimePoints])
TD = TD + noise

learnTD = TD[:ind,:]
learnPR = PR[:ind,:]

evaluateTD = TD[ind+1:,:]
evaluatePR = PR[ind+1:,:]


model = tf.keras.Sequential()
model.add(layers.Dense(128, activation=tf.nn.tanh, input_dim = nTimePoints))  #hyperbolic tangent activation
model.add(layers.Dense(64, activation=tf.nn.tanh)) #hyperbolic tangent activation
model.add(layers.Dense(64, activation=tf.nn.tanh)) #hyperbolic tangent activation
model.add(layers.Dense(256, activation=tf.nn.sigmoid)) #logistic sigmoid

model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])


model.fit(learnTD, learnPR, epochs=10)

model.evaluate(evaluateTD, evaluatePR, batch_size=nTraces-ind-1)

model.summary()


xlabel = 'r [nm]'
subtitle = 'Sample %d'

for i in range(2):

    samples = np.random.randint(ind,nTraces,(4))

    example_batch = TD[samples,:]
    example_batch.shape

    example_result = model.predict(example_batch)

    computedRs = np.zeros(example_result.shape)

    sample = 0
    for j in samples:
        r = calculateRaxis(Tmax[j],Tref)
        computedRs[sample,:] = r
        sample += 1

    fig, axes = plt.subplots(2, 2,figsize=(9,7))

    axes[0, 0].plot(computedRs[0,:],example_result[0,:],'r-',Rs[samples[0],:],PR[samples[0],:],'b--')
    axes[0, 0].set_title(subtitle % samples[0])
    axes[0, 0].set_xlabel(xlabel)

    axes[0, 1].plot(computedRs[1,:],example_result[1,:],'r-',Rs[samples[1],:],PR[samples[1],:],'b--')
    axes[0, 1].set_title(subtitle % samples[1])
    axes[0, 1].set_xlabel(xlabel)

    axes[1, 0].plot(computedRs[2,:],example_result[2,:],'r-',Rs[samples[2],:],PR[samples[2],:],'b--')
    axes[1, 0].set_title(subtitle % samples[2])
    axes[1, 0].set_xlabel(xlabel)

    axes[1, 1].plot(computedRs[3,:],example_result[3,:],'r-',Rs[samples[3],:],PR[samples[3],:],'b--')
    axes[1, 1].set_title(subtitle% samples[3])
    axes[1, 1].set_xlabel(xlabel)

    fig.tight_layout()
    plt.show()

