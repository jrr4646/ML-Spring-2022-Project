import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from numpy import newaxis, zeros

data = pd.read_csv("S_train_normalized_64.csv", header=None, skiprows=[15005])
data = pd.DataFrame(data)
ds   = data.to_numpy()

M = pd.read_csv("M.csv", header=None, skiprows=[15005])
M = pd.DataFrame(M)
M_param   = M.to_numpy()

ds = ds[...,newaxis]
M_param = M_param[...,newaxis]

print('\n Input row 1: \n')
print(ds[0])

input_shape = (64,1)

encoder_input = Input(shape = input_shape)
y = Reshape((8,8,1))(encoder_input)
y = Conv2D(16, (3,3), activation = 'relu', padding = 'same')(y) #8x8x16
y = Conv2D(32, (3,3), activation = 'relu')(y) #6x6x32
y = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(y) #6x6x64
y = Conv2D(128, (3,3), activation = 'relu')(y) #4x4x128
y = Flatten()(y) #2048x1
encoder_output = Dense(2, activation = 'relu')(y) #2x1

encoder = Model(inputs = encoder_input, outputs = encoder_output)

decoder_input = Input(shape = (2,))
y = Dense(2048, activation = 'relu')(decoder_input)
y = Reshape((4,4,128))(y)
y = Conv2DTranspose(64,(3,3), activation = 'relu')(y) #6x6x64
y = Conv2DTranspose(32,(3,3), activation = 'relu', padding = 'same')(y) #6x6x32
y = Conv2DTranspose(16,(3,3), activation = 'relu')(y) #8x8x16
y = Conv2DTranspose(1,(3,3), activation = 'relu', padding = 'same')(y) #8x8x1
decoder_output = Reshape((64,1))(y)

decoder = Model(inputs = decoder_input, outputs = decoder_output)

FOM    = Input(shape = (64,1))
latent = encoder(FOM)
Output = decoder(latent)
autoencoder = Model(inputs = FOM, outputs = Output)
autoencoder.summary()

opt_1 = Adam(lr = 0.001, decay=1e-6)
autoencoder.compile(opt_1, loss = 'mse')
autoencoder.fit(ds,ds, epochs = 1, batch_size = 100, validation_split = 0.2)

input_shape_DF = (2,1)
DF_input = Input(shape = input_shape_DF)
z = Flatten()(DF_input)
z = Dense(500, activation = 'relu')(z)
z = Dense(500, activation = 'relu')(z)
z = Dense(500, activation = 'relu')(z)
DF_output = Dense(2, activation = 'relu')(z)

DFNN = Model(inputs = DF_input, outputs = DF_output)
DFNN.summary()

opt_2 = Adam(lr = 0.0005, decay=1e-6)
DFNN.compile(opt_2, loss = 'mse')
DFNN.fit(M_param, encoder.predict(ds), epochs = 1, batch_size = 50)

example = encoder.predict([ds[0].reshape(-1,64,1)])
example2 = DFNN.predict([M_param[0].reshape(-1,2,1)])
example3 = decoder.predict(example2)

print('encoder first row: ')
print(example)
print('DFNN output first row: ')
print(example2)
print('FOM from DFNN: ')
print(example3)

train = zeros((11,64))
test = zeros((11,64))
M_37 = zeros((11,2))
M_37[:,0] = 3.7
t = np.linspace(0.0, 3e-5, num=11)

for i in range (0,11):
    M_37[i,1] = t[i]

M_37 = M_37[...,newaxis]

for i in range (0,11):
    predict_train = decoder.predict(DFNN.predict([M_param[i].reshape(-1,2,1)]))
    train[i,:] = predict_train[...,0]

    predict = decoder.predict(DFNN.predict([M_37[i].reshape(-1,2,1)]))
    test[i,:] = predict[...,0]


np.savetxt('test_3.7.csv',test,delimiter=',')
np.savetxt('train.csv',train,delimiter=',')
