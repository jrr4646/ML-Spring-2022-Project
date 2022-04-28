import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv1D, MaxPooling1D, UpSampling1D, Conv1DTranspose
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from numpy import newaxis

data = pd.read_csv("S_train_log_normalized.csv", header=None, skiprows=[15005])
data = pd.DataFrame(data)
ds   = data.to_numpy()

M = pd.read_csv("M.csv", header=None, skiprows=[15005])
M = pd.DataFrame(M)
M_param   = M.to_numpy()

ds = ds[...,newaxis]
M_param = M_param[...,newaxis]

print("Input shape: \n")
print(ds.shape)

print('\n Input row 1: \n')
print(ds[0])

input_shape = (20,1)

encoder_input = Input(shape = input_shape)
y = Conv1D(8,3, activation = 'relu')(encoder_input) #18x8
y = Conv1D(16,3, activation = 'relu')(y) #16x16
y = Conv1D(32,3, activation = 'relu')(y) #14x32
y = Conv1D(64,3,activation = 'relu')(y) #12x64
y = Conv1D(128,3,activation = 'relu')(y) #10x128
y = Flatten()(y) #1280x1
encoder_output = Dense(2, activation = 'relu')(y) #2x1

encoder = Model(inputs = encoder_input, outputs = encoder_output)

decoder_input = Input(shape = (2,))
y = Dense(1280, activation = 'relu')(decoder_input) #1280x1
y = Reshape((10,128))(y) #10x128
y = Conv1DTranspose(64,3, activation = 'relu')(y) #12x64
y = Conv1DTranspose(32,3, activation = 'relu')(y) #14x32
y = Conv1DTranspose(16,3, activation = 'relu')(y) #16x16
y = Conv1DTranspose(8,3, activation = 'relu')(y) #18x8
decoder_output = Conv1DTranspose(1,3, activation = None)(y) #20x1

decoder = Model(inputs = decoder_input, outputs = decoder_output)

FOM    = Input(shape = (20,1))
latent = encoder(FOM)
Output = decoder(latent)

autoencoder = Model(inputs = FOM, outputs = Output)
#autoencoder.summary()

opt_1 = Adam(lr = 0.001, decay=1e-6)
autoencoder.compile(opt_1, loss = 'mse')
autoencoder.fit(ds,ds, epochs = 10, batch_size = 100, validation_split = 0.2)

input_shape_DF = (2,1)
DF_input = Input(shape = input_shape_DF)
z = Flatten()(DF_input)
z = Dense(100, activation = 'relu')(z)
z = Dense(100, activation = 'relu')(z)
z = Dense(100, activation = 'relu')(z)
DF_output = Dense(2, activation = None)(z)

DFNN = Model(inputs = DF_input, outputs = DF_output)

opt_2 = Adam(lr = 0.0005, decay=1e-6)
DFNN.compile(opt_2, loss = 'mse')
DFNN.fit(M_param, encoder.predict(ds), epochs = 10, batch_size = 50)

example  = encoder.predict([ds[0].reshape(-1,20,1)])  #latent space
example2 = DFNN.predict([M_param[0].reshape(-1,2,1)]) #feed forward prediction of latent space
example3 = decoder.predict(example2)                  #estimate of full order solution 

print('encoder first row: ')
print(example)
print('DFNN output first row: ')
print(example2)
print('FOM from DFNN: ')
print(example3)

np.savetxt('DFNN_test.csv',example3[0,...],delimiter=',')
