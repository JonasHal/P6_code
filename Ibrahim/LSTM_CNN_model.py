import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from P6_code.FinishedCode.importData import ImportEV
from P6_code.FinishedCode.dataTransformation import createUsers
from P6_code.FinishedCode.functions import split_sequences
from keras.layers import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def df_to_X_y(df, window_size=4):
  df_as_np = df
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [r for r in df_as_np[i:i+window_size]]
    X.append(row)
    label = [df_as_np[i+window_size][0], df_as_np[i+window_size][1]]
    y.append(label)
  return np.array(X), np.array(y)

if __name__ == "__main__":
    start, end = "2018-06-01", "2018-11-09"
    df = ImportEV().getCaltech(start_date=start, end_date=end, removeUsers=True, userSampleLimit=25)
    Users = createUsers(df, start, end)
    User_61 = Users.getUserData(user="000000061")

    df = User_61[['chargingTime', 'kWhDelivered']]
    # X, y = User_61.drop(columns=['kWhDelivered']), User_61.kWhDelivered
    print(df.shape)

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    # mm = MinMaxScaler(feature_range=(0, 1))

    trainX = []
    trainY = []

    n_feature = 1
    n_past = 4

    for i in range(n_past, len(df_scaled) - n_feature + 1):
        trainX.append(df_scaled[i - n_past:i, 0:df.shape[1]])
        trainY.append(df_scaled[i + n_feature - 1:i + n_feature, 0])

    trainX, trainY = np.array(trainX), np.array(trainY)

    print('trainX shape == {}.'.format(trainX.shape))
    print('trainY shape == {}.'.format(trainY.shape))

    model = Sequential()
    # LSTM
    model.add(InputLayer((4,2)))
    model.add(LSTM(64, return_sequences=True))
    # model.add(LSTM(input_shape = (look_back,1), input_dim=1, output_dim=6, return_sequences=True))
    # model.add(Dense(1))
    # CNN
    model.add(Convolution1D(
                            64,  # 128
                            activation='relu',
                            kernel_size=2
                            ))


    model.add(MaxPooling1D(pool_size=2))


    # model.add(Dropout(0.25))

    model.add(Dropout(0.25))
    model.add(Activation('relu'))  # ReLU : y = max(0,x)
    model.add(Flatten())
    model.add(Dense(2))
    model.add(Activation('linear'))
    # Print whole structure of the model
    print(model.summary())

    # training the train data with n epoch
    model.compile(loss="mse", optimizer="Adam")  # adam
    model.fit(trainX,
              trainY,
              epochs=100,
              batch_size=80, verbose=1)