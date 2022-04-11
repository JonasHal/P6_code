import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from P6_code.FinishedCode.importData import ImportEV
from P6_code.FinishedCode.dataTransformation import createUsers
from P6_code.FinishedCode.functions import split_sequences

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler



if __name__ == "__main__":
    start, end = "2018-06-01", "2018-11-09"
    df = ImportEV().getCaltech(start_date=start, end_date=end, removeUsers=True, userSampleLimit=25)
    Users = createUsers(df, start, end)
    User_61 = Users.getUserData(user="000000061")




   
    df = User_61[['chargingTime','kWhDelivered']]
    #X, y = User_61.drop(columns=['kWhDelivered']), User_61.kWhDelivered
    #print(df.shape)

    # LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
    # normalize the dataset
    scaler = StandardScaler()
    scaler = scaler.fit(df)
    df_scaled = scaler.transform(df)

    #scaler = StandardScaler()
    #df_scaled = scaler.fit_transform(df)
    #mm = MinMaxScaler(feature_range=(0, 1))

    trainX = []
    trainY = []

    n_feature = 1
    n_past = 4

    for i in  range(n_past, len(df_scaled) - n_feature +1):
        trainX.append(df_scaled[i - n_past:i, 0:df.shape[1]])
        trainY.append(df_scaled[i + n_feature - 1:i + n_feature, 0])

    trainX, trainY = np.array(trainX), np.array(trainY)
    print(trainX, trainY)



"""
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences= True))
    model.add(LSTM(32, activation='relu', return_sequences= True))
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # fit the model
    history = model.fit(trainX, trainY, epochs=5, batch_size=16, validation_split=0.1, verbose=1)
"""

