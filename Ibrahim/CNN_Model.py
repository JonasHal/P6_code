import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from P6_code.FinishedCode.importData import ImportEV
from P6_code.FinishedCode.dataTransformation import createUsers
from P6_code.FinishedCode.functions import split_sequences

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler



if __name__ == "__main__":
    start, end = "2018-06-01", "2018-11-09"
    df = ImportEV().getCaltech(start_date=start, end_date=end, removeUsers=True, userSampleLimit=25)
    Users = createUsers(df, start, end)
    User_61 = Users.getUserData(user="000000061")

    ss = StandardScaler()
    mm = MinMaxScaler(feature_range=(0, 1))

    X, y = User_61.drop(columns=['kWhDelivered']), User_61.kWhDelivered

    X_trans = ss.fit_transform(X)
    y_trans = mm.fit_transform(y.values.reshape(-1, 1))
    look_back = 5

    X_ss, y_mm = split_sequences(X_trans, y_trans, 10, look_back)

    total_samples = len(X)
    train_test_cutoff = round(0.10 * total_samples)

    trainX, testX = X_ss[:-train_test_cutoff], X_ss[-train_test_cutoff:]

    trainY, testY = y_mm[:-train_test_cutoff], y_mm[-train_test_cutoff:]

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, input_shape=(10, look_back)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    #fit model
    model.fit(trainX, trainY, epochs=10000, verbose=1)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = mm.inverse_transform(trainPredict.reshape(-1, 1))
    trainY = mm.inverse_transform(trainY)
    testPredict = mm.inverse_transform(testPredict.reshape(-1, 1))
    testY = mm.inverse_transform(testY)

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[:, 0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % trainScore)
    testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % testScore)



    print('done')




