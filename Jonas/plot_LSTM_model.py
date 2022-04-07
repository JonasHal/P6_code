import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from P6_code.FinishedCode.importData import ImportEV
from P6_code.FinishedCode.dataTransformation import createUsers
from P6_code.FinishedCode.functions import split_sequences

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

if __name__ == "__main__":
    start, end = "2018-06-01", "2019-03-01"
    df = ImportEV().getCaltech(start_date=start, end_date=end, removeUsers=True, userSampleLimit=25)
    Users = createUsers(df, start, end)
    User_61 = Users.getUserData(user="000000061")

    ss = StandardScaler()
    mm = MinMaxScaler(feature_range=(0, 1))

    X, y = User_61.drop(columns="kWhDelivered"), User_61["chargingTime"]

    X_trans = ss.fit_transform(X)
    y_trans = mm.fit_transform(y.values.reshape(-1, 1))

    n_steps_in, n_steps_out = 10, 5

    X_ss, y_mm = split_sequences(X_trans, y_trans, n_steps_in, n_steps_out)

    n_features = X_ss.shape[2]

    total_samples = len(X)
    train_test_cutoff = round(0.20 * total_samples)

    trainX, testX = X_ss[:-train_test_cutoff], X_ss[-train_test_cutoff:]

    trainY, testY = y_mm[:-train_test_cutoff], y_mm[-train_test_cutoff:]

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(100, activation="relu"))
    model.add(Dense(n_steps_out))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=200, verbose=2)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = mm.inverse_transform(trainPredict)
    trainY = mm.inverse_transform(trainY)
    testPredict = mm.inverse_transform(testPredict)
    testY = mm.inverse_transform(testY)

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[:, 0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % trainScore)
    testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % testScore)

    # shift train predictions for plotting
    trainPredictPlot = np.zeros_like(y_trans)
    trainPredictPlot[:] = np.nan
    trainPredictPlot[n_steps_out:len(trainPredict) + n_steps_out] = trainPredict[:, -1].reshape(-1, 1)

    # shift test predictions for plotting
    testPredictPlot = np.zeros_like(y_trans)
    testPredictPlot[:] = np.nan
    testPredictPlot[len(trainPredict) + n_steps_out*2 + 2:len(y_trans)-1] = testPredict[:, -1].reshape(-1, 1)

    # plot baseline and predictions
    plt.plot(mm.inverse_transform(y_trans))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
