import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from P6_code.FinishedCode.importData import ImportEV
from P6_code.FinishedCode.dataTransformation import createTransformation
from P6_code.FinishedCode.functions import split_sequences

from keras.models import Sequential
from keras.layers import Dense, LSTM, ReLU
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

if __name__ == "__main__":
    start, end = "2018-06-01", "2018-12-01"
    df = ImportEV().getCaltech(start_date=start, end_date=end, removeUsers=True, userSampleLimit=25)
    Users = createTransformation(df, start, end)
    User_61 = Users.getUserData(user="000000061")

    ss = MinMaxScaler(feature_range=(0, 1))
    mm = MinMaxScaler(feature_range=(0, 1))

    X, y = User_61.drop(columns="chargingTime"), User_61["kWhDelivered"]
    X_trans = ss.fit_transform(X)
    y_trans = mm.fit_transform(y.values.reshape(-1, 1))

    n_steps_in, n_steps_out = 7, 1

    X_ss, y_mm = split_sequences(X_trans, y_trans, n_steps_in, n_steps_out)

    n_features = X_ss.shape[2]

    total_samples = len(X)
    train_test_cutoff = round(0.20 * total_samples)

    trainX, testX = X_ss[:-train_test_cutoff], X_ss[-train_test_cutoff:-1]
    trainY, testY = y_mm[1:-train_test_cutoff + 1], y_mm[-train_test_cutoff + 1:]

    model = Sequential()
    model.add(LSTM(5, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(5, activation='relu'))
    model.add(ReLU())
    model.add(Dense(n_steps_out))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(trainX, trainY, epochs=200, verbose=1, validation_data=(testX, testY))

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = mm.inverse_transform(trainPredict)
    trainY = mm.inverse_transform(trainY.reshape(-1, n_steps_out))
    testPredict = mm.inverse_transform(testPredict)
    testY = mm.inverse_transform(testY.reshape(-1, n_steps_out))

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[:, 0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % trainScore)
    testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % testScore)

    # shift train predictions for plotting
    trainPredictPlot = np.zeros_like(np.concatenate((y_trans, np.array([[np.nan]]*n_steps_in)), axis=0))
    trainPredictPlot[:] = np.nan
    trainPredictPlot[n_steps_in:len(trainPredict) + n_steps_in] = trainPredict[:, 0].reshape(-1, 1)

    # shift test predictions for plotting
    testPredictPlot = np.zeros_like(np.concatenate((y_trans, np.array([[np.nan]]*n_steps_in)), axis=0))
    testPredictPlot[:] = np.nan
    testPredictPlot[len(trainPredict) + (n_steps_in + n_steps_out) : len(y_trans) + 1] = testPredict[:, 0].reshape(-1, 1)

    # plot baseline and predictions
    plt.plot(mm.inverse_transform(y_trans))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

    #val_loss and loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()