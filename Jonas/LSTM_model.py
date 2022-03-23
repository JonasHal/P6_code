import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from P6_code.FinishedCode.importData import ImportEV
from P6_code.FinishedCode.dataTransformation import createUsers

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences):
            break

        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)

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
    model.add(LSTM(4, input_shape=(10, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

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
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = np.zeros_like(y_trans)
    trainPredictPlot[:] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.zeros_like(y_trans)
    testPredictPlot[:] = np.nan
    testPredictPlot[len(trainPredict) + (look_back*2)+1:len(y_trans)-2] = testPredict

    # plot baseline and predictions
    plt.plot(mm.inverse_transform(y_trans))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()