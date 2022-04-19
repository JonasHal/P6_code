import numpy as np
import matplotlib.pyplot as plt
from P6_code.FinishedCode.importData import ImportEV
from P6_code.FinishedCode.dataTransformation import createUsers
import math

from keras.models import Sequential
from keras.layers import *
from keras.layers.convolutional import MaxPooling1D

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def df_to_X_y(df, window_size=7):
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


    df=User_61[['chargingTime','kWhDelivered']]

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = df_to_X_y(df_scaled)

    X_train, y_train = X[:124], y[:124] # TODO: more general
    X_val, y_val = X[124:139], y[124:139]
    X_test, y_test = X[139:], y[139:]
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

    name = '1_Layer'
    lstm_model_1 = Sequential([
        InputLayer((7, 2)),
        LSTM(64, activation='relu', name='LSTM1'),
        Dense(64, activation='relu', name='Dense'),
        Dense(2)
    ], name=name)

    name = '2_Layer'
    lstm_model_2 = Sequential([
        InputLayer((7, 2)),
        LSTM(64, activation='relu', return_sequences=True, name='LSTM1'),
        LSTM(64, activation='relu', name='LSTM2'),
        Dense(64, activation='relu', name='Dense'),
        Dense(2)
    ], name=name)

    name='3_layer'
    lstm_model_3 = Sequential([
        InputLayer((7, 2)),
        LSTM(64, activation='relu', kernel_initializer='he_normal',return_sequences=True, name='LSTM1'),
        LSTM(64, activation='relu', return_sequences=True, name='LSTM2'),
        LSTM(64, activation='relu', name='LSTM3'),
        Dense(64, activation='relu', name='Dense'),
        Dense(2)
    ], name=name)

    lstm_models = [lstm_model_1, lstm_model_2, lstm_model_3]

    history_dict = {}
    for model in lstm_models:
        model.compile(
            loss='mse',
            optimizer='adam',
        )

        history = model.fit(
            X_train, y_train,
            batch_size=16,
            epochs=50, verbose=1,
            validation_data=(X_val, y_val)
        )

        """
        trainPredict = model.predict(X_train)
        testPredict = model.predict(X_test)

        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform(y_train)
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform(y_test)

        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[:, 0], trainPredict[:, 0]))
        print('Train Score: %.2f RMSE' % trainScore)
        testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % testScore)

        for j in range(len(df.columns)):
            # shift train predictions for plotting
            trainPredictPlot = np.zeros_like(y[:, j])
            trainPredictPlot[:] = np.nan
            trainPredictPlot[:124] = trainPredict[:, j]

            # shift test predictions for plotting
            testPredictPlot = np.zeros_like(y[:, j])
            testPredictPlot[:] = np.nan
            testPredictPlot[139:] = testPredict[:, j]

            # plot baseline and predictions
            plt.plot(scaler.inverse_transform(y)[:, j])
            plt.plot(trainPredictPlot)
            plt.plot(testPredictPlot)
            plt.title(model.name)
            plt.show()

        #Epoch_ Val_loss
        """
        history_dict[model.name] = history
        for history in history_dict:
            val_loss = history_dict[history].history['val_loss']
            plt.plot(val_loss, label=history)
        plt.ylabel('Validation Loss')
        plt.xlabel('Epochs')
        plt.title('LSTM loss')
        plt.legend()
        plt.show()
