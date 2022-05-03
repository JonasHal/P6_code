import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from P6_code.FinishedCode.importData import ImportEV
from P6_code.FinishedCode.dataTransformation import createTransformation
from P6_code.FinishedCode.functions import split_sequences
from keras.layers import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D

from sklearn.metrics import mean_squared_error, mean_absolute_error
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
    Users = createTransformation(df, start, end)
    User_61 = Users.getUserData(user="000000061")

    df = User_61[['chargingTime', 'kWhDelivered']]
    # X, y = User_61.drop(columns=['kWhDelivered']), User_61.kWhDelivered
    print(df.shape)

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = df_to_X_y(df_scaled)

    X_train, y_train = X[:124], y[:124]
    X_val, y_val = X[124:139], y[124:139]
    X_test, y_test = X[139:], y[139:]

    model = Sequential()
    # LSTM
    model.add(InputLayer((4,2)))
    model.add(LSTM(64, activation='relu', return_sequences=True))
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
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(2))
    model.add(Activation('linear'))
    # Print whole structure of the model
    print(model.summary())

    # training the train data with n epoch
    model.compile(loss="mse", optimizer="Adam")  # adam
    history = model.fit(X_train,
              y_train,
              epochs=200,
              batch_size=80, verbose=1, shuffle=False, validation_data= (X_val, y_val))
    # plot the loss
    '''
    val_loss = history.history['val_loss']
    plt.plot(val_loss, label=history)
    plt.ylabel('Validation Loss')
    plt.xlabel('Epochs')
    plt.title('LSTM-CNN loss')
    plt.legend()
    plt.show()
    '''
    testPredict = model.predict(X_test)

    predictions = scaler.inverse_transform(testPredict)
    y_test = scaler.inverse_transform(y_test)

    print(predictions)
    print(predictions[:, 0])
    print(predictions[:, 1])

    print('The RMSE is ', '%e' % math.sqrt(mean_squared_error()))
    print('The RMAE is ', '%e' % math.sqrt(
        mean_absolute_error(df.loc[df.index >= df.index[int(len(df.index) * 0.8)], 'Dollar'],
                            df.loc[df.index >= df.index[int(len(df.index) * 0.8)], 'Pred'])))
    print('The MAPE is ', '%e' % mape(df.loc[df.index >= df.index[int(len(df.index) * 0.8)], 'Dollar'],
                                      df.loc[df.index >= df.index[int(len(df.index) * 0.8)], 'Pred']))

    chargingTime_preds, kWhDelivered_preds = predictions[:, 0], predictions[:, 1]
    chargingTime_actuals, kWhDelivered_actuals = y_test[:, 0], y_test[:, 1]
    df = pd.DataFrame(data={'chargingTime Predictions': chargingTime_preds,
                            'chargingTime Actuals': chargingTime_actuals,
                            'kWhDelivered Predictions': kWhDelivered_preds,
                            'kWhDelivered Actuals': kWhDelivered_actuals
                            })
    print(df.to_string())

    plt.plot(df['kWhDelivered Predictions'])
    plt.plot(df['kWhDelivered Actuals'])
    plt.legend(['Predictions', 'Actuals'])
    plt.show()