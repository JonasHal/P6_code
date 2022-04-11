import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from P6_code.FinishedCode.importData import ImportEV
from P6_code.FinishedCode.dataTransformation import createUsers
from P6_code.FinishedCode.functions import split_sequences
from sklearn.metrics import mean_squared_error as mse
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
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

    #LSTM
    """
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences= True))
    model.add(LSTM(32, activation='relu'))
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    
    #GRU

    model = Sequential()
    model.add(InputLayer((7, 2)))
    model.add(GRU(64))
    model.add(Dense(8, 'relu'))
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    """
    #CNN

    model = Sequential()
    model.add(InputLayer((7, 2)))
    model.add(Conv1D(64, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(64, kernel_size=2))
    model.add(Dense(100, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(8, 'relu'))

    model.add(Dense(2))
    model.summary()
    model.compile(optimizer='adam', loss='mse')
    

    #cp4 = ModelCheckpoint('model4/', save_best_only=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10) # we can use validation_split=0.1

    testPredict = model.predict(X_test)

    predictions = scaler.inverse_transform(testPredict)
    y_test = scaler.inverse_transform(y_test)

    print(predictions)
    print(predictions[:, 0])
    print(predictions[:, 1])

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



    #print(plot_predictions(model, X_test, y_test).to_string())
    print('done')


