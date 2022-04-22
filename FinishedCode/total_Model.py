import math
import numpy as np
import pandas as pd

from P6_code.FinishedCode.importData import ImportEV
from P6_code.FinishedCode.total_dataTransformation import createTotal
from P6_code.FinishedCode.functions import split_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, RepeatVector, TimeDistributed
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt


class Model:
    def __init__(self):
        # Variables to create the model
        self.train_start = "2018-06-01"
        self.train_end = "2018-12-01"
        self.val_split = 0.2

        # Scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Model Hyperparameters (configs)
        self.model = Sequential()
        self.n_steps_in = 20
        self.n_steps_out = 10
        self.n_nodes = 50

        self.batch_size = 200
        self.epochs = 1000

    def create_model(self, type="LSTM", data="Caltech"):
        if data == "Caltech":
            df = ImportEV().getCaltech(start_date=self.train_start, end_date=self.train_end)
        elif data == "JPL":
            df = ImportEV().getJPL(start_date=self.train_start, end_date=self.train_end)
        elif data == "Office":
            df = ImportEV().getJPL(start_date=self.train_start, end_date=self.train_end)
        else:
            print("Error, data parameter should be Caltech, JPL or Office")

        total = createTotal(df, self.train_start, self.train_end).getTotalData()

        print("Making Model")

        # Create Input and Target Features
        X, Y = total.copy(), total.copy()

        self.scaler = self.scaler.fit(X)
        X_trans = self.scaler.transform(X)
        Y_trans = self.scaler.transform(Y)

        # Info about the input features
        print("The input features are: " + str(X.columns))
        self.n_features = len(X.columns)

        # Split the data into training and validation data
        total_samples = len(X)
        train_val_cutoff = round(self.val_split * total_samples)

        total_X, total_Y = split_sequences(X_trans, Y_trans, self.n_steps_in, self.n_steps_out)

        X_train, X_val = total_X[:-train_val_cutoff], total_X[-train_val_cutoff:-1]
        Y_train, Y_val = total_Y[1:-train_val_cutoff + 1], total_Y[-train_val_cutoff + 1:]

        # Create the model
        if type == "LSTM":
            self.model.add(LSTM(self.n_nodes, input_shape=(self.n_steps_in, self.n_features)))
        elif type == "GRU":
            self.model.add(GRU(self.n_nodes, input_shape=(self.n_steps_in, self.n_features)))
        else:
            raise Exception("The type of the model should either be LSTM or GRU")

        self.title = type

        self.model.add(RepeatVector(self.n_steps_out))
        self.model.add(LSTM(self.n_nodes, activation='relu', return_sequences=True))
        self.model.add(TimeDistributed(Dense(self.n_features)))
        self.model.compile(optimizer='adam', loss='mse')

        # Fit the data and trains the model
        self.history = self.model.fit(x=X_train, y=Y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                               validation_data=(X_val, Y_val))

        # Make and Invert predictions
        train_predict = self.scaler.inverse_transform(self.model.predict(X_train)[:,-1,:].reshape(-1, self.n_features))
        val_predict = self.scaler.inverse_transform(self.model.predict(X_val)[:,-1,:].reshape(-1, self.n_features))
        print(train_predict.shape)
        print(Y_train[:, -1, :].shape)

        # calculate root mean squared error
        self.trainScore = math.sqrt(mean_squared_error(Y_train[:, -1, :].reshape(-1, self.n_features), train_predict))
        self.valScore = math.sqrt(mean_squared_error(Y_val[:, -1, :].reshape(-1, self.n_features), val_predict))

        # Return the model and the scalers
        return self

    def PredictTestSample(self, start, end, userSampleLimit):
        # Import the data
        df_test = ImportEV().getCaltech(start_date=start, end_date=end, removeUsers=True,
                                        userSampleLimit=userSampleLimit)
        users = createTotal(df_test, start, end)

        # Save the user_ids for return
        user_id = users.data.userID.unique()
        user_df_test = []

        for user in user_id:
            user_df_test.append(users.getUserData(user=user))

        # Create Input and Target Features
        X_test, Y_test = [], []

        for user in user_df_test:
            Y_test.append(user[self.target_feature])
            X_test.append(user.drop(columns=[self.drop_feature]))

        # Scale the Data
        X_test_scaled = []

        for user in X_test:
            X_test_scaled.append(self.ss.transform(user))

        # Split the data for prediction in the RNN models
        users_test_X, self.users_test_Y = [], []

        for user in range(len(X_test_scaled)):
            user_test_X, user_test_Y = split_sequences(X_test_scaled[user], np.array(Y_test[user]).reshape(-1, 1),
                                                       self.n_steps_in, self.n_steps_out)
            users_test_X.append(user_test_X)
            self.users_test_Y.append(user_test_Y)

        # Predict the data
        self.test_predict = []
        self.testScore = []

        # Make and Invert predictions
        for user in range(len(users_test_X)):
            self.test_predict.append(
                self.mm.inverse_transform(self.model.predict(users_test_X[user]).reshape(-1, self.n_steps_out)))

            # calculate root mean squared error
            self.testScore.append(
                math.sqrt(mean_squared_error(self.users_test_Y[user][:, 0], self.test_predict[user][:, 0])))

        return self

    def PlotTestSample(self, user=0):
        # plot baseline and predictions
        plt.plot(self.users_test_Y[user][:, 0])
        plt.plot(self.test_predict[user][:, 0])
        plt.show()

    def PlotLoss(self):
        loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]

        plt.plot(loss, label="train_loss")
        plt.plot(val_loss, label="val_loss")
        plt.title(self.title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # The model will always be first input
    model = Model().create_model(type="LSTM")
    #model = model.PredictTestSample("2018-12-01", "2019-01-01", 15)
    print(model.trainScore)
    print(model.valScore)
    #print(model.testScore)

    model.PlotLoss()

    #model.PlotTestSample(user=3)
