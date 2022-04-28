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
        self.data = "Caltech"
        self.train_start = "2018-10-01"
        self.train_end = "2018-12-01"
        self.val_split = 0.2

        # Scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Model Hyperparameters (configs)
        self.model = Sequential()
        self.n_steps_in = 25
        self.n_steps_out = 10
        self.n_nodes = 64

        self.batch_size = 25
        self.epochs = 500

    def create_model(self, type="LSTM"):
        if self.data == "Caltech":
            df = ImportEV().getCaltech(start_date=self.train_start, end_date=self.train_end)
        elif self.data == "JPL":
            df = ImportEV().getJPL(start_date=self.train_start, end_date=self.train_end)
        elif self.data == "Office":
            df = ImportEV().getOffice(start_date=self.train_start, end_date=self.train_end)
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

        # Printing the Structure of the model and compile it
        print(self.model.summary())
        self.model.compile(optimizer='adam', loss='mse')

        # Fit the data and trains the model
        self.history = self.model.fit(x=X_train, y=Y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                               validation_data=(X_val, Y_val))

        # Make and Invert predictions
        train_predict = self.scaler.inverse_transform(self.model.predict(X_train)[:,-1,:].reshape(-1, self.n_features))
        val_predict = self.scaler.inverse_transform(self.model.predict(X_val)[:,-1,:].reshape(-1, self.n_features))

        # calculate root mean squared error
        self.trainScore = math.sqrt(mean_squared_error(Y_train[:, -1, :].reshape(-1, self.n_features), train_predict))
        self.valScore = math.sqrt(mean_squared_error(Y_val[:, -1, :].reshape(-1, self.n_features), val_predict))

        # Return the model and the scalers
        return self

    def PredictTestSample(self, start, end):
        start = str(pd.to_datetime(start) - pd.Timedelta(1, "D"))

        # Import the data
        if self.data == "Caltech":
            df = ImportEV().getCaltech(start_date=start, end_date=end)
        elif self.data == "JPL":
            df = ImportEV().getJPL(start_date=start, end_date=end)
        elif self.data == "Office":
            df = ImportEV().getOffice(start_date=start, end_date=end)
        else:
            print("Error, data parameter should be Caltech, JPL or Office")

        total = createTotal(df, start, end).getTotalData()

        print("Making Model")
        print(total)

        # Create Input and Target Features
        X, Y = total.copy(), total.copy()

        # Scale the Data
        X_trans = self.scaler.transform(X)
        Y_trans = self.scaler.transform(Y)

        # Split the data for prediction in the RNN models
        X_test, Y_test = split_sequences(X_trans, Y_trans, self.n_steps_in, self.n_steps_out)

        # Make and Invert predictions
        self.test_predict = self.scaler.inverse_transform(
            self.model.predict(X_test)[:, -1, :].reshape(-1, self.n_features))
        self.Y_test = self.scaler.inverse_transform(Y_test[:, -1, :].reshape(-1, self.n_features))

        self.testScore = math.sqrt(mean_squared_error(self.Y_test, self.test_predict))

        return self

    def PlotTestSample(self, column_to_predict=0):
        # plot baseline and predictions
        plt.plot(self.Y_test[:, column_to_predict])
        plt.plot(self.test_predict[:, column_to_predict])
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
    model = model.PredictTestSample("2018-12-01", "2019-01-01")
    print(model.trainScore)
    print(model.valScore)
    print(model.testScore)

    model.PlotLoss()

    for i in range(model.n_features):
        model.PlotTestSample(column_to_predict=i)
