import math
import pandas as pd

from P6_code.FinishedCode.importData import ImportEV
from P6_code.FinishedCode.dataTransformation import createTransformation
from P6_code.FinishedCode.functions import split_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, RepeatVector, TimeDistributed
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class mtotalModel:
    """A Model class used to predict multivariate data from the total dataframe.
    @param data: The data to create the model from. Has to be from:
    createTransformation(*params).getTotalData()
    @return: The model object, call createModel() to fit it.
    """
    def __init__(self, data):
        # Variables to create the model
        self.totalData = data
        self.val_split = 0.2

        # Scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Model Hyperparameters (configs)
        self.model = Sequential()
        self.n_steps_in = 10
        self.n_steps_out = 5
        self.n_nodes = 50

        self.batch_size = 50
        self.epochs = 250

    def createModel(self, type="LSTM"):
        """Creates the model with the given type and fits the data.
        @param type: The type of model that should be created. Can be the following:
        LSTM, GRU, CNN or LSTM-CNN

        @return: The model object, with a fitted model, which can be used for prediction.
        """
        print("Making Model")
        # Create Input and Target Features
        X, Y = self.totalData.copy(), self.totalData.copy()

        #Scale the data
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

        X_train, X_val = total_X[:-train_val_cutoff], total_X[-train_val_cutoff:]
        Y_train, Y_val = total_Y[:-train_val_cutoff], total_Y[-train_val_cutoff:]

        # Create the model
        self.title = type

        if type == "LSTM":
            self.model.add(LSTM(self.n_nodes, input_shape=(self.n_steps_in, self.n_features)))
        elif type == "GRU":
            self.model.add(GRU(self.n_nodes, input_shape=(self.n_steps_in, self.n_features)))
        else:
            raise Exception("The type of the model should either be LSTM or GRU")

        self.model.add(RepeatVector(self.n_steps_out))

        if type == "LSTM":
            self.model.add(LSTM(self.n_nodes, activation='relu', return_sequences=True))
        elif type == "GRU":
            self.model.add(GRU(self.n_nodes, activation='relu', return_sequences=True))
        else:
            raise Exception("The type of the model should either be LSTM or GRU")

        self.model.add(TimeDistributed(Dense(self.n_features)))

        # Printing the Structure of the model and compile it
        print(self.model.summary())
        self.model.compile(optimizer='adam', loss='mse', metrics=["mean_absolute_error"])

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

    def PredictTestSample(self, dataName, start, end):
        """Note: Should be run after creating the model
        Predicts a given timeframe from a given dataset.
        @param dataName: The dataset, where the prediction should take place
        @param start: The start date of the prediction
        @param end: The end date of the prediction

        @return: The model object, with the prediction results.
        """
        start = str(pd.to_datetime(start) - pd.Timedelta(1, "D"))

        # Import the data
        if dataName == "Caltech":
            df = ImportEV().getCaltech(start_date=start, end_date=end)
        elif dataName == "JPL":
            df = ImportEV().getJPL(start_date=start, end_date=end)
        elif dataName == "Office":
            df = ImportEV().getOffice(start_date=start, end_date=end)
        elif dataName == "Both":
            df = ImportEV().getBoth(start_date=start, end_date=end)
        else:
            raise Exception("Error, data parameter should be Caltech, JPL, Both or Office")

        total = createTransformation(df, start, end).getTotalData()

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
        """Note: Should be run after making a test prediction
        Makes a graph with the predictions and real values on the test data.
        After specifying the index of which column to predict
        @param column_to_predict: The index of the feature to plot
        0 is "kWhDelivered"
        1 is "carsCharging"
        2 is "carsIdle"
        """
        # plot baseline and predictions
        plt.plot(self.Y_test[:, column_to_predict])
        plt.plot(self.test_predict[:, column_to_predict])
        plt.show()

    def PlotLoss(self):
        """Note: Should be run after creating the model
        Makes a graph with the loss from each epoch when fitting the model.
        """
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
    start, end = "2018-05-01", "2018-11-01"
    df = ImportEV().getCaltech(start_date=start, end_date=end, removeUsers=False)
    Total_df = createTransformation(df, start, end).remove_outliers().getTotalData()

    model = mtotalModel(Total_df).createModel(type="GRU")
    model = model.PredictTestSample("Caltech", "2018-11-01", "2018-12-01")
    print(model.trainScore)
    print(model.valScore)
    print(model.testScore)

    model.PlotLoss()

    for i in range(model.n_features):
        model.PlotTestSample(column_to_predict=i)
