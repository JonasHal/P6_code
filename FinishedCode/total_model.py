import math
import numpy as np
import pandas as pd

from P6_code.FinishedCode.importData import ImportEV
from P6_code.FinishedCode.dataTransformation import createTransformation
from P6_code.FinishedCode.functions import split_sequences, getModelStructure
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class totalModel:
    """A Model class used to predict one given feature from the total dataframe.
    @param data: The data to create the model from. Has to be from:
    createTransformation(*params).getTotalData()
    @param feature_name: The feature that should be predicted
    @return: The model object, call createModel() to fit it.
    """
    def __init__(self, data, feature_name="carsCharging"):
        # Variables to create the model
        self.totalData = data
        self.val_split = 0.2
        self.feature_name = feature_name

        # Scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.mmy = MinMaxScaler(feature_range=(0, 1))

        # Model Hyperparameters (configs)
        self.model = Sequential()
        self.n_steps_in = 3
        self.n_steps_out = 3
        self.n_nodes = 50
        self.n_nodes_cnn = 64

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
        X, y = self.totalData.copy(), self.totalData[self.feature_name].values.reshape(-1, 1)

        #Scale og fit the data
        self.scaler = self.scaler.fit(X)
        self.mmy = self.mmy.fit(y)
        X_trans = self.scaler.transform(X)
        self.y_trans = self.mmy.transform(y)

        # Info about the input features
        print("The input features are: " + str(X.columns))
        self.n_features = len(X.columns)

        # Split the data into training and validation data
        total_samples = len(X)
        train_val_cutoff = round(self.val_split * total_samples)

        total_X, total_y = split_sequences(X_trans, self.y_trans, self.n_steps_in, self.n_steps_out)

        X_train, X_val = total_X[:-train_val_cutoff], total_X[-train_val_cutoff:]
        y_train, y_val = total_y[:-train_val_cutoff], total_y[-train_val_cutoff:]

        # Create the model
        self.title = type
        self.model = getModelStructure(type, self.n_steps_in, self.n_steps_out, self.n_features, self.n_nodes, self.n_nodes_cnn)

        # Printing the Structure of the model and compile it
        print(self.model.summary())
        self.model.compile(optimizer='adam', loss='mse', metrics=["mean_absolute_error"])

        # Fit the data and trains the model
        self.history = self.model.fit(x=X_train, y=y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                               validation_data=(X_val, y_val))

        # Make and Invert predictions
        self.trainPredict = self.mmy.inverse_transform(self.model.predict(X_train))
        trainY = self.mmy.inverse_transform(y_train.reshape(-1, self.n_steps_out))
        self.valPredict = self.mmy.inverse_transform(self.model.predict(X_val))
        valY = self.mmy.inverse_transform(y_val.reshape(-1, self.n_steps_out))

        # Calculate the following: Root Mean Squared Error, Mean absolute error and Mean Absolute Percentage Error
        self.trainRMSE_Score = math.sqrt(mean_squared_error(trainY[:, -1], self.trainPredict[:, -1]))
        self.trainMAE_Score = mean_absolute_error(trainY[:, -1], self.trainPredict[:, -1])

        self.valRMSE_Score = math.sqrt(mean_squared_error(valY[:, -1], self.valPredict[:, -1]))
        self.valMAE_Score = mean_absolute_error(valY[:, -1], self.valPredict[:, -1])

        print('Train Score: %.2f RMSE, ' % self.trainRMSE_Score + '%.2f MAE' % self.trainMAE_Score)
        print('Validation Score: %.2f RMSE, ' % self.valRMSE_Score + '%.2f MAE' % self.valMAE_Score)

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
        X, y = total.copy(), total[self.feature_name].values.reshape(-1, 1)

        # Scale the Data
        X_trans = self.scaler.transform(X)
        y_trans = self.mmy.transform(y)

        # Split the data for prediction in the RNN models
        X_test, self.y_test = split_sequences(X_trans, y_trans, self.n_steps_in, self.n_steps_out)

        # Make and Invert predictions
        self.test_predict = self.model.predict(X_test)

        # Calculate the following: Root Mean Squared Error, Mean absolute error and Mean Absolute Percentage Error
        self.testRMSE_Score = math.sqrt(mean_squared_error(self.y_test[:, -1], self.test_predict[:, -1]))
        self.testMAE_Score = mean_absolute_error(self.y_test[:, -1], self.test_predict[:, -1])

        return self

    def PlotTestSample(self):
        """Note: Should be run after making a test prediction
        Makes a graph with the predictions and real values on the test data.
        """

        # plot baseline and predictions
        plt.plot(self.y_test[:, 0])
        plt.plot(self.test_predict[:, 0])
        plt.title(self.title)
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

    def PlotTrainVal(self):
        """Note: Should be run after creating the model
        Makes a graph with the predictions and real values on the training and validation data.
        The yellow line is Training
        The Green line is Validation
        """
        # shift train predictions for plotting
        trainPredictPlot = np.zeros_like(np.concatenate((self.y_trans, np.array([[np.nan]] * self.n_steps_in)), axis=0))
        trainPredictPlot[:] = np.nan
        trainPredictPlot[self.n_steps_in:len(self.trainPredict) + self.n_steps_in] = self.trainPredict[:, 0].reshape(-1, 1)

        # shift test predictions for plotting
        valPredictPlot = np.zeros_like(np.concatenate((self.y_trans, np.array([[np.nan]] * self.n_steps_in)), axis=0))
        valPredictPlot[:] = np.nan
        valPredictPlot[len(self.trainPredict) + (self.n_steps_in + self.n_steps_out) : len(self.y_trans) + 1] = self.valPredict[:, 0].reshape(-1, 1)

        # plot baseline and predictions
        plt.title(self.title)
        plt.plot(self.mmy.inverse_transform(self.y_trans))
        plt.plot(trainPredictPlot)
        plt.plot(valPredictPlot)
        plt.show()

if __name__ == "__main__":
    # The model will always be first input
    start, end = "2018-10-20", "2018-11-01"
    df = ImportEV().getCaltech(start_date=start, end_date=end, removeUsers=False)
    Total_df = createTransformation(df, start, end).remove_outliers().getTotalData()

    model = totalModel(Total_df).createModel(type="LSTM-CNN")
    model = model.PredictTestSample("Caltech", "2019-01-01", "2019-01-15")
    print(model.trainRMSE_Score)
    print(model.valRMSE_Score)
    print(model.testRMSE_Score)

    model.PlotLoss()

    model.PlotTestSample()

    model.PlotTrainVal()
