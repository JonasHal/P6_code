import math
import pandas as pd

from P6_code.FinishedCode.importData import ImportEV
from P6_code.FinishedCode.dataTransformation import createTransformation
from P6_code.FinishedCode.functions import split_sequences, getModelStructure
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, RepeatVector, TimeDistributed
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class totalModel:
    def __init__(self, data):
        # Variables to create the model
        self.totalData = data
        self.val_split = 0.2
        self.feature_name = "carsCharging"

        # Scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.mmy = MinMaxScaler(feature_range=(0, 1))

        # Model Hyperparameters (configs)
        self.model = Sequential()
        self.n_steps_in = 10
        self.n_steps_out = 5
        self.n_nodes = 50
        self.n_nodes_cnn = 64

        self.batch_size = 50
        self.epochs = 250

    def create_model(self, type="LSTM"):
        print("Making Model")
        # Create Input and Target Features
        X, y = self.totalData.copy(), self.totalData[self.feature_name]

        #Scale the data
        self.scaler = self.scaler.fit(X)
        X_trans = self.scaler.transform(X)
        y_trans = self.mmy.transform(y)

        # Info about the input features
        print("The input features are: " + str(X.columns))
        self.n_features = len(X.columns)

        # Split the data into training and validation data
        total_samples = len(X)
        train_val_cutoff = round(self.val_split * total_samples)

        total_X, total_y = split_sequences(X_trans, y_trans, self.n_steps_in, self.n_steps_out)

        X_train, X_val = total_X[:-train_val_cutoff], total_X[-train_val_cutoff:-1]
        y_train, y_val = total_y[1:-train_val_cutoff + 1], total_y[-train_val_cutoff + 1:]

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
        train_predict = self.mmy.inverse_transform(self.model.predict(X_train))
        trainY = self.mmy.inverse_transform(y_train.reshape(-1, self.n_steps_out))
        val_predict = self.mmy.inverse_transform(self.model.predict(X_val))
        valY = self.mmy.inverse_transform(y_val.reshape(-1, self.n_steps_out))

        # Calculate the following: Root Mean Squared Error, Mean absolute error and Mean Absolute Percentage Error
        self.trainRMSE_Score = math.sqrt(mean_squared_error(trainY[:, 0], self.trainPredict[:, 0]))
        self.trainMAE_Score = mean_absolute_error(trainY[:, 0], self.trainPredict[:, 0])

        self.valRMSE_Score = math.sqrt(mean_squared_error(valY[:, 0], self.valPredict[:, 0]))
        self.valMAE_Score = mean_absolute_error(valY[:, 0], self.valPredict[:, 0])

        print('Train Score: %.2f RMSE, ' % self.trainRMSE_Score + '%.2f MAE' % self.trainMAE_Score)
        print('Validation Score: %.2f RMSE, ' % self.valRMSE_Score + '%.2f MAE' % self.valMAE_Score)

        # Return the model and the scalers
        return self

    def PredictTestSample(self, dataName, start, end):
        """

        """
        start = str(pd.to_datetime(start) - pd.Timedelta(1, "D"))

        # Import the data
        if dataName == "Caltech":
            df = ImportEV().getCaltech(start_date=start, end_date=end)
        elif dataName == "JPL":
            df = ImportEV().getJPL(start_date=start, end_date=end)
        elif dataName == "Office":
            df = ImportEV().getOffice(start_date=start, end_date=end)
        else:
            print("Error, data parameter should be Caltech, JPL or Office")

        total = createTransformation(df, start, end).getTotalData()

        # Create Input and Target Features
        X, y = self.totalData.copy(), self.totalData[self.feature_name]

        # Scale the Data
        X_trans = self.scaler.transform(X)
        y_trans = y

        # Split the data for prediction in the RNN models
        X_test, self.y_test = split_sequences(X_trans, y_trans, self.n_steps_in, self.n_steps_out)

        # Make and Invert predictions
        self.test_predict = self.model.predict(X_test)

        self.testScore = math.sqrt(mean_squared_error(self.y_test[:, -1], self.test_predict[:, -1]))

        return self

    def PlotTestSample(self):
        # plot baseline and predictions
        plt.plot(self.y_test[:, 0])
        plt.plot(self.test_predict[:, 0])
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
    start, end = "2018-05-01", "2018-11-01"
    df = ImportEV().getCaltech(start_date=start, end_date=end, removeUsers=False)
    Total_df = createTransformation(df, start, end).remove_outliers().getTotalData()

    model = totalModel(Total_df).create_model(type="LSTM")
    model = model.PredictTestSample("Caltech", "2018-11-01", "2018-12-01")
    print(model.trainScore)
    print(model.valScore)
    print(model.testScore)

    model.PlotLoss()

    model.PlotTestSample()
