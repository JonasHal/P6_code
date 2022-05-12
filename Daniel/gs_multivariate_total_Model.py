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

import warnings
warnings.filterwarnings("ignore")


class mtotalModel:
    """A Model class used to predict multivariate data from the total dataframe.
    @param data: The data to create the model from. Has to be from:
    createTransformation(*params).getTotalData()
    @return: The model object, call createModel() to fit it.
    """
    def __init__(self, data, n_steps_in, n_nodes):
        # Variables to create the model
        self.totalData = data
        self.val_split = 0.2

        # Scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Model Hyperparameters (configs)
        self.model = Sequential()
        self.n_steps_in = n_steps_in
        self.n_steps_out = 50
        self.n_nodes = n_nodes

        self.batch_size = 25
        self.epochs = 200

    def createModel(self, type="LSTM"):
        """Creates the model with the given type and fits the data.
        @param type: The type of model that should be created. Can be the following:
        LSTM, GRU, CNN or LSTM-CNN

        @return: The model object, with a fitted model, which can be used for prediction.
        """
        #print("Making Model")
        # Create Input and Target Features
        X, Y = self.totalData.copy(), self.totalData.copy()

        #Scale the data
        self.scaler = self.scaler.fit(X)
        X_trans = self.scaler.transform(X)
        Y_trans = self.scaler.transform(Y)

        # Info about the input features
        #print("The input features are: " + str(X.columns))
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
        #print(self.model.summary())
        self.model.compile(optimizer='adam', loss='mse', metrics=["mean_absolute_error"])

        # Fit the data and trains the model
        self.history = self.model.fit(x=X_train, y=Y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=0,
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

        plt.clf()
        plt.plot(loss, label="train_loss")
        plt.plot(val_loss, label="val_loss")
        plt.title(self.title + ', n_in: ' + str(self.n_steps_in) + ', n_nodes: ' + str(self.n_nodes))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('mvtm/' + self.title + '_n_steps_in' + str(self.n_steps_in) + '_n_nodes' + str(self.n_nodes))


if __name__ == "__main__":
    # The model will always be first input
    start, end = "2018-08-01", "2018-11-01"
    df = ImportEV().getCaltech(start_date=start, end_date=end, removeUsers=False)
    Total_df = createTransformation(df, start, end).remove_outliers().getTotalData()

    grid_df = pd.DataFrame(columns=['model type', 'n_steps_in', 'n_nodes', 'train', 'val'])

    for model_type in ["LSTM", "GRU"]:
        for n_steps_in in [3, 15, 50]:
            for n_nodes in [5, 50, 100]:
                model = mtotalModel(Total_df, n_steps_in, n_nodes).createModel(type=model_type)

                grid_df = grid_df.append({'model type': model_type, 'n_steps_in': n_steps_in, 'n_nodes': n_nodes,
                                          'train': model.trainScore, 'val': model.valScore
                                          }, ignore_index=True)
                model.PlotLoss()
                print({'model type': model_type, 'n_steps_in': n_steps_in, 'n_nodes': n_nodes,
                                          'train': model.trainScore, 'val': model.valScore
                                          })

    print(grid_df.to_string())
    grid_df.to_csv('grid_df.csv')

    # for i in range(model.n_features):
    #     model.PlotTestSample(column_to_predict=i)
