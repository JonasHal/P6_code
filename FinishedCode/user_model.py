import math
import matplotlib.pyplot as plt
import numpy as np
from P6_code.FinishedCode.importData import ImportEV
from P6_code.FinishedCode.dataTransformation import createTransformation
from P6_code.FinishedCode.functions import split_sequences, getModelStructure
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)

class userModel:
    """A Model class used to predict one given feature from the total dataframe.
    @param data: The data to create the model from. Has to be from:
    createTransformation(*params).getUserData()
    @return: The model object, call createModel() to fit it.
    """

    def __init__(self, data):
        # Variables and defining loss evaluation to create the model
        self.userData = data
        self.val_split = 0.2
        self.target_feature = "kWhDelivered"
        self.drop_feature = 'chargingTime'
        self.lossType = "mean_squared_error"

        # Scaler
        self.mmX = MinMaxScaler(feature_range=(0, 1))
        self.mmy = MinMaxScaler(feature_range=(0, 1))

        # Model Hyperparameters (configs)
        self.n_steps_in = 3
        self.n_steps_out = 2
        self.n_nodes = 20
        self.n_nodes_cnn = 128

        self.batch_size = 25
        self.epochs = 250

    def createModel(self, type="LSTM", layers=1):
        """Creates the model with the given type and fits the data.
        @param type: The type of model that should be created. Can be the following:
        LSTM, GRU, CNN or LSTM-CNN

        @return: The model object, with a fitted model, which can be used for prediction.
        """
        # Scale the Data
        X, y = self.userData.drop(columns=self.drop_feature), self.userData[self.target_feature]
        print("The input features are: " + str(X.columns))

        X_trans = self.mmX.fit_transform(X)
        self.y_trans = self.mmy.fit_transform(y.values.reshape(-1, 1))

        # Split the data into training and validation data
        total_samples = len(X)
        train_val_cutoff = round(self.val_split * total_samples)

        X_scaled, y_scaled = split_sequences(X_trans, self.y_trans, self.n_steps_in, self.n_steps_out)

        # Info about the input features
        self.n_features = X_scaled.shape[2]

        X_train, X_val = X_scaled[:-train_val_cutoff], X_scaled[-train_val_cutoff:]
        y_train, y_val = y_scaled[:-train_val_cutoff], y_scaled[-train_val_cutoff:]

        # Create the model
        self.title = type, layers
        self.model = getModelStructure(type, layers, self.n_steps_in, self.n_steps_out, self.n_features, self.n_nodes, self.n_nodes_cnn)

        # Printing the Structure of the model and compile it
        print(self.model.summary())
        self.model.compile(loss=self.lossType, optimizer='adam')

        # Fit the data and trains the model
        self.history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0, validation_data=(X_val, y_val))

        # Make and Invert predictions
        self.trainPredict = self.mmy.inverse_transform(self.model.predict(X_train))
        trainY = self.mmy.inverse_transform(y_train.reshape(-1, self.n_steps_out))
        self.valPredict = self.mmy.inverse_transform(self.model.predict(X_val))
        valY = self.mmy.inverse_transform(y_val.reshape(-1, self.n_steps_out))

        # Calculate the following: Root Mean Squared Error, Mean absolute error and Mean Absolute Percentage Error
        self.trainRMSE_Score = math.sqrt(mean_squared_error(trainY[:, 0], self.trainPredict[:, 0]))
        self.trainMAE_Score = mean_absolute_error(trainY[:, 0], self.trainPredict[:, 0])

        self.valRMSE_Score = math.sqrt(mean_squared_error(valY[:, 0], self.valPredict[:, 0]))
        self.valMAE_Score = mean_absolute_error(valY[:, 0], self.valPredict[:, 0])

        print('Train Score: %.2f RMSE, ' % self.trainRMSE_Score + '%.2f MAE'% self.trainMAE_Score)
        print('Validation Score: %.2f RMSE, ' % self.valRMSE_Score + '%.2f MAE' % self.valMAE_Score)

        # Return the model and the scalers
        return self

    def PlotLoss(self):
        """Note: Should be run after creating the model
        Makes a graph with the loss from each epoch when fitting the model.
        """
        loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]
        plt.figure(figsize=(5, 4))
        plt.plot(loss, label="train_loss", linewidth=2.5)
        plt.plot(val_loss, label="val_loss", linewidth=2.5)
        #plt.title(self.title, fontsize=20)
        plt.xlabel('Epochs', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
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
        plt.title(self.title, fontsize=14)
        plt.plot(self.mmy.inverse_transform(self.y_trans))
        plt.plot(trainPredictPlot)
        plt.plot(valPredictPlot)
        plt.show()


if __name__ == "__main__":
    start, end = "2018-08-01", "2018-11-01"
    df = ImportEV().getCaltech(start_date=start, end_date=end, removeUsers=True, userSampleLimit=25)
    Users = createTransformation(df, start, end)
    User_61 = Users.getUserData(user="000000061")

    LSTM_CNN = userModel(User_61).createModel("LSTM", layers=2)

    LSTM_CNN.PlotLoss()

    LSTM_CNN.PlotTrainVal()
    print('doneeeeeeeeee')