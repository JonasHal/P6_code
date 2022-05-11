import math
import numpy as np
import pandas as pd
from P6_code.FinishedCode.importData import ImportEV
from P6_code.FinishedCode.dataTransformation import createTransformation
from P6_code.FinishedCode.functions import split_sequences, getModelStructure
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class usersModel:
	"""A Model class used to predict one given feature from the total dataframe.
	@param data: The data to create the model from. Has to be from:
    createTransformation(*params)
	@return: The model object, call createModel() to fit it.
	"""
	def __init__(self, data, n_steps_in, n_nodes):
		# Variables and defining loss evaluation to create the model
		self.usersData = data
		self.val_split = 0.2
		self.target_feature = "kWhDelivered"
		self.drop_feature = 'chargingTime'
		self.lossType = "mean_squared_error"


		#Scalers
		self.mmX = MinMaxScaler(feature_range=(0, 1))
		self.mmy = MinMaxScaler(feature_range=(0, 1))

		#Model Hyperparameters (configs)
		self.n_steps_in = n_steps_in
		self.n_steps_out = 2
		self.n_nodes = n_nodes
		self.n_nodes_cnn = n_nodes

		self.batch_size = 25
		self.epochs = 200

	def createModel(self, type="LSTM", layers=1):
		"""Creates the model with the given type and fits the data.
		@param type: The type of model that should be created. Can be the following:
		LSTM, GRU, CNN or LSTM-CNN

		@return: The model object, with a fitted model, which can be used for prediction.
		"""
		# Find all the unique userID
		usersID = self.usersData.data.userID.unique()
		users_df = []

		for user in usersID:
			users_df.append(self.usersData.getUserData(user=user))

		#Create Input and Target Features
		X, Y = [], []

		for user in users_df:
			Y.append(user[self.target_feature])
			X.append(user.drop(columns=[self.drop_feature]))

		#Info about the input features
		print("The input features are: " + str(X[0].columns))
		self.n_features = len(X[0].columns)

		#Scale the Data
		X_scaled = []
		Y_scaled = []

		for user in X:
			self.mmX.fit(user)
		for user in X:
			X_scaled.append(self.mmX.transform(user))
		for user in Y:
			self.mmy.fit(user.values.reshape(-1, 1))
		for user in Y:
			Y_scaled.append(self.mmy.transform(user.values.reshape(-1, 1)))

		#Split the data into training and validation data
		total_samples = len(X[0])
		train_val_cutoff = round(self.val_split * total_samples)

		X_train, X_val = [], []
		Y_train, Y_val = [], []

		for user in range(len(Y_scaled)):
			user_X, user_Y = split_sequences(X_scaled[user], Y_scaled[user], self.n_steps_in, self.n_steps_out)

			X_train.append(user_X[:-train_val_cutoff])
			X_val.append(user_X[-train_val_cutoff:])

			Y_train.append(user_Y[:-train_val_cutoff])
			Y_val.append(user_Y[-train_val_cutoff:])

		# Create the model
		print(self.n_features)
		self.title = '{type} layer:{layers}, n_nodes:{nodes} n_steps{steps_in}'.format(type=type, layers=layers, nodes=n_nodes, steps_in=n_steps_in)
		self.model = getModelStructure(type, layers, self.n_steps_in, self.n_steps_out, self.n_features, self.n_nodes, self.n_nodes_cnn)

		# Printing the Structure of the model and compile it
		print(self.model.summary())
		self.model.compile(loss=self.lossType, optimizer='adam')

		#Fit the data and trains the model
		progress = 0
		self.history = []
		for i in range(len(X_train)):
			self.history.append(self.model.fit(x=X_train[i], y=Y_train[i], batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_data=(X_val[i], Y_val[i])))
			progress += 1
			print("Number of Users trained on: " + str(progress) + "/" + str(len(usersID)))

		#Make and Invert predictions
		train_predict, val_predict = [], []
		self.trainScore, self.valScore = [], []

		for i in range(len(X_train)):
			train_predict.append(self.mmy.inverse_transform(self.model.predict(X_train[i]).reshape(-1, self.n_steps_out)))
			val_predict.append(self.mmy.inverse_transform(self.model.predict(X_val[i]).reshape(-1, self.n_steps_out)))

			# calculate root mean squared error
			self.trainScore.append(math.sqrt(mean_squared_error(Y_train[i][:, 0], train_predict[i][:, 0])))
			self.valScore.append(math.sqrt(mean_squared_error(Y_val[i][:, 0], val_predict[i][:, 0])))

		#Return the model and the scalers
		return self

	def PredictTestSample(self, dataName, start, end, userSampleLimit):
		"""Note: Should be run after creating the model
		Predicts a given timeframe from a given dataset.
		@param dataName: The dataset, where the prediction should take place
		@param start: The start date of the prediction
		@param end: The end date of the prediction
		@param userSampleLimit: Lower bound for the least amount of charges a user has made.

		@return: The model object, with the prediction results.
		"""
		# Import the data
		if dataName == "Caltech":
			df = ImportEV().getCaltech(start_date=start, end_date=end, removeUsers=True, userSampleLimit=userSampleLimit)
		elif dataName == "JPL":
			df = ImportEV().getJPL(start_date=start, end_date=end, removeUsers=True, userSampleLimit=userSampleLimit)
		elif dataName == "Office":
			df = ImportEV().getOffice(start_date=start, end_date=end, removeUsers=True, userSampleLimit=userSampleLimit)
		elif dataName == "Both":
			df = ImportEV().getBoth(start_date=start, end_date=end, removeUsers=True, userSampleLimit=userSampleLimit)
		else:
			print("Error, data parameter should be Caltech, JPL, Both or Office")

		users = createTransformation(df, start, end)

		#Save the user_ids for return
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
			X_test_scaled.append(self.mmX.transform(user))

		#Split the data for prediction in the RNN models
		users_test_X, self.users_test_Y = [], []

		for user in range(len(X_test_scaled)):
			user_test_X, user_test_Y = split_sequences(X_test_scaled[user], np.array(Y_test[user]).reshape(-1, 1), self.n_steps_in, self.n_steps_out)
			users_test_X.append(user_test_X)
			self.users_test_Y.append(user_test_Y)

		#Predict the data
		self.test_predict = []
		self.testScore = []

		# Make and Invert predictions
		for user in range(len(users_test_X)):
			self.test_predict.append(self.mmy.inverse_transform(self.model.predict(users_test_X[user]).reshape(-1, self.n_steps_out)))

			# calculate root mean squared error
			self.testScore.append(math.sqrt(mean_squared_error(self.users_test_Y[user][:, 0], self.test_predict[user][:, 0])))

		return self

	def PlotTestSample(self, user=0):
		"""Note: Should be run after making a test prediction
		Makes a graph with the predictions and real values on the test data.
		@param user: The index of the user to draw the plot from.
		"""
		# plot baseline and predictions
		plt.title(self.title)
		plt.plot(self.users_test_Y[user][:, 0])
		plt.plot(self.test_predict[user][:, 0])
		plt.show()

	def PlotLoss(self):
		"""Note: Should be run after creating the model
        Makes a graph with the loss from each epoch when fitting the model.
        """
		loss = [x.history["loss"] for x in self.history]
		val_loss = [x.history["val_loss"] for x in self.history]

		loss = [sum(x) / len(x) for x in zip(*loss)]
		val_loss = [sum(x) / len(x) for x in zip(*val_loss)]

		plt.plot(loss, label="train_loss")
		plt.plot(val_loss, label="val_loss")
		plt.title(self.title)
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.show()


if __name__ == "__main__":
	start, end = "2018-08-01", "2018-12-01"
	df = ImportEV().getBoth(start_date=start, end_date=end, removeUsers=True, userSampleLimit=30)
	Users = createTransformation(df, start, end)


	grid_df = pd.DataFrame(columns=['model type', 'n_steps_in', 'n_nodes', 'train', 'val'])

	for model_type in ["LSTM", "GRU", 'CNN']:
		for n_steps_in in [3, 15, 50]:
			for n_nodes in [5, 50, 64]:
				for layers in [1, 2]:
					model = usersModel(Users, n_steps_in, n_nodes).createModel(type=model_type)
					grid_df = grid_df.append({'model type': model_type, 'n_steps_in': n_steps_in, 'n_nodes': n_nodes,'train': model.trainScore, 'val': model.valScore }, ignore_index=True)
					model.PlotLoss()
					print({'model type': model_type, 'n_steps_in': n_steps_in, 'n_nodes': n_nodes,'train': model.trainScore, 'val': model.valScore})

	print(grid_df.to_string())
	grid_df.to_csv('grid_df.csv')