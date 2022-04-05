import math
import numpy as np
from P6_code.FinishedCode.importData import ImportEV
from P6_code.FinishedCode.dataTransformation import createUsers
from P6_code.FinishedCode.functions import split_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#Ikke sikkert den skal bruges
def model_configs():
	# define scope of configs
	n_steps_in = [5, 8, 10]
	n_nodes = [4, 8, 16]
	n_batch = [1, 20, 150]

	# create configs
	configs = list()

	for i in n_steps_in:
		for j in n_nodes:
			for k in n_batch:
				cfg = [i, j, k]
				configs.append(cfg)

	print('Total configs: %d' % len(configs))

	return configs

class Model:
	def __init__(self):
		#Variables to create the model
		self.train_start = "2018-06-01"
		self.train_end = "2018-11-09"
		self.userSampleLimit = 25
		self.val_split = 0.2
		self.target_feature = 'chargingTime'

		#Scalers
		self.ss = StandardScaler()
		self.mm = MinMaxScaler(feature_range=(0, 1))

		#Model Hyperparameters (configs)
		self.model = Sequential()
		self.n_steps_in = 10
		self.n_nodes = 4
		self.n_batch = 20
		self.epochs = 20


	def create_model(self, type="LSTM"):
		df_train = ImportEV().getCaltech(start_date=self.train_start, end_date=self.train_end, removeUsers=True, userSampleLimit=self.userSampleLimit)
		users = createUsers(df_train, self.train_start, self.train_end)
		usersID = users.data.userID.unique()
		users_df = []

		for user in usersID:
			users_df.append(users.getUserData(user=user))

		#Create Input and Target Features
		X, Y = [], []

		for user in users_df:
			Y.append(user[self.target_feature])
			X.append(user.drop(columns=[self.target_feature]))

		lenInputFeature = len(X[0].columns)

		#Scale the Data
		X_scaled = []
		Y_scaled = []

		for user in X:
			self.ss.fit(user)
		for user in X:
			X_scaled.append(self.ss.transform(user))
		for user in Y:
			self.mm.fit(user.values.reshape(-1, 1))
		for user in Y:
			Y_scaled.append(self.mm.transform(user.values.reshape(-1, 1)))

		#Split the data into training and validation data
		total_samples = len(X[0])
		train_test_cutoff = round(self.val_split * total_samples)

		X_train, X_val = [], []
		Y_train, Y_val = [], []

		for user in range(len(Y_scaled)):
			user_X, user_Y = split_sequences(X_scaled[user], Y_scaled[user], 10, lenInputFeature)

			X_train.append(user_X[:-train_test_cutoff])
			X_val.append(user_X[-train_test_cutoff:])

			Y_train.append(user_Y[:-train_test_cutoff])
			Y_val.append(user_Y[-train_test_cutoff:])

		#Create the model
		if type == "LSTM":
			self.model.add(LSTM(self.n_nodes, input_shape=(self.n_steps_in, lenInputFeature)))  # todo: add inputs steps
		elif type == "GRU":
			self.model.add(GRU(self.n_nodes, input_shape=(self.n_steps_in, lenInputFeature)))  # todo: add inputs steps
		else:
			raise Exception("The type of the model should either be LSTM or GRU")

		self.model.add(Dense(1))
		self.model.compile(optimizer='adam', loss='mse')

		#Fit the data and trains the model
		for i in range(len(X_train)):
			self.model.fit(x=X_train[i], y=Y_train[i], epochs=self.epochs, batch_size=self.n_batch, verbose=2)

		#Make and Invert predictions
		train_predict, val_predict = [], []
		self.trainScore, self.valScore = [], []

		for i in range(len(X_train)):
			train_predict.append(self.mm.inverse_transform(self.model.predict(X_train[i]).reshape(-1, 1)))
			val_predict.append(self.mm.inverse_transform(self.model.predict(X_val[i]).reshape(-1, 1)))

			# calculate root mean squared error
			self.trainScore.append(math.sqrt(mean_squared_error(Y_train[i][:, 0], train_predict[i][:, 0])))
			self.valScore.append(math.sqrt(mean_squared_error(Y_val[i][:, 0], val_predict[i][:, 0])))

		#Return the model and the scalers
		return self

	def PredictTestSample(self, start, end, userSampleLimit):
		#Import the data
		df_test = ImportEV().getCaltech(start_date=start, end_date=end, removeUsers=True, userSampleLimit=userSampleLimit)
		users = createUsers(df_test, start, end)

		#Save the user_ids for return
		user_id = users.data.userID.unique()
		user_df_test = []

		for user in user_id:
			user_df_test.append(users.getUserData(user=user))

		# Create Input and Target Features
		X_test, Y_test = [], []

		for user in user_df_test:
			Y_test.append(user[self.target_feature])
			X_test.append(user.drop(columns=[self.target_feature]))

		# Scale the Data
		X_test_scaled = []

		for user in X_test:
			X_test_scaled.append(self.ss.transform(user))

		#Split the data for prediction in the RNN models
		users_test_X, users_test_Y = [], []

		for user in range(len(X_test_scaled)):
			user_test_X, user_test_Y = split_sequences(X_test_scaled[user], np.array(Y_test[user]).reshape(-1, 1), 10, 6)
			users_test_X.append(user_test_X)
			users_test_Y.append(user_test_Y)

		#Predict the data
		test_predict = []
		self.testScore = []

		# Make and Invert predictions
		for i in range(len(users_test_X)):
			test_predict.append(self.mm.inverse_transform(self.model.predict(users_test_X[i]).reshape(-1, 1)))

			# calculate root mean squared error
			self.testScore.append(math.sqrt(mean_squared_error(users_test_Y[i][:, 0], test_predict[i][:, 0])))

		return self

if __name__ == "__main__":
	#The model will always be first input
	model = Model().create_model()
	model = model.PredictTestSample("2018-11-09", "2019-01-01", 15)
	print(model.trainScore)
	print(model.valScore)
	print(model.testScore)


	""" Ikke sikkert det skal bruges
	cfg_list = model_configs()
	results = pd.DataFrame(
		columns=["userID", "n_steps_in", "n_nodes", "n_batch", "errorTrain", "errorTest"])
	index = 0
	"""
