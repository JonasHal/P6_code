import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from P6_code.FinishedCode.importData import ImportEV
from P6_code.FinishedCode.dataTransformation import createUsers

from keras.models import Sequential
from keras.layers import Dense, GRU
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# create a list of configs to try
def model_configs():
	# define scope of configs
	n_steps_in = [8, 10]
	n_steps_out = [5]
	n_nodes = [4, 8]
	n_epochs = [50]
	n_batch = [1, 150]

	# create configs
	configs = list()

	for i in n_steps_in:
		for j in n_steps_out:
			for k in n_nodes:
				for l in n_epochs:
					for m in n_batch:
						cfg = [i, j, k, l, m]
						configs.append(cfg)

	print('Total configs: %d' % len(configs))

	return configs

def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
	X, y = list(), list() # instantiate X and y

	for i in range(len(input_sequences)):
		# find the end of the input, output sequence
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out - 1
		# check if we are beyond the dataset
		if out_end_ix > len(input_sequences):
			break

		# gather input and output of the pattern
		seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
		X.append(seq_x), y.append(seq_y)
	return np.array(X), np.array(y)

# fit a model
def model_score(X_trans, y_trans, config):
	# unpack config
	n_steps_in, n_steps_out, n_nodes, n_epochs, n_batch = config
	total_samples = len(X_trans)
	train_test_cutoff = round(0.20 * total_samples)

	# transform series into supervised format
	X_ss, y_mm = split_sequences(X_trans, y_trans, n_steps_in, n_steps_out)

	trainX, testX = X_ss[:-train_test_cutoff], X_ss[-train_test_cutoff:]
	trainY, testY = y_mm[:-train_test_cutoff], y_mm[-train_test_cutoff:]

	print(X_ss.shape)
	# define model
	model = Sequential()
	model.add(GRU(n_nodes, input_shape=(n_steps_in, n_steps_out)))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')

	# fit model
	model.fit(trainX, trainY, epochs=n_epochs, batch_size=n_batch, verbose=2)

	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)

	# invert predictions
	trainPredict = mm.inverse_transform(trainPredict.reshape(-1, 1))
	trainY = mm.inverse_transform(trainY)
	testPredict = mm.inverse_transform(testPredict.reshape(-1, 1))
	testY = mm.inverse_transform(testY)

	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[:, 0], trainPredict[:, 0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:, 0]))
	print('Test Score: %.2f RMSE' % (testScore))

	return config, trainScore, testScore

# grid search configs
def grid_search(X_trans, y_trans, cfg_list):
	# evaluate configs
	scores = [model_score(X_trans, y_trans, cfg) for cfg in cfg_list]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[2])
	return scores

if __name__ == "__main__":
	start, end = "2018-09-01", "2018-11-09"
	df = ImportEV().getCaltech(start_date=start, end_date=end, removeUsers=True, userSampleLimit=25)
	Users = createUsers(df, start, end)
	print(Users.data.userID.unique())
	userID = ['000000061', '000000022', '000000324', '000000066']

	cfg_list = model_configs()
	results = pd.DataFrame(columns=["userID", "n_steps_in", "n_steps_out", "n_nodes", "n_epochs", "n_batch", "errorTrain", "errorTest"])
	print(results)
	index = 0

	for i in userID:
		User_61 = Users.getUserData(user=i)

		ss = StandardScaler()
		mm = MinMaxScaler(feature_range=(0, 1))

		X, y = User_61.drop(columns=['kWhDelivered']), User_61.kWhDelivered

		X_trans = ss.fit_transform(X)
		y_trans = mm.fit_transform(y.values.reshape(-1, 1))

		scores = grid_search(X_trans, y_trans, cfg_list)

		# list top 3 configs
		for cfg, errorTrain, errorTest in scores[:3]:
			print(cfg, errorTrain, errorTest)
			results.loc[index] = [i, cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], round(errorTrain, 2), round(errorTest, 2)]
			index += 1

	print(results.to_string())