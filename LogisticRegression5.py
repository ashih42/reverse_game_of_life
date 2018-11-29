from DataParser import DataParser
from MatrixDataParser import MatrixDataParser
from exceptions import LogisticRegressionException

from colorama import Fore, Back, Style
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import shutil


import fast_wrangle



'''
m = number of training examples		= 100 ~ 50,000
n = number of features				= 123
'''

'''
This class tries to solve an independent model for each of 400 outputs.
Appears to work well for each output, but combined together, very few training examples have all 400 predictions correct.
'''

def clamp(x, low, high):
	return max(0, min(x, high))


def ft_exp(x):
	try:
		answer = math.exp(x)
	except OverflowError:
		answer = float('inf')
	return answer

class LogisticRegression5:
	np.random.seed(42)
	# __NUM_FEATURES = 11*11 + 2
	__NUM_FEATURES = 11*11 + 2
	__NUM_LABELS = 400

	# __NUM_FEATURES = 402			# try with just the given data, no new features

	__ALPHA = 0.00001
	# __ALPHA = 0.001
	# __ALPHA = 0.0001
	
	# __MAX_EPOCHS = 20
	__MAX_EPOCHS = 10

	__IS_VERBOSE = os.getenv('RGOL_VERBOSE') == 'TRUE'
	__SHOW_PLOTS = os.getenv('RGOL_SHOW_PLOTS') == 'TRUE'
	__SAVE_PLOTS = os.getenv('RGOL_SAVE_PLOTS') == 'TRUE'

	__PLOTS_DIRECTORY = 'Plots/'

	__SIGMOID = np.vectorize(lambda x: 1 / (1 + ft_exp(-x)))
	__PREDICT = np.vectorize(lambda x : 1 if x > 0 else 0)

	__BATCH_SIZE = 100
	
	def __init__(self, param_filename=None):
		if param_filename is not None:
			# print('self.__NUM_FEATURES = %d', self.__NUM_FEATURES)
			parser = MatrixDataParser(param_filename, num_rows=self.__NUM_FEATURES, num_cols=400)
			self.__theta = np.array(parser.data, dtype = float)

	def process_data(self, filename):
		parser = DataParser(filename, is_training_data=True)
		X, Y = self.__wrangle_data(parser.data, is_training_data=True)
		self.__write_processed_data_to_file(X, 'processed_X.dat')
		self.__write_processed_data_to_file(Y, 'processed_Y.dat')

	def train_processed_data(self, X_filename, Y_filename):
		X_parser = MatrixDataParser(X_filename, num_rows=None, num_cols=self.__NUM_FEATURES)
		X = np.array(X_parser.data, dtype = float)
		m = X.shape[0]
		Y_parser = MatrixDataParser(Y_filename, num_rows=m, num_cols=self.__NUM_LABELS)
		Y = np.array(Y_parser.data, dtype = float)
		X_train, Y_train, X_cv, Y_cv = self.__generate_train_cv_sets(X, Y)
		self.__train(X_train, Y_train, X_cv, Y_cv)

	def train(self, filename):
		parser = DataParser(filename, is_training_data=True)
		X, Y = self.__wrangle_data(parser.data, is_training_data=True)

		# X_train, Y_train, X_cv, Y_cv = self.__generate_train_cv_sets(X, Y)
		# self.__train(X_train, Y_train, X_cv, Y_cv)

		self.__train(X, Y)
		

	def predict(self, filename):
		parser = DataParser(filename, is_training_data=False)
		test_data = np.array(parser.data, dtype = float)
		X_test, _ = self.__wrangle_data(parser.data, is_training_data=False)
		

		# predictions = self.__PREDICT(X_test @ self.__theta)
		
		# print('X_test.shape = ', X_test.shape)

		# print('theta.shape = ', self.__theta.shape)

		predictions = np.empty((X_test.shape[1], 400))
		for i in range(400):
			X_test_subset = X_test[i, :, :]

			# theta_col = self.__theta[:, i].reshape(self.__theta.shape[0], 1)
			theta_col = self.__theta[:, i:i+1]
			# theta_col = theta_col[:, np.newaxis]

			# print('X_test_subset.shape = ', X_test_subset.shape)
			predictions[:, i] = self.__PREDICT(X_test_subset @ theta_col).flatten()

		self.__write_predictions_to_file(predictions)

	def __wrangle_data(self, data, is_training_data):
		print('Wrangling data...')
		data = np.array(data, dtype=float)

		m = data.shape[0]

		if is_training_data:
			X = data[:, 401:802]						# grab 'stop cell' values
		else:
			X = data[:, 1:401]								

		X = np.c_[data[:, 0], X]						# grab 'delta' value in index 0
		X[:, 0] = (X[:, 0] - 3) / 2						# feature scaling from range [1 ... 5] to [-1 ... 1]

		# X = self.__format_X(X)
		X = fast_wrangle.format_X(X)

		print('finished formatting X')

		if is_training_data:
			Y = data[:, 1:401]							# grab 'start cell' values in index 1 - 400
		else:
			Y = None

		# print('X.shape = ', X.shape)
		# print('Y.shape = ', Y.shape)

		return X, Y

	# def __format_Y(self, Y):
	# 	Y_new = Y.reshape(Y.shape[0] * Y.shape[1], 1)
	# 	return Y_new
	
	# def __format_X(self, X):
	# 	X_new = np.empty((0, 1 + 121))

	# 	for i in range(X.shape[0]):								# THIS LOOP IS REALLY SLOW

	# 		if self.__IS_VERBOSE:
	# 			print('Formatting entry %d / %d' % (i, X.shape[0]))

	# 		row_data = np.empty((400, 122))

	# 		delta = X[i, 0]
			
	# 		board = X[i, 1:].reshape((20, 20))
			
	# 		# extend the board with 5 cell margins around borders
	# 		board = np.c_[ np.zeros((20, 5)), board, np.zeros((20, 5))]
	# 		board = np.r_[ np.zeros((5, 30)), board, np.zeros((5, 30))]

	# 		for x in range(20):
	# 			for y in range(20):
	# 				x_origin = x + 5
	# 				y_origin = y + 5
	# 				x_low = clamp(x_origin - 5, 0, 29)
	# 				x_high = clamp(x_origin + 5, 0, 29)
	# 				y_low = clamp(y_origin - 5, 0, 29)
	# 				y_high = clamp(y_origin + 5, 0, 29)

	# 				sub_board = board[x_low:x_high+1, y_low:y_high+1]

	# 				row_index = x * 20 + y
	# 				row_data[row_index, 1:] = sub_board.reshape((1, 121))

	# 		row_data[:, 0] = delta
			
	# 		X_new = np.append(X_new, row_data, axis=0)

	# 	X_new = np.c_[np.ones(X_new.shape[0]), X_new]				# add column of 1s

	# 	print('X_new.shape = ', X_new.shape)

	# 	return X_new


	def __generate_train_cv_sets(self, X, Y, percent_train=0.7):
		# return fast_wrangle.get_train_cv_sets(X, Y, percent_train)

		m = X.shape[1]
		
		split_index = int(m * percent_train)
		
		X_train = X[:, :split_index, :]
		X_cv = X[:, split_index:, :]

		Y_train = Y[:split_index, :]
		Y_cv = Y[split_index:, :]

		return X_train, Y_train, X_cv, Y_cv

		# m = X.shape[1]
		
		# indices = np.arange(m)
		# np.random.shuffle(indices)

		# split_index = int(m * percent_train)
		# train_indices = indices[:split_index]
		# cv_indices = indices[split_index:]

		# X_train = X[:, train_indices, :]
		# X_cv = X[:, cv_indices, :]

		# Y_train = Y[train_indices, :]
		# Y_cv = Y[cv_indices, :]

		# return X_train, Y_train, X_cv, Y_cv

	
	# def __generate_train_cv_sets(self, X, Y, percent_train=0.7):
	# 	m = X.shape[0]
	# 	indices = np.arange(m)
	# 	np.random.shuffle(indices)

	# 	X_temp = X[indices, :]
	# 	Y_temp = Y[indices, :]

	# 	split_row_index = int(m * percent_train)

	# 	X_train = X_temp[:split_row_index, :]
	# 	Y_train = Y_temp[:split_row_index, :]

	# 	X_cv = X_temp[split_row_index:, :]
	# 	Y_cv = Y_temp[split_row_index:, :]

	# 	return X_train, Y_train, X_cv, Y_cv


	def __train(self, X_train, Y_train, X_cv=None, Y_cv=None):
		try:
			shutil.rmtree(self.__PLOTS_DIRECTORY)
		except IOError as e:
			pass

		if X_cv is None or Y_cv is None:
			self.__is_cv = False;
		else:
			self.__is_cv = True;

		self.__init_plots()
		self.__theta = np.zeros((self.__NUM_FEATURES, self.__NUM_LABELS))
		for i in range(self.__NUM_LABELS):
			self.__current_label = i + 1
			print('Training on label: Start Cell %d...' % self.__current_label)
			self.__init_lists()

			X_train_subset = X_train[i, :, :]

			Y_train_col = Y_train[:, i:i+1]
			# Y_train_col = Y_train_col.reshape(Y_train_col.shape[0], 1)
			# Y_train_col = Y_train_col[:, np.newaxis]

			if self.__is_cv:
				X_cv_subset = X_cv[i, :, :]

				Y_cv_col = Y_cv[:, i:i+1]
				# Y_cv_col = Y_cv_col.reshape(Y_cv_col.shape[0], 1)
				# Y_cv_col = Y_cv_col[:, np.newaxis]
			else:
				X_cv_subset = None
				Y_cv_col = None
			
			theta_col = self.__run_gradient_descent(X_train_subset, Y_train_col, X_cv_subset, Y_cv_col, batch_size=100)			# mini batch size = 100
			self.__theta[:, i] = theta_col.flatten()

			if self.__SAVE_PLOTS:
				self.__update_plots()
				self.__save_plots()

		self.__write_parameters_to_file()

	def __run_gradient_descent(self, X_train, Y_train, X_cv, Y_cv, batch_size):
		theta = np.zeros((self.__NUM_FEATURES, 1))
		self.__epoch = 0

		for i in range(self.__MAX_EPOCHS):
			self.__epoch += 1

			j_limit = X_train.shape[0] // self.__BATCH_SIZE

			for j in range(j_limit):
				X_batch = X_train[j*self.__BATCH_SIZE:(j+1)*self.__BATCH_SIZE, :]
				Y_batch = Y_train[j*self.__BATCH_SIZE:(j+1)*self.__BATCH_SIZE, :]
				theta = theta - self.__ALPHA * X_batch.T @ (self.__SIGMOID(X_batch @ theta) - Y_batch)

			# X_batch, Y_batch = self.__select_x_y_batch(X_train, Y_train, batch_size)
			# theta = theta - self.__ALPHA * X_batch.T @ (self.__SIGMOID(X_batch @ theta) - Y_batch)
			
			if self.__IS_VERBOSE or self.__SHOW_PLOTS or self.__SAVE_PLOTS:
				self.__measure_performance(X_train, Y_train, theta, X_cv, Y_cv)

			if self.__IS_VERBOSE:
				print(Fore.BLUE + 'Training:         ' + Fore.RESET +
					'Start Cell %d: Epoch = %d, Cost = %.3f, Accuracy = %.3f, F1 Score = %.3f' % (
					self.__current_label,
					self.__epoch_list[-1],
					self.__train_cost_list[-1],
					self.__train_accuracy_list[-1],
					self.__train_f1_list[-1]))
				if self.__is_cv:
					print(Fore.GREEN + 'Cross Validation: ' + Fore.RESET +
						'Start Cell %d: Epoch = %d, Cost = %.3f, Accuracy = %.3f, F1 Score = %.3f' % (
						self.__current_label,
						self.__epoch_list[-1],
						self.__cv_cost_list[-1],
						self.__cv_accuracy_list[-1],
						self.__cv_f1_list[-1]))
			
			if self.__SHOW_PLOTS:
				self.__update_plots()
				plt.pause(0.001)

		# print('epoch = %d, cost = %f, correct = %d / %d' % (self.__epoch, cost, self.__count_correct_predictions(Y, theta), self.__m))
		return theta

	def	__select_x_y_batch(self, X, Y, batch_size):
		m = X.shape[0]
		if m > batch_size:
			# indices = np.arange(m)
			# indices = np.random.permutation(indices)[:batch_size]

			indices = np.arange(100)

			X_batch = X[indices, :]
			Y_batch = Y[indices, :]
			return X_batch, Y_batch
		else:
			return X, Y

	def __measure_performance(self, X_train, Y_train, theta, X_cv=None, Y_cv=None):
		self.__epoch_list.append(self.__epoch)

		self.__train_cost_list.append(self.__get_cost(X_train, Y_train, theta))
		train_predictions = self.__PREDICT(X_train @ theta)
		self.__train_accuracy_list.append(self.__get_accuracy(train_predictions, Y_train))
		self.__train_f1_list.append(self.__get_f1_score(train_predictions, Y_train))

		if self.__is_cv:
			self.__cv_cost_list.append(self.__get_cost(X_cv, Y_cv, theta))
			cv_predictions = self.__PREDICT(X_cv @ theta)
			self.__cv_accuracy_list.append(self.__get_accuracy(cv_predictions, Y_cv))
			self.__cv_f1_list.append(self.__get_f1_score(cv_predictions, Y_cv))

	def __get_cost(self, X, Y, theta):
		m = X.shape[0]
		return 1 / m * np.sum(np.sum(
			-Y * np.log(self.__SIGMOID(X @ theta)) -
			(1 - Y) * (np.log(self.__SIGMOID(1 - (X @ theta))))))

	def __get_accuracy(self, predictions, Y):
		num_correct = np.sum(predictions == Y)
		total = Y.shape[0] * Y.shape[1]
		accuracy = num_correct / total
		return accuracy

	def __get_f1_score(self, predictions, Y):
		true_positives = np.sum((predictions == 1) & (Y == 1))
		false_positives = np.sum((predictions == 1) & (Y == 0))
		false_negatives = np.sum((predictions == 0) & (Y == 1))

		if true_positives == 0 and false_positives == 0 and false_negatives == 0:
			f1_score = 1
		elif (true_positives + false_positives) == 0 or (true_positives + false_negatives) == 0:
			f1_score = 0
		else:
			precision = true_positives / (true_positives + false_positives)
			recall = true_positives / (true_positives + false_negatives)

			if (precision + recall) == 0:
				f1_score = 0
			else:
				f1_score = 2 * precision * recall / (precision + recall)

		return f1_score

	def __init_plots(self):
		try:
			os.stat(self.__PLOTS_DIRECTORY)
		except:
			os.mkdir(self.__PLOTS_DIRECTORY)

		self.__fig = plt.figure(figsize=(18, 12))
		self.__fig.tight_layout()
		
		self.__ax_train_cost = self.__fig.add_subplot(2, 3, 1)
		self.__ax_train_accuracy = self.__fig.add_subplot(2, 3, 2)
		self.__ax_train_f1 = self.__fig.add_subplot(2, 3, 3)
		
		self.__ax_cv_cost = self.__fig.add_subplot(2, 3, 4)
		self.__ax_cv_accuracy = self.__fig.add_subplot(2, 3, 5)
		self.__ax_cv_f1 = self.__fig.add_subplot(2, 3, 6)

	def __init_lists(self):
		self.__epoch_list = []

		self.__train_cost_list = []
		self.__train_accuracy_list = []
		self.__train_f1_list = []

		self.__cv_cost_list = []
		self.__cv_accuracy_list = []
		self.__cv_f1_list = []

	def __update_plots(self):
		# Training: Cost vs Epoch
		self.__ax_train_cost.clear()
		self.__ax_train_cost.plot(self.__epoch_list, self.__train_cost_list)
		self.__ax_train_cost.fill_between(self.__epoch_list, 0, self.__train_cost_list, facecolor='blue', alpha=0.5)
		self.__ax_train_cost.set_xlabel('Epoch')
		self.__ax_train_cost.set_ylabel('Cost')
		self.__ax_train_cost.set_title('Training: Start Cell %d\nCost at Epoch %d: %.3f' %
			(self.__current_label, self.__epoch_list[-1], self.__train_cost_list[-1]))
		# Training: Prediction Accuracy vs Epoch
		self.__ax_train_accuracy.clear()
		self.__ax_train_accuracy.plot(self.__epoch_list, self.__train_accuracy_list)
		self.__ax_train_accuracy.fill_between(self.__epoch_list, 0, self.__train_accuracy_list, facecolor='cyan', alpha=0.5)
		self.__ax_train_accuracy.set_xlabel('Epoch')
		self.__ax_train_accuracy.set_ylabel('Accuracy')
		self.__ax_train_accuracy.set_title('Training: Start Cell %d\nPrediction Accuracy at Epoch %d: %.3f' %
			(self.__current_label, self.__epoch_list[-1], self.__train_accuracy_list[-1]))
		# Training: F1 Score vs Epoch
		self.__ax_train_f1.clear()
		self.__ax_train_f1.plot(self.__epoch_list, self.__train_f1_list)
		self.__ax_train_f1.fill_between(self.__epoch_list, 0, self.__train_f1_list, facecolor='red', alpha=0.5)
		self.__ax_train_f1.set_xlabel('Epoch')
		self.__ax_train_f1.set_ylabel('F1 Score')
		self.__ax_train_f1.set_title('Training: Start Cell %d\nF1 Score at Epoch %d: %.3f' %
			(self.__current_label, self.__epoch_list[-1], self.__train_f1_list[-1]))

		if self.__is_cv:
			# CV: Cost vs Epoch
			self.__ax_cv_cost.clear()
			self.__ax_cv_cost.plot(self.__epoch_list, self.__cv_cost_list)
			self.__ax_cv_cost.fill_between(self.__epoch_list, 0, self.__cv_cost_list, facecolor='blue', alpha=0.5)
			self.__ax_cv_cost.set_xlabel('Epoch')
			self.__ax_cv_cost.set_ylabel('Cost')
			self.__ax_cv_cost.set_title('Cross Validation: Start Cell %d\nCost at Epoch %d: %.3f' %
				(self.__current_label, self.__epoch_list[-1], self.__cv_cost_list[-1]))
			# CV: Prediction Accuracy vs Epoch
			self.__ax_cv_accuracy.clear()
			self.__ax_cv_accuracy.plot(self.__epoch_list, self.__cv_accuracy_list)
			self.__ax_cv_accuracy.fill_between(self.__epoch_list, 0, self.__cv_accuracy_list, facecolor='cyan', alpha=0.5)
			self.__ax_cv_accuracy.set_xlabel('Epoch')
			self.__ax_cv_accuracy.set_ylabel('Accuracy')
			self.__ax_cv_accuracy.set_title('Cross Validation: Start Cell %d\nPrediction Accuracy at Epoch %d: %.3f' %
				(self.__current_label, self.__epoch_list[-1], self.__cv_accuracy_list[-1]))
			# CV: F1 Score vs Epoch
			self.__ax_cv_f1.clear()
			self.__ax_cv_f1.plot(self.__epoch_list, self.__cv_f1_list)
			self.__ax_cv_f1.fill_between(self.__epoch_list, 0, self.__cv_f1_list, facecolor='red', alpha=0.5)
			self.__ax_cv_f1.set_xlabel('Epoch')
			self.__ax_cv_f1.set_ylabel('F1 Score')
			self.__ax_cv_f1.set_title('Cross Validation: Start Cell %d\nF1 Score at Epoch %d: %.3f' %
				(self.__current_label, self.__epoch_list[-1], self.__cv_f1_list[-1]))

	def __save_plots(self):
		filename = self.__PLOTS_DIRECTORY + 'Training Performance on Start Cell %d' % self.__current_label
		plt.savefig(filename)
		print('Saved plots in ' + Fore.BLUE + filename + '.png' + Fore.RESET)	

	def __write_processed_data_to_file(self, data, filename):
		print('Writing processed data in ' + Fore.BLUE + filename + Fore.RESET)
		with open(filename, 'wb') as file:
			np.savetxt(file, data, delimiter=',')
		
	def __write_parameters_to_file(self, filename='param.dat'):
		print('Writing model parameters in ' + Fore.BLUE + filename + Fore.RESET)

		print('theta.shape = ', self.__theta.shape)

		with open(filename, 'wb') as file:
			np.savetxt(file, self.__theta, delimiter=',')
		
	def __write_predictions_to_file(self, predictions, filename='submission.csv'):
		print('Writing predictions in ' + Fore.BLUE + filename + Fore.RESET)
		with open(filename, 'wb') as file:
			file.write(b'id,start.1,start.2,start.3,start.4,start.5,start.6,start.7,start.8,start.9,start.10,start.11,start.12,start.13,start.14,start.15,start.16,start.17,start.18,start.19,start.20,start.21,start.22,start.23,start.24,start.25,start.26,start.27,start.28,start.29,start.30,start.31,start.32,start.33,start.34,start.35,start.36,start.37,start.38,start.39,start.40,start.41,start.42,start.43,start.44,start.45,start.46,start.47,start.48,start.49,start.50,start.51,start.52,start.53,start.54,start.55,start.56,start.57,start.58,start.59,start.60,start.61,start.62,start.63,start.64,start.65,start.66,start.67,start.68,start.69,start.70,start.71,start.72,start.73,start.74,start.75,start.76,start.77,start.78,start.79,start.80,start.81,start.82,start.83,start.84,start.85,start.86,start.87,start.88,start.89,start.90,start.91,start.92,start.93,start.94,start.95,start.96,start.97,start.98,start.99,start.100,start.101,start.102,start.103,start.104,start.105,start.106,start.107,start.108,start.109,start.110,start.111,start.112,start.113,start.114,start.115,start.116,start.117,start.118,start.119,start.120,start.121,start.122,start.123,start.124,start.125,start.126,start.127,start.128,start.129,start.130,start.131,start.132,start.133,start.134,start.135,start.136,start.137,start.138,start.139,start.140,start.141,start.142,start.143,start.144,start.145,start.146,start.147,start.148,start.149,start.150,start.151,start.152,start.153,start.154,start.155,start.156,start.157,start.158,start.159,start.160,start.161,start.162,start.163,start.164,start.165,start.166,start.167,start.168,start.169,start.170,start.171,start.172,start.173,start.174,start.175,start.176,start.177,start.178,start.179,start.180,start.181,start.182,start.183,start.184,start.185,start.186,start.187,start.188,start.189,start.190,start.191,start.192,start.193,start.194,start.195,start.196,start.197,start.198,start.199,start.200,start.201,start.202,start.203,start.204,start.205,start.206,start.207,start.208,start.209,start.210,start.211,start.212,start.213,start.214,start.215,start.216,start.217,start.218,start.219,start.220,start.221,start.222,start.223,start.224,start.225,start.226,start.227,start.228,start.229,start.230,start.231,start.232,start.233,start.234,start.235,start.236,start.237,start.238,start.239,start.240,start.241,start.242,start.243,start.244,start.245,start.246,start.247,start.248,start.249,start.250,start.251,start.252,start.253,start.254,start.255,start.256,start.257,start.258,start.259,start.260,start.261,start.262,start.263,start.264,start.265,start.266,start.267,start.268,start.269,start.270,start.271,start.272,start.273,start.274,start.275,start.276,start.277,start.278,start.279,start.280,start.281,start.282,start.283,start.284,start.285,start.286,start.287,start.288,start.289,start.290,start.291,start.292,start.293,start.294,start.295,start.296,start.297,start.298,start.299,start.300,start.301,start.302,start.303,start.304,start.305,start.306,start.307,start.308,start.309,start.310,start.311,start.312,start.313,start.314,start.315,start.316,start.317,start.318,start.319,start.320,start.321,start.322,start.323,start.324,start.325,start.326,start.327,start.328,start.329,start.330,start.331,start.332,start.333,start.334,start.335,start.336,start.337,start.338,start.339,start.340,start.341,start.342,start.343,start.344,start.345,start.346,start.347,start.348,start.349,start.350,start.351,start.352,start.353,start.354,start.355,start.356,start.357,start.358,start.359,start.360,start.361,start.362,start.363,start.364,start.365,start.366,start.367,start.368,start.369,start.370,start.371,start.372,start.373,start.374,start.375,start.376,start.377,start.378,start.379,start.380,start.381,start.382,start.383,start.384,start.385,start.386,start.387,start.388,start.389,start.390,start.391,start.392,start.393,start.394,start.395,start.396,start.397,start.398,start.399,start.400\n')
			id_column = np.arange(predictions.shape[0]) + 1				# insert 'id' at column 0
			predictions = np.c_[id_column, predictions]
			np.savetxt(file, predictions.astype(int), fmt='%i', delimiter=',')
		

















