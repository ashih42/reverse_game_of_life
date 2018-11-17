from TrainingDataParser import TrainingDataParser
from exceptions import LogisticRegressionException
from colorama import Fore, Back, Style

import numpy as np
import math

'''
m = number of training examples		= 100 ~ 50,000
n = number of features				= 401 + 1
'''

'''
This class will preprocess the data:

For now, filter to only process cases where delta = 1

For each row (delta = 1, stop cells 1 ... 400), generate:

for i = 1:400
	X[i] = (stop cell i, stop cell neighbors 1 ... 25)
	Y[i] = (start cell i)




'''

def ft_exp(x):
	try:
		answer = math.exp(x)
	except OverflowError:
		answer = float('inf')
	return answer


class LogisticRegression3:
	# __ALPHA = 0.00001			# works for large dataset, 1 label
	#__ALPHA = 0.000001			# works for large dataset, 400 labels, SLOWLY
	# __ALPHA = 0.003
	__ALPHA = 0.001
	__SIGMOID = np.vectorize(lambda x: 1 / (1 + ft_exp(-x)))
	__PREDICT = np.vectorize(lambda x : 1 if x > 0 else 0)
	
	def __init__(self):
		np.random.seed(42)
		pass

	def write_submission_file(self, filename='submission.csv'):
		with open(filename, 'wb') as submission_file:
			submission_file.write(b'id,start.1,start.2,start.3,start.4,start.5,start.6,start.7,start.8,start.9,start.10,start.11,start.12,start.13,start.14,start.15,start.16,start.17,start.18,start.19,start.20,start.21,start.22,start.23,start.24,start.25,start.26,start.27,start.28,start.29,start.30,start.31,start.32,start.33,start.34,start.35,start.36,start.37,start.38,start.39,start.40,start.41,start.42,start.43,start.44,start.45,start.46,start.47,start.48,start.49,start.50,start.51,start.52,start.53,start.54,start.55,start.56,start.57,start.58,start.59,start.60,start.61,start.62,start.63,start.64,start.65,start.66,start.67,start.68,start.69,start.70,start.71,start.72,start.73,start.74,start.75,start.76,start.77,start.78,start.79,start.80,start.81,start.82,start.83,start.84,start.85,start.86,start.87,start.88,start.89,start.90,start.91,start.92,start.93,start.94,start.95,start.96,start.97,start.98,start.99,start.100,start.101,start.102,start.103,start.104,start.105,start.106,start.107,start.108,start.109,start.110,start.111,start.112,start.113,start.114,start.115,start.116,start.117,start.118,start.119,start.120,start.121,start.122,start.123,start.124,start.125,start.126,start.127,start.128,start.129,start.130,start.131,start.132,start.133,start.134,start.135,start.136,start.137,start.138,start.139,start.140,start.141,start.142,start.143,start.144,start.145,start.146,start.147,start.148,start.149,start.150,start.151,start.152,start.153,start.154,start.155,start.156,start.157,start.158,start.159,start.160,start.161,start.162,start.163,start.164,start.165,start.166,start.167,start.168,start.169,start.170,start.171,start.172,start.173,start.174,start.175,start.176,start.177,start.178,start.179,start.180,start.181,start.182,start.183,start.184,start.185,start.186,start.187,start.188,start.189,start.190,start.191,start.192,start.193,start.194,start.195,start.196,start.197,start.198,start.199,start.200,start.201,start.202,start.203,start.204,start.205,start.206,start.207,start.208,start.209,start.210,start.211,start.212,start.213,start.214,start.215,start.216,start.217,start.218,start.219,start.220,start.221,start.222,start.223,start.224,start.225,start.226,start.227,start.228,start.229,start.230,start.231,start.232,start.233,start.234,start.235,start.236,start.237,start.238,start.239,start.240,start.241,start.242,start.243,start.244,start.245,start.246,start.247,start.248,start.249,start.250,start.251,start.252,start.253,start.254,start.255,start.256,start.257,start.258,start.259,start.260,start.261,start.262,start.263,start.264,start.265,start.266,start.267,start.268,start.269,start.270,start.271,start.272,start.273,start.274,start.275,start.276,start.277,start.278,start.279,start.280,start.281,start.282,start.283,start.284,start.285,start.286,start.287,start.288,start.289,start.290,start.291,start.292,start.293,start.294,start.295,start.296,start.297,start.298,start.299,start.300,start.301,start.302,start.303,start.304,start.305,start.306,start.307,start.308,start.309,start.310,start.311,start.312,start.313,start.314,start.315,start.316,start.317,start.318,start.319,start.320,start.321,start.322,start.323,start.324,start.325,start.326,start.327,start.328,start.329,start.330,start.331,start.332,start.333,start.334,start.335,start.336,start.337,start.338,start.339,start.340,start.341,start.342,start.343,start.344,start.345,start.346,start.347,start.348,start.349,start.350,start.351,start.352,start.353,start.354,start.355,start.356,start.357,start.358,start.359,start.360,start.361,start.362,start.363,start.364,start.365,start.366,start.367,start.368,start.369,start.370,start.371,start.372,start.373,start.374,start.375,start.376,start.377,start.378,start.379,start.380,start.381,start.382,start.383,start.384,start.385,start.386,start.387,start.388,start.389,start.390,start.391,start.392,start.393,start.394,start.395,start.396,start.397,start.398,start.399,start.400\n')
			answer = self.__PREDICT(self.__X @ self.__theta)
			id_column = np.arange(self.__m) + 1				# insert 'id' at column 0
			answer = np.c_[id_column, answer]
			np.savetxt(submission_file, answer.astype(int), fmt='%i', delimiter=',')


	def __get_neighbors(self, board, i):
		neighbors = []
		x_current = i % 20
		y_current = i // 20

		for x_offset in range(-2, 3):
			for y_offset in range(-2, 3):
				x = x_current + x_offset
				y = y_current + y_offset
				if not (0 <= x < 20 and 0 <= y < 20):
					neighbor_value = 0
				else:
					neighbor_value = 1 if board[int(y * 20 + x)] == 1 else -1
				neighbors.append(neighbor_value)

		return neighbors

	def __generate_data_from_board(self, start_board, stop_board):
		for i in range(400):
			self.__Y = np.append(self.__Y, start_board[i].reshape(1, 1), axis=0)
			neighbors = self.__get_neighbors(stop_board, i)
			self.__X = np.append(self.__X, np.array(neighbors).reshape(1, 25), axis=0)

	def __wrangle_data(self, filename):
		print('Parsing data in %s...' % filename)
		parser = TrainingDataParser(filename)
		print('Wrangling data...')
		all_data = np.array(parser.data, dtype=float)

		all_data = all_data[ all_data[:, 0] == 1 ]			# select only rows where delta == 1

		print('all_data.shape = ', all_data.shape)


		stop_boards = all_data[:, 401:802]					# grab 'stop cell' values in index 401 - 801
		start_boards = all_data[:, 1:401]					# grab 'start cell' values in index 1 - 400

		# assert start_boards.shape == stop_boards.shape

		self.__X = np.empty((0, 25))
		self.__Y = np.empty((0, 1))

		for i in range(stop_boards.shape[0]):
			self.__generate_data_from_board(start_boards[i, :], stop_boards[i, :])

		

		self.__m = self.__X.shape[0]

		self.__X = np.c_[np.ones(self.__m), self.__X]	# add column of 1s

		print('Y.shape = ', self.__Y.shape)
		print('X.shape = ', self.__X.shape)


	def train(self, filename):
		self.__wrangle_data(filename)
		
		

		# theta_col = self.__run_gradient_descent(Y_col, batch_size=1, epoch_limit=1000)		# online
		# self.__theta = self.__run_gradient_descent(self.__Y, batch_size=100, epoch_limit=1000)		# mini batch size = 100
		self.__theta = self.__run_gradient_descent(self.__Y, batch_size=self.__m, epoch_limit=1000)	# full batch
		

		print('theta.shape = ', self.__theta.shape)
		print('ALL COLUMNS correct = %d / %d' % (self.__count_correct_predictions(self.__Y, self.__theta), self.__m))

		self.write_submission_file()

	def	__select_x_y_batch(self, Y, batch_size):
		if self.__m > batch_size:
			indices = np.arange(self.__m)
			indices = np.random.permutation(indices)[:batch_size]
			x_batch = self.__X[indices, :]
			y_batch = Y[indices, :]
			return x_batch, y_batch
		else:
			return self.__X, Y

	def __run_gradient_descent(self, Y, batch_size, epoch_limit):
		assert 1 <= batch_size <= self.__m

		theta = np.zeros((26, 1))
		self.__iteration = 0

		# print('running MINI-BATCH gradient descent...')
		for i in range(epoch_limit):
			self.__iteration += 1
			x_batch, y_batch = self.__select_x_y_batch(Y, batch_size)

			# print('x_batch.shape = ', x_batch.shape)
			# print('y_batch.shape = ', y_batch.shape)
			# print('theta.shape = ', theta.shape)



			theta = theta - self.__ALPHA * x_batch.T @ (self.__SIGMOID(x_batch @ theta) - y_batch)
			cost = self.__compute_cost_batch(x_batch, y_batch, theta)
			# print('iteration = %d, cost = %f, correct = %d / %d' % (self.__iteration, cost, self.__count_correct_predictions(), self.__m))
			print('iteration = %d, cost = %f' % (self.__iteration, cost))

		print('iteration = %d, cost = %f, correct = %d / %d' % (self.__iteration, cost, self.__count_correct_predictions(Y, theta), self.__m))
		return theta

	def __compute_cost_batch(self, x_batch, y_batch, theta):
		batch_size = x_batch.shape[0]
		return 1 / batch_size * np.sum(np.sum(
			-y_batch * np.log(self.__SIGMOID(x_batch @ theta)) -
			(1 - y_batch) * (np.log(self.__SIGMOID(1 - (x_batch @ theta))))))

	def __count_correct_predictions(self, Y, theta):
		predictions = self.__PREDICT(self.__X @ theta)
		num_correct_per_row = np.sum(np.equal(predictions, Y), axis=1)
		num_correct = (num_correct_per_row == Y.shape[1]).sum()
		return num_correct
		

		
		



		























