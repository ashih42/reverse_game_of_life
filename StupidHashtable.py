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

class StupidHashtable:
	np.random.seed(42)
	# __NUM_FEATURES = 11*11 + 2
	__NUM_FEATURES = 11*11 + 2
	__NUM_LABELS = 400

	# __NUM_FEATURES = 402			# try with just the given data, no new features

	__SIGMOID = np.vectorize(lambda x: 1 / (1 + ft_exp(-x)))
	__PREDICT = np.vectorize(lambda x : 1 if x > 0 else 0)

	__BATCH_SIZE = 100
	
	def __init__(self, param_filename=None):
		if param_filename is not None:
			# print('self.__NUM_FEATURES = %d', self.__NUM_FEATURES)
			parser = MatrixDataParser(param_filename, num_rows=self.__NUM_FEATURES, num_cols=400)
			self.__theta = np.array(parser.data, dtype = float)


	def train(self, filename):
		parser = DataParser(filename, is_training_data=True)
		self.X, self.Y = self.__wrangle_data(parser.data, is_training_data=True)

		print('X.shape = ', self.X.shape)
		print('Y.shape = ', self.Y.shape)

		# self.__train(X, Y)
	
	def predict_lol(self, filename):
		print('predict_lol()...')
		parser = DataParser(filename, is_training_data=False)
		test_data = np.array(parser.data, dtype = float)
		X_test, _ = self.__wrangle_data(parser.data, is_training_data=False)

		# X_test = self.X

		print('X_test.shape = ', X_test.shape)

		predictions = np.zeros((X_test.shape[0], 400))
		print('predictions.shape = ', predictions.shape)

		matches = 0

		for i in range(X_test.shape[0]):
			print('predict_lol() i = %d' % i)
			X_test_hash = X_test[i, 0]
			X_test_delta = X_test[i, 1]
			

			for j in range(self.X.shape[0]):
				X_hash = self.X[i, 0]
				X_delta = self.X[i, 1]
				if X_hash == X_test_hash and X_delta == X_test_delta:
					predictions[i, :] = self.Y[i, :]
					matches += 1
					break

		print('matches = %d / %d' % (matches, predictions.shape[0]))
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

		X = self.__format_X(X)

		print('finished formatting X')

		if is_training_data:
			Y = data[:, 1:401]							# grab 'start cell' values in index 1 - 400
		else:
			Y = None

		# print('X.shape = ', X.shape)
		# print('Y.shape = ', Y.shape)

		return X, Y
	
	def __format_X(self, X):
		board = X[:, 1:]

		hash_col = np.empty((X.shape[0], 1))

		for i in range(X.shape[0]):
			# print('i = ', i)
			hash_sum = 0
			for j in range(400):
				if board[i, j] == 1:
					hash_sum += 2 ** j
			hash_col[i, 0] = hash_sum

		X_new = np.c_[ hash_col, X ]

		return X_new


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
		

















