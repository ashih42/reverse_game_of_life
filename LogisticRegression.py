from TrainingDataParser import TrainingDataParser
from exceptions import LogisticRegressionException
from colorama import Fore, Back, Style

import numpy as np
import math

'''

m = number of training examples
n = number of features = 401 + 1

'''

def ft_exp(x):
	try:
		answer = math.exp(x)
	except OverflowError:
		answer = float('inf')
	return answer


class LogisticRegression:
	__N = 402
	# __ALPHA = 0.00001			# works for large dataset, 1 label
	#__ALPHA = 0.000001			# works for large dataset, 400 labels, SLOWLY
	__ALPHA = 0.003
	# __ALPHA = 0.001
	# __SIGMOID = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
	__SIGMOID = np.vectorize(lambda x: 1 / (1 + ft_exp(-x)))
	__PREDICT = np.vectorize(lambda x : 1 if x > 0 else 0)
	
	def __init__(self):
		pass

	def train(self, filename):
		print('begin parsing...')
		parser = TrainingDataParser(filename)

		print('finished parsing...')


		all_data = np.array(parser.data, dtype=float)

		self.__m = all_data.shape[0]

		self.__X = all_data[:, 401:]					# grab stop cells in index 401 - 801
		
		self.__X = np.c_[ all_data[:, 0], self.__X]		# grab delta in index 0
		self.__X[:, 0] = (self.__X[:, 0] - 3) / 4		# feature scaling: delta_scaled = (delta - 3) / 4
		self.__X = np.c_[np.ones(self.__m), self.__X]	# add column of 1s

		self.__Y = all_data[:, 1:401]					# grab start cells in index 1 - 400
		self.__theta = np.zeros((402, 400))

		# self.__Y = all_data[:, 1]						# grab start cells in index 1
		# self.__Y = self.__Y.reshape(self.__Y.shape[0], 1)
		# self.__theta = np.zeros((402, 1))

		# print(all_data.shape)
		# print('m = ', self.__m)
		# print('X.shape = ', self.__X.shape)
		# print('Y.shape = ', self.__Y.shape)
		# print('theta.shape = ', self.__theta.shape)

		# self.__run_gradient_descent(is_batch=True)
		self.__run_gradient_descent(is_batch=False)

		print('DONE LOL')

	def __run_gradient_descent(self, is_batch):
		self.__iteration = 0
		cost = self.__compute_cost()

		if is_batch:
			print('running BATCH gradient descent...')
			while True:
				self.__iteration += 1

				self.__theta = self.__theta - self.__ALPHA * self.__X.T @ (self.__SIGMOID(self.__X @ self.__theta) - self.__Y)

				old_cost = cost
				cost = self.__compute_cost()

				if cost > old_cost:
					raise LogisticRegressionException('learning rate is too high')
				if abs(cost - old_cost) < 1e-6:
					break

				# print('iteration = %d, cost = %f' % (self.__iteration, ', cost = ', cost))
				print('iteration = %d, cost = %f, correct = %d / %d' % (self.__iteration, cost, self.__count_correct_predictions(), self.__m))

		else:
			print('running ONLINE gradient descent...')
			# while True:
			for i in range(20):
				self.__iteration += 1

				for i in range(self.__m):
					x_row = self.__X[i, :]
					y_row = self.__Y[i, :]
					x_row = x_row.reshape(1, x_row.shape[0])
					y_row = y_row.reshape(1, y_row.shape[0])

					# print('x_row.shape = ', x_row.shape)
					# print('y_row.shape = ', y_row.shape)
					# print('theta.shape = ', self.__theta.shape)

					self.__theta = self.__theta - self.__ALPHA * x_row.T @ (self.__SIGMOID(x_row @ self.__theta) - y_row)

					# print('iteration %d, row %d, cost = %f' % (self.__iteration, i, self.__compute_cost_fast(x_row, y_row)))
					# print('iteration %d, row %d' % (self.__iteration, i))

				old_cost = cost
				cost = self.__compute_cost()

				# if cost > old_cost:
				# 	raise LogisticRegressionException('learning rate is too high')
				# if abs(cost - old_cost) < 1e-6:
				# 	break

				# print('iteration = %d, cost = %f' % (self.__iteration, ', cost = ', cost))
				print('iteration = %d, cost = %f, correct = %d / %d' % (self.__iteration, cost, self.__count_correct_predictions(), self.__m))


			# print('running ONLINE gradient descent...')
			# while True:
			# 	self.__iteration += 1

			# 	for i in range(self.__m):
			# 		x_row = self.__X[i, :]
			# 		y_row = self.__Y[i, :]
			# 		x_row = x_row.reshape(1, x_row.shape[0])
			# 		y_row = y_row.reshape(1, y_row.shape[0])

			# 		# print('x_row.shape = ', x_row.shape)
			# 		# print('y_row.shape = ', y_row.shape)
			# 		# print('theta.shape = ', self.__theta.shape)

			# 		self.__theta = self.__theta - self.__ALPHA * x_row.T @ (self.__SIGMOID(x_row @ self.__theta) - y_row)

			# 		# print('iteration %d, row %d, cost = %f' % (self.__iteration, i, self.__compute_cost()))
			# 		# print('iteration %d, row %d' % (self.__iteration, i))

			# 	old_cost = cost
			# 	cost = self.__compute_cost()

			# 	if cost > old_cost:
			# 		raise LogisticRegressionException('learning rate is too high')
			# 	if abs(cost - old_cost) < 1e-6:
			# 		break

			# 	# print('iteration = %d, cost = %f' % (self.__iteration, ', cost = ', cost))
			# 	print('iteration = %d, cost = %f, correct = %d / %d' % (self.__iteration, cost, self.__count_correct_predictions(), self.__m))


	def __compute_cost(self):
		return 1 / self.__m * np.sum(np.sum(
			-self.__Y * np.log(self.__SIGMOID(self.__X @ self.__theta)) -
			(1 - self.__Y) * (np.log(self.__SIGMOID(1 - (self.__X @ self.__theta))))))

	def __compute_cost_fast(self, x_row, y_row):
		return 1 / self.__m * np.sum(np.sum(
			-y_row * np.log(self.__SIGMOID(x_row @ self.__theta)) -
			(1 - y_row) * (np.log(self.__SIGMOID(1 - (x_row @ self.__theta))))))

	def __count_correct_predictions(self):
		predictions = self.__PREDICT(self.__X @ self.__theta)
		num_correct_per_row = np.sum(np.equal(predictions, self.__Y), axis=1)
		num_correct = (num_correct_per_row == 400).sum()
		return num_correct

		# predictions = self.__PREDICT(self.__X @ self.__theta)
		# num_correct_per_row = np.sum(np.equal(predictions, self.__Y), axis=1)
		# num_correct = (num_correct_per_row == 1).sum()
		# return num_correct
		

		# print('predictions.shape = ', predictions.shape)
		# print('predictions = ')
		# print(predictions)
		
		



		























