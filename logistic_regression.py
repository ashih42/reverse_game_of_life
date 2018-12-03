from colorama import Fore, Back, Style
import matplotlib.pyplot as plt
import numpy as np
import os

from measures import get_accuracy, get_f1_score
from matrix_data_parser import MatrixDataParser

def sigmoid(X):
	return 1 / (1 + np.exp(-X))

def predict(X):
	return (X > 0).astype(float)

def preprocess_X(X):
	return np.c_[ np.ones(X.shape[0]), X ]

class LogisticRegression:
	__NUM_FEATURES = 11*11 + 1
	__NUM_LABELS = 1

	__LEARNING_RATE = 0.001
	__MAX_EPOCHS = 10
	
	__BATCH_SIZE = 100

	__IS_VERBOSE = os.getenv('RGOL_VERBOSE') == 'TRUE'
	__IS_PLOTS = os.getenv('RGOL_PLOTS') == 'TRUE'

	__PLOTS_DIRECTORY = 'LR_Plots/'
	__PARAM_DIRECTORY = 'LR_Param/'

	__NUM_MODELS = 0

	def __init__(self):
		LogisticRegression.__NUM_MODELS += 1
		self.__delta = LogisticRegression.__NUM_MODELS
		self.__Theta = np.zeros((self.__NUM_FEATURES, self.__NUM_LABELS))

	def load_param(self, param_filename):
		parser = MatrixDataParser(param_filename, num_rows=self.__NUM_FEATURES, num_cols=self.__NUM_LABELS)
		self.__Theta = np.array(parser.data, dtype = float)

	def predict(self, X):
		X = preprocess_X(X)
		return predict(X @ self.__Theta)

	def fit(self, X_train, Y_train, X_cv, Y_cv, is_cv):
		self.__init_lists()
		self.__init_plots()
		
		X_train = preprocess_X(X_train)
		self.__is_cv = is_cv
		if self.__is_cv:
			X_cv = preprocess_X(X_cv)
		
		self.__run_gradient_descent(X_train, Y_train, X_cv, Y_cv)

		if self.__IS_PLOTS:
			self.__update_plots()
			self.__save_plots()
		
		self.__write_parameters_to_file()

	def __run_gradient_descent(self, X_train, Y_train, X_cv, Y_cv):
		for epoch in range(self.__MAX_EPOCHS):
			self.__run_gradient_descent_step(X_train, Y_train, X_cv, Y_cv)

			if self.__IS_VERBOSE or self.__IS_PLOTS:
				self.__measure_performance(epoch, X_train, Y_train, X_cv, Y_cv)

			if self.__IS_VERBOSE:
				self.__print_performance()

	def __run_gradient_descent_step(self, X_train, Y_train, X_cv, Y_cv):
		n_batches = X_train.shape[0] // self.__BATCH_SIZE

		for batch_number in range(n_batches):
			X_batch, Y_batch = self.__get_X_Y_batch(X_train, Y_train, batch_number)

			# print('X_batch shape = ', X_batch.shape)
			# print('Y_batch shape = ', Y_batch.shape)

			self.__Theta = self.__Theta - self.__LEARNING_RATE * X_batch.T @ (
				sigmoid(X_batch @ self.__Theta) - Y_batch)
			
	def __get_X_Y_batch(self, X, Y, batch_number):
		low = batch_number * self.__BATCH_SIZE
		high = (batch_number + 1) * self.__BATCH_SIZE
		X_batch = X[ low:high, : ]
		Y_batch = Y[ low:high, : ]
		return X_batch, Y_batch

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

	def __measure_performance(self, epoch, X_train, Y_train, X_cv, Y_cv):
		self.__epoch_list.append(epoch)

		self.__train_cost_list.append(self.__get_cost(X_train, Y_train))
		train_predictions = predict(X_train @ self.__Theta)
		self.__train_accuracy_list.append(get_accuracy(train_predictions, Y_train))
		self.__train_f1_list.append(get_f1_score(train_predictions, Y_train))

		if self.__is_cv:
			self.__cv_cost_list.append(self.__get_cost(X_cv, Y_cv))
			cv_predictions = predict(X_cv @ self.__Theta)
			self.__cv_accuracy_list.append(get_accuracy(cv_predictions, Y_cv))
			self.__cv_f1_list.append(get_f1_score(cv_predictions, Y_cv))

	def __get_cost(self, X, Y):
		m = X.shape[0]
		return 1 / m * np.sum(np.sum(
			-Y * np.log(sigmoid(X @ self.__Theta)) -
			(1 - Y) * (np.log(sigmoid(1 - (X @ self.__Theta))))))

	def __print_performance(self):
		print(Fore.BLUE + 'Training:         ' + Fore.RESET +
			'Delta = %d: Epoch = %d, Cost = %.6f, Accuracy = %.6f, F1 Score = %.6f' % (
			self.__delta,
			self.__epoch_list[-1],
			self.__train_cost_list[-1],
			self.__train_accuracy_list[-1],
			self.__train_f1_list[-1]))

		if self.__is_cv:
			print(Fore.GREEN + 'Cross Validation: ' + Fore.RESET +
				'Delta = %d: Epoch = %d, Cost = %.6f, Accuracy = %.6f, F1 Score = %.6f' % (
				self.__delta,
				self.__epoch_list[-1],
				self.__cv_cost_list[-1],
				self.__cv_accuracy_list[-1],
				self.__cv_f1_list[-1]))

	def __update_plots(self):
		# Training: Cost vs Epoch
		self.__ax_train_cost.clear()
		self.__ax_train_cost.plot(self.__epoch_list, self.__train_cost_list)
		self.__ax_train_cost.fill_between(self.__epoch_list, 0, self.__train_cost_list, facecolor='blue', alpha=0.5)
		self.__ax_train_cost.set_xlabel('Epoch')
		self.__ax_train_cost.set_ylabel('Cost')
		self.__ax_train_cost.set_title('Training: Delta = %d\nCost at Epoch %d: %.3f' %
			(self.__delta, self.__epoch_list[-1], self.__train_cost_list[-1]))
		# Training: Prediction Accuracy vs Epoch
		self.__ax_train_accuracy.clear()
		self.__ax_train_accuracy.plot(self.__epoch_list, self.__train_accuracy_list)
		self.__ax_train_accuracy.fill_between(self.__epoch_list, 0, self.__train_accuracy_list, facecolor='cyan', alpha=0.5)
		self.__ax_train_accuracy.set_xlabel('Epoch')
		self.__ax_train_accuracy.set_ylabel('Accuracy')
		self.__ax_train_accuracy.set_title('Training: Delta = %d\nPrediction Accuracy at Epoch %d: %.3f' %
			(self.__delta, self.__epoch_list[-1], self.__train_accuracy_list[-1]))
		# Training: F1 Score vs Epoch
		self.__ax_train_f1.clear()
		self.__ax_train_f1.plot(self.__epoch_list, self.__train_f1_list)
		self.__ax_train_f1.fill_between(self.__epoch_list, 0, self.__train_f1_list, facecolor='red', alpha=0.5)
		self.__ax_train_f1.set_xlabel('Epoch')
		self.__ax_train_f1.set_ylabel('F1 Score')
		self.__ax_train_f1.set_title('Training: Delta = %d\nF1 Score at Epoch %d: %.3f' %
			(self.__delta, self.__epoch_list[-1], self.__train_f1_list[-1]))

		if self.__is_cv:
			# CV: Cost vs Epoch
			self.__ax_cv_cost.clear()
			self.__ax_cv_cost.plot(self.__epoch_list, self.__cv_cost_list)
			self.__ax_cv_cost.fill_between(self.__epoch_list, 0, self.__cv_cost_list, facecolor='blue', alpha=0.5)
			self.__ax_cv_cost.set_xlabel('Epoch')
			self.__ax_cv_cost.set_ylabel('Cost')
			self.__ax_cv_cost.set_title('Cross Validation: Delta = %d\nCost at Epoch %d: %.3f' %
				(self.__delta, self.__epoch_list[-1], self.__cv_cost_list[-1]))
			# CV: Prediction Accuracy vs Epoch
			self.__ax_cv_accuracy.clear()
			self.__ax_cv_accuracy.plot(self.__epoch_list, self.__cv_accuracy_list)
			self.__ax_cv_accuracy.fill_between(self.__epoch_list, 0, self.__cv_accuracy_list, facecolor='cyan', alpha=0.5)
			self.__ax_cv_accuracy.set_xlabel('Epoch')
			self.__ax_cv_accuracy.set_ylabel('Accuracy')
			self.__ax_cv_accuracy.set_title('Cross Validation: Delta = %d\nPrediction Accuracy at Epoch %d: %.3f' %
				(self.__delta, self.__epoch_list[-1], self.__cv_accuracy_list[-1]))
			# CV: F1 Score vs Epoch
			self.__ax_cv_f1.clear()
			self.__ax_cv_f1.plot(self.__epoch_list, self.__cv_f1_list)
			self.__ax_cv_f1.fill_between(self.__epoch_list, 0, self.__cv_f1_list, facecolor='red', alpha=0.5)
			self.__ax_cv_f1.set_xlabel('Epoch')
			self.__ax_cv_f1.set_ylabel('F1 Score')
			self.__ax_cv_f1.set_title('Cross Validation: Delta = %d\nF1 Score at Epoch %d: %.3f' %
				(self.__delta, self.__epoch_list[-1], self.__cv_f1_list[-1]))

	def __save_plots(self):
		filename = self.__PLOTS_DIRECTORY + 'Training Performance for Delta = %d' % self.__delta
		plt.savefig(filename)
		print('Saved plots in ' + Fore.BLUE + filename + '.png' + Fore.RESET)

	def __write_parameters_to_file(self):
		try:
			os.stat(self.__PARAM_DIRECTORY)
		except:
			os.mkdir(self.__PARAM_DIRECTORY)
		filename = self.__PARAM_DIRECTORY + 'param_%d.dat' % self.__delta
		print('Writing model parameters in ' + Fore.BLUE + filename + Fore.RESET)
		with open(filename, 'wb') as file:
			np.savetxt(file, self.__Theta, delimiter=',')

	








