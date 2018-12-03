from colorama import Fore, Back, Style
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

from measures import get_accuracy, get_f1_score

def get_gini_index(left, right, class_values):
	n_instances = left.shape[0] + right.shape[0]
	return get_gini_index_partial(left, n_instances, class_values) + \
		get_gini_index_partial(right, n_instances, class_values)

def get_gini_index_partial(group, n_instances, class_values):
	if group.shape[0] == 0:
		return 0.0
	else:
		score = 0.0
		for value in class_values:
			score += (np.sum(group[:, -1] == value) / group.shape[0]) ** 2
		return (1 - score) * group.shape[0] / n_instances

def test_split(dataset, index):
	left = dataset[ dataset[:, index] == 0 ]
	right = dataset[ dataset[:, index] == 1 ]
	return left, right

def get_split(dataset):
	class_values = [0, 1]
	best_index = None
	best_score = float('inf')
	best_left = None
	best_right = None
	for index in range(dataset.shape[1] - 1):
		left, right = test_split(dataset, index)
		gini_index = get_gini_index(left, right, class_values)
		if gini_index < best_score:
			best_index = index
			best_score = gini_index
			best_left = left
			best_right = right
	return {'index': best_index, 'left': best_left, 'right': best_right}

def build_tree(dataset, max_depth, min_size):
	root = get_split(dataset)
	split(root, max_depth, min_size, 1)
	return root

def split(node, max_depth, min_size, depth):
	left = node['left']
	right = node['right']
	del(node['left'])
	del(node['right'])

	if left.shape[0] == 0:
		node['left'] = node['right'] = to_terminal(right)
		return
	if right.shape[0] == 0:
		node['left'] = node['right']= to_terminal(left)
		return
	if depth >= max_depth:
		node['left'] = to_terminal(left)
		node['right'] = to_terminal(right)
		return
	
	if left.shape[0] <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth + 1)

	if right.shape[0] <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth + 1)

def to_terminal(group):
	count_0s = np.sum(group[:, -1] == 0)
	count_1s = np.sum(group[:, -1] == 1)
	return 1 if count_1s > count_0s else 0

def predict(row, node):
	if row[node['index']] == 0:
		if isinstance(node['left'], dict):
			return predict(row, node['left'])
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(row, node['right'])
		else:
			return node['right']

class DecisionTree:
	__MAX_DEPTH = 5
	__MIN_SIZE = 10

	__PARAM_DIRECTORY = 'DT_Param/'

	__IS_VERBOSE = os.getenv('RGOL_VERBOSE') == 'TRUE'

	def __init__(self, delta):
		self.__delta = delta
		
	def fit(self, X_train, Y_train, X_cv, Y_cv, is_cv):
		dataset_train = np.c_[ X_train, Y_train ]
		self.__root = build_tree(dataset_train, self.__MAX_DEPTH, self.__MIN_SIZE)
		self.__write_parameters_to_file()

		if self.__IS_VERBOSE:
			self.__measure_performance(X_train, Y_train, X_cv, Y_cv, is_cv)

	def __measure_performance(self, X_train, Y_train, X_cv, Y_cv, is_cv):
		predictions_train = self.predict(X_train)
		print(Fore.BLUE + 'Training:         ' + Fore.RESET +
			'Delta = %d: Accuracy = %.6f, F1 Score = %.6f' % (
			self.__delta,
			get_accuracy(predictions_train, Y_train),
			get_f1_score(predictions_train, Y_train)))

		if is_cv:
			predictions_cv = self.predict(X_cv)
			accuracy_cv = get_accuracy(predictions_cv, Y_cv)
			f1_score_cv = get_f1_score(predictions_cv, Y_cv)
			print(Fore.GREEN + 'Cross Validation: ' + Fore.RESET +
				'Delta = %d: Accuracy = %.6f, F1 Score = %.6f' % (
				self.__delta,
				get_accuracy(predictions_cv, Y_cv),
				get_f1_score(predictions_cv, Y_cv)))


	def predict(self, X):
		# predictions = np.empty((X.shape[0], 1))
		# for i in range(X.shape[0]):
		# 	predictions[i, :] = predict(X[ i, : ], self.__root)
		# return predictions
		return np.apply_along_axis(predict, 1, X, self.__root).reshape(-1, 1)

	def load_param(self, param_filename):
		with open(param_filename, 'rb') as file:
			self.__root = pickle.load(file)

	def __write_parameters_to_file(self):
		try:
			os.stat(self.__PARAM_DIRECTORY)
		except:
			os.mkdir(self.__PARAM_DIRECTORY)
		filename = self.__PARAM_DIRECTORY + 'param_%d.dat' % self.__delta
		print('Writing model parameters in ' + Fore.BLUE + filename + Fore.RESET)
		with open(filename, 'wb') as file:
			pickle.dump(self.__root, file)

















