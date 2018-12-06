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

def get_split(dataset, feature_indices):
	class_values = [0, 1]
	best_index = None
	best_score = float('inf')
	best_left = None
	best_right = None
	for index in feature_indices:
		left, right = test_split(dataset, index)
		gini_index = get_gini_index(left, right, class_values)
		if gini_index < best_score:
			best_index = index
			best_score = gini_index
			best_left = left
			best_right = right
	return {'index': best_index, 'left': best_left, 'right': best_right}

def build_tree(dataset, max_depth, min_size, feature_indices):
	root = get_split(dataset, feature_indices)
	split(root, max_depth, min_size, feature_indices, 1)
	return root

def split(node, max_depth, min_size, feature_indices, depth):
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
		node['left'] = get_split(left, feature_indices)
		split(node['left'], max_depth, min_size, feature_indices, depth + 1)

	if right.shape[0] <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right, feature_indices)
		split(node['right'], max_depth, min_size, feature_indices, depth + 1)

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

class RandomForest:
	__MAX_DEPTH = 10
	__MIN_SIZE = 10
	__SAMPLE_RATIO = 0.01
	__N_FEATURES = 20
	__N_TREES = 100

	__PARAM_DIRECTORY = 'RF_Param/'

	__IS_CV = os.getenv('RGOL_CV') == 'TRUE'
	__IS_VERBOSE = os.getenv('RGOL_VERBOSE') == 'TRUE'

	def __init__(self, delta):
		self.__delta = delta

	def fit(self, X_train, Y_train, X_cv, Y_cv):
		dataset_train = np.c_[ X_train, Y_train ]
		self.__trees = []
		tree_id = 0
		for tree in range(self.__N_TREES):
			tree = self.__build_tree(dataset_train)
			self.__trees.append(tree)
			self.__write_parameters_to_file(tree, tree_id)
			tree_id += 1

		if self.__IS_VERBOSE:
			self.__measure_performance(X_train, Y_train, X_cv, Y_cv)

	def load_param(self):
		self.__trees = []
		tree_id = 0
		for tree in range(self.__N_TREES):
			param_filename = self.__PARAM_DIRECTORY + 'param_%d_%03d.dat' % (self.__delta, tree_id)
			print('Loading model parameters in ' + Fore.BLUE + param_filename + Fore.RESET)
			with open(param_filename, 'rb') as file:
				self.__trees.append(pickle.load(file))
			tree_id += 1

	def __write_parameters_to_file(self, tree, tree_id):
		try:
			os.stat(self.__PARAM_DIRECTORY)
		except:
			os.mkdir(self.__PARAM_DIRECTORY)
		filename = self.__PARAM_DIRECTORY + 'param_%d_%03d.dat' % (self.__delta, tree_id)
		print('Writing model parameters in ' + Fore.BLUE + filename + Fore.RESET)
		with open(filename, 'wb') as file:
			pickle.dump(tree, file)

	def __measure_performance(self, X_train, Y_train, X_cv, Y_cv):
		predictions_train = self.predict(X_train)
		print(Fore.BLUE + 'Training:         ' + Fore.RESET +
			'Delta = %d: Accuracy = %.6f, F1 Score = %.6f' % (
			self.__delta,
			get_accuracy(predictions_train, Y_train),
			get_f1_score(predictions_train, Y_train)))
		if self.__IS_CV:
			predictions_cv = self.predict(X_cv)
			accuracy_cv = get_accuracy(predictions_cv, Y_cv)
			f1_score_cv = get_f1_score(predictions_cv, Y_cv)
			print(Fore.GREEN + 'Cross Validation: ' + Fore.RESET +
				'Delta = %d: Accuracy = %.6f, F1 Score = %.6f' % (
				self.__delta,
				get_accuracy(predictions_cv, Y_cv),
				get_f1_score(predictions_cv, Y_cv)))

	def __build_tree(self, dataset):
		# select random sample subset from dataset
		sample_indices = np.arange(dataset.shape[0])
		np.random.shuffle(sample_indices)
		sample_indices = sample_indices[ :int(self.__SAMPLE_RATIO * sample_indices.shape[0]) ]
		dataset_sample = dataset[ sample_indices ]

		# select random feature indices
		feature_indices = np.arange(dataset.shape[1] - 1)
		np.random.shuffle(feature_indices)
		feature_indices = feature_indices[ :self.__N_FEATURES ]

		return build_tree(dataset_sample, self.__MAX_DEPTH, self.__MIN_SIZE, feature_indices)


	def predict(self, X):
		sum_predictions = np.zeros((X.shape[0], 1))
		tree_id = 0
		for tree in self.__trees:
			print('RandomForest.predict(): tree_id = ', tree_id)
			tree_id += 1
			sum_predictions += np.apply_along_axis(predict, 1, X, tree).reshape(-1, 1)
		return (sum_predictions >= self.__N_TREES / 2).astype(int)

class DecisionTree:
	__MAX_DEPTH = 5
	__MIN_SIZE = 10

	__PARAM_DIRECTORY = 'DT_Param/'

	__IS_CV = os.getenv('RGOL_CV') == 'TRUE'
	__IS_VERBOSE = os.getenv('RGOL_VERBOSE') == 'TRUE'

	def __init__(self, delta):
		self.__delta = delta

	def fit(self, X_train, Y_train, X_cv, Y_cv):
		dataset_train = np.c_[ X_train, Y_train ]
		feature_indices = np.arange(X_train.shape[1])
		self.__root = build_tree(dataset_train, self.__MAX_DEPTH, self.__MIN_SIZE, feature_indices)
		self.__write_parameters_to_file()
		if self.__IS_VERBOSE:
			self.__measure_performance(X_train, Y_train, X_cv, Y_cv)

	def __measure_performance(self, X_train, Y_train, X_cv, Y_cv):
		predictions_train = self.predict(X_train)
		print(Fore.BLUE + 'Training:         ' + Fore.RESET +
			'Delta = %d: Accuracy = %.6f, F1 Score = %.6f' % (
			self.__delta,
			get_accuracy(predictions_train, Y_train),
			get_f1_score(predictions_train, Y_train)))
		if self.__IS_CV:
			predictions_cv = self.predict(X_cv)
			accuracy_cv = get_accuracy(predictions_cv, Y_cv)
			f1_score_cv = get_f1_score(predictions_cv, Y_cv)
			print(Fore.GREEN + 'Cross Validation: ' + Fore.RESET +
				'Delta = %d: Accuracy = %.6f, F1 Score = %.6f' % (
				self.__delta,
				get_accuracy(predictions_cv, Y_cv),
				get_f1_score(predictions_cv, Y_cv)))

	def predict(self, X):
		return np.apply_along_axis(predict, 1, X, self.__root).reshape(-1, 1)

	def load_param(self):
		param_filename = self.__PARAM_DIRECTORY + 'param_%d.dat' % self.__delta
		print('Loading model parameters in ' + Fore.BLUE + param_filename + Fore.RESET)
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

















