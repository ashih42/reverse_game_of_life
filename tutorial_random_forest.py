# Random Forest Algorithm on Sonar Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Select the best split point for a dataset
def get_split(dataset, n_features):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	features = list()
	print('len(dataset) = ', len(dataset))
	while len(features) < n_features:
		print('len(dataset[0]) = ', len(dataset[0]) )
		index = randrange(len(dataset[0])-1)
		print('index = ', index)
		if index not in features:
			features.append(index)
			print('  APPENDED index = ', index)
	print(' GOT HERE 1')
	for index in features:
		# print('index = ', index)
		for row in dataset:
			# print(' row = ', row)
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			print('  gini = ', gini)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	print(' GOT HERE 2')
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left, n_features)
		split(node['left'], max_depth, min_size, n_features, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right, n_features)
		split(node['right'], max_depth, min_size, n_features, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
	root = get_split(train, n_features)
	split(root, max_depth, min_size, n_features, 1)
	return root

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample

# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
	predictions = [predict(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count)

# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
	trees = list()
	for i in range(n_trees):
		sample = subsample(train, sample_size)
		tree = build_tree(sample, max_depth, min_size, n_features)
		trees.append(tree)
	predictions = [bagging_predict(trees, row) for row in test]
	return(predictions)


import numpy as np
import os

def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[split on feature: %d]' % ((' ' * depth, node['index'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[predict: %s]' % ((' ' * depth, node)))

def array_to_list(array):
	lst = []
	for row in array:
		lst.append(row.tolist())
	return lst


class Tut_RandomForest:
	np.random.seed(42)

	__MAX_DEPTH = 10
	__MIN_SIZE = 1		# unused
	__SAMPLE_RATIO = 1	# unused
	__N_FEATURES = 7	# unused
	__N_TREES = 10
	__PARAM_DIRECTORY = 'RF_Param/'

	__IS_CV = os.getenv('RGOL_CV') == 'TRUE'
	__IS_VERBOSE = os.getenv('RGOL_VERBOSE') == 'TRUE'

	def __init__(self, delta):
		self.__delta = delta

	def fit(self, X_train, Y_train, X_cv, Y_cv):
		dataset_train = np.c_[ X_train, Y_train ]

		dataset_list = array_to_list(dataset_train)

		# feature_indices = list(range(dataset_train.shape[1] - 1))
		# # feature_indices = np.arange(dataset.shape[1] - 1)
		# np.random.shuffle(feature_indices)
		# feature_indices = feature_indices[ :self.__N_FEATURES ]
		

		self.__trees = []
		tree_id = 0
		for i in range(self.__N_TREES):
			tree = build_tree(dataset_list, self.__MAX_DEPTH, self.__MIN_SIZE, self.__N_FEATURES)
			self.__trees.append(tree)

			print('tree_id = ', tree_id)
			print_tree(tree)

			tree_id += 1

	def load_param(self):
		print('sk_RandomForest.load_param(): NOP')

	def __write_parameters_to_file(self, tree, tree_id):
		pass
		# try:
		# 	os.stat(self.__PARAM_DIRECTORY)
		# except:
		# 	os.mkdir(self.__PARAM_DIRECTORY)
		# filename = self.__PARAM_DIRECTORY + 'param_%d_%03d.dat' % (self.__delta, tree_id)
		# print('Writing model parameters in ' + Fore.BLUE + filename + Fore.RESET)
		# with open(filename, 'wb') as file:
		# 	pickle.dump(tree, file)



	def predict(self, X):
		return self.__model.predict(X).reshape(-1, 1)

