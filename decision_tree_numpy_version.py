import numpy as np

def get_gini_index_half(group, n_instances, class_values):
	if group.shape[0] == 0:
		return 0.0
	else:
		score = 0.0
		for value in class_values:
			score += (np.sum(group[:, -1] == value) / group.shape[0]) ** 2
		return (1 - score) * group.shape[0] / n_instances

def get_gini_index(left, right, class_values):
	n_instances = left.shape[0] + right.shape[0]
	return get_gini_index_half(left, n_instances, class_values) + \
		get_gini_index_half(right, n_instances, class_values)

def test_split(index, value, dataset):
	left = dataset[ dataset[:, index] < value ]
	right = dataset[ dataset[:, index] >= value ]
	return left, right

def get_split(dataset):
	class_values = [0, 1]
	b_index = None
	b_value = None
	b_score = float('inf')
	b_left = None
	b_right = None
	for index in range(dataset.shape[1] - 1):
		for row in dataset:
			left, right = test_split(index, row[index], dataset)
			gini_index = get_gini_index(left, right, class_values)
			if gini_index < b_score:
				b_index = index
				b_value = row[index]
				b_score = gini_index
				b_left = left
				b_right = right
	return {'index': b_index, 'value': b_value, 'left': b_left, 'right': b_right}


def to_terminal(group):
	count_0s = np.sum(group[:, -1] == 0)
	count_1s = np.sum(group[:, -1] == 1)
	return 1 if count_1s > count_0s else 0

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

def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root


def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((' ' * depth, (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((' ' * depth, node)))

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


from random import seed
from random import randrange
from csv import reader
 
# Load a CSV file
def load_csv(filename):
	file = open(filename, "rt")
	lines = reader(file)
	dataset = list(lines)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, max_depth, min_size):
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
		predicted = algorithm(train_set, test_set, max_depth, min_size)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores



# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
	train_array = np.array(train)
	tree = build_tree(train_array, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)
 
# Test CART on Bank Note dataset
seed(1)
# load and prepare data
filename = 'data_banknote_authentication.csv'
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)



# evaluate algorithm
n_folds = 5
max_depth = 5
min_size = 10
scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))























# dataset = [
# 	[2.771244718,1.784783929,0],
# 	[1.728571309,1.169761413,0],
# 	[3.678319846,2.81281357,0],
# 	[3.961043357,2.61995032,0],
# 	[2.999208922,2.209014212,0],
# 	[7.497545867,3.162953546,1],
# 	[9.00220326,3.339047188,1],
# 	[7.444542326,0.476683375,1],
# 	[10.12493903,3.234550982,1],
# 	[6.642287351,3.319983761,1] ]

# dataset = np.array(dataset)


# stump = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
# for row in dataset:
# 	prediction = predict(stump, row)
# 	print('Expected=%d, Got=%d' % (row[-1], prediction))

# tree = build_tree(dataset, 1, 1)
# print_tree(tree)
# print()

# tree = build_tree(dataset, 2, 1)
# print_tree(tree)
# print()

# tree = build_tree(dataset, 3, 1)
# print_tree(tree)
# print()


# split = get_split(dataset)
# print('split type = ', type(split))
# # print(split)
# print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))


# classes = [0, 1]


# left = np.array( [ [1, 1], [1, 0] ] )
# right = np.array( [ [1, 1], [1, 0] ] )
# gini_index = get_gini_index(left, right, classes)
# print('gini_index = ', gini_index)


# left = np.array( [ [1, 0], [1, 0] ] )
# right = np.array( [ [1, 1], [1, 1] ] )
# gini_index = get_gini_index(left, right, classes)
# print('gini_index = ', gini_index)