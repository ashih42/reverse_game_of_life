import numpy as np
from collections import Counter
from random import seed
from random import randrange
import math

# Decision Tree algorithm as implemented in https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

# Returns well the data is split
# [ [1,0], [1,0] ], [ [1,1], [1,1] ] = 0.0 = Best Case
# [ [1,1], [1,0] ], [ [1,1], [1,0] ] = 0.5 = Worst Case

def subsample(data, exp, ratio=0.7):
    sample = []
    sample_exp = []
    n_sample = round(len(data) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(data))
        sample.append(data[index][:])
        sample_exp.append(exp[index])
    return sample, sample_exp

def bagging_predict(trees, row):
    predictions = []
    for tree in trees:
        predictions.append(tree.predict(row))
    return max(set(predictions), key=predictions.count)

class RandomForest():

    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
    
    def fit(self, train, expected):
        assert(len(train) == len(expected))
        features = int(math.sqrt(len(train[0])))
        #features = -1
        trees = []
        for i in range(self.n_estimators):
            print('Fitting RandomForest...', i, '/', self.n_estimators)
            sample, exp = subsample(train, expected, 0.01)
            tree = DecisionTree(n_features=features)
            tree.fit(sample, exp, check_data=False)
            trees.append(tree)
        self.forest = trees
    
    def predict(self, data):
        predictions = [bagging_predict(self.forest, row) for row in data]
        return predictions


# 'Green', 3, 0
# 'Red', 6, 1
# 'Green', 6, 0
# 'Red', 3, 1

class DecisionTree():

    def __init__(self, max_depth=10, min_leaf=5, n_features=-1):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.n_features = n_features
    
    def fit(self, train, expected, check_data=True):
        if check_data:
            assert(len(train) == len(expected))
        data = train[:]
        for x, y in zip(data, expected):
            x.append(y)
        root = get_best_split(train, self.n_features)
        split(root, self.max_depth, self.min_leaf, self.n_features, 1)
        self.tree = root
    
    def predict(self, data):
        return predict_b(self.tree, data)


def gini_index(groups, values):
    total_samples = sum([len(group) for group in groups])
    result = 0.0
    for group in groups:
        count = len(group)
        if count == 0:
            continue
        proportions = []
        curr_group = [x[-1] for x in group]
        for value in values:
            proportion = curr_group.count(value) / count
            proportions.append(proportion * proportion)
        result += (1.0 - sum(proportions)) * (count / total_samples)
    return result

def split_data(i, val, data):
    left, right = [], []
    for row in data:
        if row[i] < val:
            left.append(row)
        else:
            right.append(row)
    return left, right

# We assume the final entry in data is the class
def get_best_split(data, n_features=-1):
    values = list(set(x[-1] for x in data))
    best_split = {}
    min_gini = float('Inf')
    features = []
    if n_features == -1:
        features = range(len(data[0]) - 1)
    else:
        while len(features) < n_features:
            i = randrange(len(data[0]) - 1)
            if i not in features:
                features.append(i)
    for i in features:
        for row in data:
            groups = split_data(i, row[i], data)
            gini = gini_index(groups, values)
            if gini < min_gini:
                min_gini = gini
                best_split = {'index':i, 'value':row[i], 'groups':groups}
    return best_split

# Get the most common outcome in a group to create a terminal node
# Terminal nodes are the nodes at the bottom of the branches, where the tree decides what to predict
def create_terminal_node(group):
    outcomes = [row[-1] for row in group]
    return Counter(outcomes).most_common(1)[0][0]

def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = create_terminal_node(left if left else right)
        return
    if depth >= max_depth:
        node['left'] = create_terminal_node(left)
        node['right'] = create_terminal_node(right)
        return
    if len(left) <= min_size:
        node['left'] = create_terminal_node(left)
    else:
        node['left'] = get_best_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth + 1)
    if len(right) <= min_size:
        node['right'] = create_terminal_node(right)
    else:
        node['right'] = get_best_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)

def print_tree(node, depth = 0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index'] + 1), node['value'])))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth*' ', node)))

def predict_b(node, data):
    if data[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict_b(node['left'], data)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict_b(node['right'], data)
        else:
            return node['right']
"""
#def main():
    dataset = [[2.771244718,1.784783929],
	[1.728571309,1.169761413],
	[3.678319846,2.81281357],
	[3.961043357,2.61995032],
	[2.999208922,2.209014212],
	[7.497545867,3.162953546],
	[9.00220326,3.339047188],
	[7.444542326,0.476683375],
	[10.12493903,3.234550982],
	[6.642287351,3.319983761]]

    expected = [0,0,0,0,0,1,1,1,1,1]
    
    #ml = DecisionTree()
    ml = RandomForest()

    ml.fit(dataset, expected)

    print(ml.predict(dataset))



#if __name__ == '__main__':
#    main()
"""