import numpy as np

# [[0, 0], [1, 1]], [0, 1]

# Returns well the data is split
# [ [1,0], [1,0] ], [ [1,1], [1,1] ] = 0.0 = Best Case
# [ [1,1], [1,0] ], [ [1,1], [1,0] ] = 0.5 = Worst Case

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

def split(i, val, data):
    left, right = [], []
    for row in data:
        if row[i] < val:
            left.append(row)
        else:
            right.append(row)
    return left, right

# We assume the final entry in data is the class
def get_best_split(data):
    values = list(set(x[-1] for x in data))
    print(values)
    best_split = []
    min_gini = 1000
    for i in range(len(data[0]) - 1):
        for row in data:
            groups = split(i, row[i], data)
            gini = gini_index(groups, values)
            if gini < min_gini:
                min_gini = gini
                best_split = [i, row[i], groups]
    return best_split


"""
def get_best_split(data):
    groups = test_split(index, row[index], data)
    gini = gini_index(groups, class_values)
    if gini < b_score:
        update
"""

class RandomForest():
    def __init__(self):
        self.foo = 5

def main():
    dataset = [[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,0],
	[3.961043357,2.61995032,0],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]]
    splits = get_best_split(dataset)
    print('Split: [X%d < %.3f]' % ((splits[0]+1), splits[1]))


if __name__ == '__main__':
    main()