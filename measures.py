import numpy as np

def get_accuracy(predictions, Y):
	num_correct = np.sum(predictions == Y)
	total = Y.shape[0] * Y.shape[1]
	accuracy = num_correct / total
	return accuracy

def get_f1_score(predictions, Y):
	true_positives = np.sum((predictions == 1) & (Y == 1))
	false_positives = np.sum((predictions == 1) & (Y == 0))
	false_negatives = np.sum((predictions == 0) & (Y == 1))

	if true_positives == 0 and false_positives == 0 and false_negatives == 0:
		f1_score = 1
	elif (true_positives + false_positives) == 0 or (true_positives + false_negatives) == 0:
		f1_score = 0
	else:
		precision = true_positives / (true_positives + false_positives)
		recall = true_positives / (true_positives + false_negatives)
		if (precision + recall) == 0:
			f1_score = 0
		else:
			f1_score = 2 * precision * recall / (precision + recall)
	return f1_score
