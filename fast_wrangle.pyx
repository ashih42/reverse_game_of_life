#cython: language_level=3
#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np

cdef clamp(int x, int low, int high):
	return max(0, min(x, high))

# example:
# HALF_STRIDE = 5
# looking at 11 x 11 area


HALF_STRIDE = 5
AREA_WIDTH = HALF_STRIDE * 2 + 1

cpdef format_X(X):
	
	X_new = np.empty((400, X.shape[0], AREA_WIDTH ** 2 + 2))
	# X_new = np.empty((400, X.shape[0], 81 + 2))


	# print('X_new.shape = ', X_new.shape)

	# print('(fast_wrangle) X_new.shape = ', X_new.shape)

	percentile = 10

	for i in range(X.shape[0]):
		# print('(fast_wrangle) i = ', i)

		if i == X.shape[0] // 100 * percentile:
			print('format_X() %d percent...' % percentile)
			percentile += 10


		# row_data = np.empty((X.shape[0], 122))
		
		delta = X[i, 0]

		board = X[i, 1:].reshape((20, 20))
		board = np.c_[ np.zeros((20, HALF_STRIDE)), board, np.zeros((20, HALF_STRIDE))]
		board = np.r_[ np.zeros((HALF_STRIDE, 20 + HALF_STRIDE * 2)), board, np.zeros((HALF_STRIDE, 20 + HALF_STRIDE * 2))]

		for x in range(20):
			for y in range(20):
				x_origin = x + HALF_STRIDE
				y_origin = y + HALF_STRIDE
				x_low = clamp(x_origin - HALF_STRIDE, 0, 20 + HALF_STRIDE * 2 - 1)
				x_high = clamp(x_origin + HALF_STRIDE, 0, 20 + HALF_STRIDE * 2 - 1)
				y_low = clamp(y_origin - HALF_STRIDE, 0, 20 + HALF_STRIDE * 2 - 1)
				y_high = clamp(y_origin + HALF_STRIDE, 0, 20 + HALF_STRIDE * 2 - 1)

				# row_data[x * 20 + y, 1:] = board[x_low:x_high+1, y_low:y_high+1].reshape((1, 121))

				# print('this thing shape = ', X_new[x * 20 + y, i, :].shape)

				X_new[x * 20 + y, i, 2:] = board[x_low:x_high+1, y_low:y_high+1].reshape((1, AREA_WIDTH ** 2))


		X_new[:, i, :] = delta

		# row_data[:, 0] = delta

		# X_new[i*400:(i+1)*400, 1:] = row_data

		# X_new[i, :, 1:] = row_data

	X_new[:, :, 0] = 1

	print('format_X() complete')

	return X_new

# cpdef get_train_cv_sets(X, Y, percent_train=0.7):
# 	m = X.shape[1]
		
# 	indices = np.arange(m)
# 	np.random.shuffle(indices)

# 	split_index = int(m * percent_train)
# 	train_indices = indices[:split_index]
# 	cv_indices = indices[split_index:]

# 	X_train = X[:, train_indices, :]
# 	X_cv = X[:, cv_indices, :]

# 	Y_train = Y[train_indices, :]
# 	Y_cv = Y[cv_indices, :]

# 	return X_train, Y_train, X_cv, Y_cv







