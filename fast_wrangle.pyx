# cython: language_level=3

import numpy as np

cdef clamp(int x, int low, int high):
	return max(0, min(x, high))

cpdef format_X(X):
	X_new = np.empty((X.shape[0] * 400, 122 + 1))

	# cdef double[:, :] X_view = X
	# cdef double[:, :] X_new_view = X_new

	print('(fast_wrangle) X_new.shape = ', X_new.shape)

	for i in range(X.shape[0]):
		print('(fast_wrangle) i = ', i)

		row_data = np.empty((400, 122))
		
		delta = X[i, 0]

		board = X[i, 1:].reshape((20, 20))
		board = np.c_[ np.zeros((20, 5)), board, np.zeros((20, 5))]
		board = np.r_[ np.zeros((5, 30)), board, np.zeros((5, 30))]

		for x in range(20):
			for y in range(20):
				x_origin = x + 5
				y_origin = y + 5
				x_low = clamp(x_origin - 5, 0, 29)
				x_high = clamp(x_origin + 5, 0, 29)
				y_low = clamp(y_origin - 5, 0, 29)
				y_high = clamp(y_origin + 5, 0, 29)

				row_data[x * 20 + y, 1:] = board[x_low:x_high+1, y_low:y_high+1].reshape((1, 121))

		row_data[:, 0] = delta

		X_new[i*400:(i+1)*400, 1:] = row_data

	X_new[:, 0] = 1

	return X_new
