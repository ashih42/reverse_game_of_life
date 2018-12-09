#cython: language_level=3
#cython: boundscheck=False, wraparound=False, nonecheck=False

from colorama import Fore, Back, Style
import numpy as np

from data_parser import DataParser

cpdef process_data_file(filename, half_stride, is_training_data):
	parser = DataParser(filename, is_training_data)
	X, Y = wrangle_data(parser.data, half_stride, is_training_data)
	return X, Y

cpdef wrangle_data(data, half_stride, is_training_data):
	data = np.array(data, dtype=float)

	if is_training_data:
		X = data[:, 401:802]				# grab 'stop cell' values
	else:
		X = data[:, 1:401]
	X = np.c_[ data[:, 0], X ]				# grab 'delta' value
	X_new = format_X(X, half_stride)

	if is_training_data:
		Y = data[:, 1:401].reshape(-1, 1)	# grab 'start cell' values
	else:
		Y = None

	return X_new, Y

cdef clamp(int x, int low, int high):
	return max(0, min(x, high))

cpdef format_X(X, half_stride):
	print(Style.BRIGHT + 'Formatting data...' + Style.RESET_ALL)
	
	area_width = half_stride * 2 + 1
	X_new = np.empty((400 * X.shape[0], area_width ** 2 + 1))
	percentile = 10

	for i in range(X.shape[0]):
		if i == (X.shape[0] // 100 * percentile):
			print('%d%% ...' % percentile)
			percentile += 10

		delta = X[i, 0]
		board = X[i, 1:].reshape((20, 20))
		board = np.c_[ np.zeros((20, half_stride)), board, np.zeros((20, half_stride))]
		board = np.r_[ np.zeros((half_stride, 20 + half_stride * 2)), board, np.zeros((half_stride, 20 + half_stride * 2))]

		for x in range(20):
			for y in range(20):
				x_origin = x + half_stride
				y_origin = y + half_stride
				x_low = clamp(x_origin - half_stride, 0, 20 + half_stride * 2 - 1)
				x_high = clamp(x_origin + half_stride, 0, 20 + half_stride * 2 - 1)
				y_low = clamp(y_origin - half_stride, 0, 20 + half_stride * 2 - 1)
				y_high = clamp(y_origin + half_stride, 0, 20 + half_stride * 2 - 1)

				X_new[i*400 + x*20 + y, 1:] = board[x_low:x_high+1, y_low:y_high+1].reshape((1, area_width ** 2))

		X_new[i*400:(i+1)*400, 0] = delta

	print('complete!')
	return X_new
