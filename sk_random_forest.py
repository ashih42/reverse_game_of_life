from sklearn.ensemble import RandomForestClassifier
import os

class Sk_RandomForest:
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
		self.__model = RandomForestClassifier(verbose=2,
			max_depth=self.__MAX_DEPTH,
			n_estimators=self.__N_TREES)
		self.__model.fit(X_train, Y_train.flatten())

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

