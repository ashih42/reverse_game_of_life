# (☞ﾟヮﾟ)☞  train_processed_data.py

from LogisticRegression4 import LogisticRegression4
from exceptions import ParserException, LogisticRegressionException
from colorama import Fore, Back, Style
import sys

def main():
	# check argv
	if len(sys.argv) != 3:
		print('usage: ' + Fore.RED + 'python3' + Fore.BLUE + ' train.py ' + Fore.RESET + 'training_data.csv')
		sys.exit(-1)
	X_file = sys.argv[1]
	Y_file = sys.argv[2]
	# initialize a new model and train on training data
	try:
		logistic_regression = LogisticRegression4()
		logistic_regression.train_processed_data(X_file, Y_file)
	except IOError as e:
		print(Style.BRIGHT + Fore.RED + 'I/O Error: ' + Style.RESET_ALL + Fore.RESET + str(e))
	except ParserException as e:
		print(Style.BRIGHT + Fore.RED + 'ParserException: ' + Style.RESET_ALL + Fore.RESET + str(e))
	except LogisticRegressionException as e:
		print(Style.BRIGHT + Fore.RED + 'LogisticRegressionException: ' + Style.RESET_ALL + Fore.RESET + str(e))

if __name__ == '__main__':
	main()
