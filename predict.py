# (☞ﾟヮﾟ)☞  predict.py

from LogisticRegression4 import LogisticRegression4
from exceptions import ParserException, LogisticRegressionException
from colorama import Fore, Back, Style
import sys

def main():
	# check argv
	if len(sys.argv) != 3:
		print('usage: ' + Fore.RED + 'python3' + Fore.BLUE + ' predict.py ' + Fore.RESET + 'param.dat test_data.csv')
		sys.exit(-1)
	param_file = sys.argv[1]
	test_file = sys.argv[2]
	# load model parameters and make predictions on test data
	try:
		logistic_regression = LogisticRegression4(param_file)
		logistic_regression.predict(test_file)
	except IOError as e:
		print(Style.BRIGHT + Fore.RED + 'I/O Error: ' + Style.RESET_ALL + Fore.RESET + str(e))
	except ParserException as e:
		print(Style.BRIGHT + Fore.RED + 'ParserException: ' + Style.RESET_ALL + Fore.RESET + str(e))
	except LogisticRegressionException as e:
		print(Style.BRIGHT + Fore.RED + 'LogisticRegressionException: ' + Style.RESET_ALL + Fore.RESET + str(e))

if __name__ == '__main__':
	main()
