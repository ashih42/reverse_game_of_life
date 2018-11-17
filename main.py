# (☞ﾟヮﾟ)☞  Just for testing

from LogisticRegression import LogisticRegression
from LogisticRegression2 import LogisticRegression2
from LogisticRegression3 import LogisticRegression3
from exceptions import ParserException, LogisticRegressionException
from colorama import Fore, Back, Style
import sys

def main():
	# check argv
	if len(sys.argv) != 2:
		print('usage: ' + Fore.RED + 'python3' + Fore.BLUE + ' main.py ' + Fore.RESET + 'training_data.csv')
		sys.exit(-1)
	training_file = sys.argv[1]
	try:
		# logistic_regression = LogisticRegression()
		# logistic_regression = LogisticRegression2()
		logistic_regression = LogisticRegression3()
		logistic_regression.train(training_file)
	except IOError as e:
		print(Style.BRIGHT + Fore.RED + 'I/O Error: ' + Style.RESET_ALL + Fore.RESET + str(e))
	except ParserException as e:
		print(Style.BRIGHT + Fore.RED + 'ParserException: ' + Style.RESET_ALL + Fore.RESET + str(e))
	except LogisticRegressionException as e:
		print(Style.BRIGHT + Fore.RED + 'LogisticRegressionException: ' + Style.RESET_ALL + Fore.RESET + str(e))

if __name__ == '__main__':
	main()
