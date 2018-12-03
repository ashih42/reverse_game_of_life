# (☞ﾟヮﾟ)☞  try_my_Forest.py

from MyRandomForest import MyRandomForest
from exceptions import ParserException, LogisticRegressionException
from colorama import Fore, Back, Style
import sys

def main():
	# check argv
	if len(sys.argv) != 3:
		print('usage: ' + Fore.RED + 'python3' + Fore.BLUE + ' try_my_forest.py ' + Fore.RESET + 'training_data.csv test_data.csv')
		sys.exit(-1)
	training_file = sys.argv[1]
	test_file = sys.argv[2]
	# initialize a new model and train on training data
	try:
		model = MyRandomForest()
		model.train(training_file)
		model.predict(test_file)
	except IOError as e:
		print(Style.BRIGHT + Fore.RED + 'I/O Error: ' + Style.RESET_ALL + Fore.RESET + str(e))
	except ParserException as e:
		print(Style.BRIGHT + Fore.RED + 'ParserException: ' + Style.RESET_ALL + Fore.RESET + str(e))
	except LogisticRegressionException as e:
		print(Style.BRIGHT + Fore.RED + 'LogisticRegressionException: ' + Style.RESET_ALL + Fore.RESET + str(e))

if __name__ == '__main__':
	main()
