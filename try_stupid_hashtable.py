# (☞ﾟヮﾟ)☞  try_stupid_hashtable.py

from StupidHashtable import StupidHashtable
from LogisticRegression5 import LogisticRegression5
from exceptions import ParserException, LogisticRegressionException
from colorama import Fore, Back, Style
import sys

def main():
	# check argv
	if len(sys.argv) != 3:
		print('usage: ' + Fore.RED + 'python3' + Fore.BLUE + ' train.py ' + Fore.RESET + 'training_data.csv test_data.csv')
		sys.exit(-1)
	training_file = sys.argv[1]
	test_file = sys.argv[2]
	# initialize a new model and train on training data
	try:
		stupid_hashtable = StupidHashtable()
		stupid_hashtable.train(training_file)
		stupid_hashtable.predict_lol(test_file)
	except IOError as e:
		print(Style.BRIGHT + Fore.RED + 'I/O Error: ' + Style.RESET_ALL + Fore.RESET + str(e))
	except ParserException as e:
		print(Style.BRIGHT + Fore.RED + 'ParserException: ' + Style.RESET_ALL + Fore.RESET + str(e))
	except LogisticRegressionException as e:
		print(Style.BRIGHT + Fore.RED + 'LogisticRegressionException: ' + Style.RESET_ALL + Fore.RESET + str(e))

if __name__ == '__main__':
	main()
