# (☞ﾟヮﾟ)☞  train_and_predict.py

from exceptions import ParserException, SolverException
from colorama import Fore, Back, Style
import sys

from solver import Solver

def main():
	if len(sys.argv) != 4:
		print('usage: ' + Fore.RED + 'python3' + Fore.BLUE + ' train.py ' + Fore.RESET + '( LR | DT | RF ) training_data.csv test_data.csv')
		sys.exit(-1)
	
	model_type = sys.argv[1]
	training_file = sys.argv[2]
	test_file = sys.argv[3]
	
	try:
		solver = Solver(model_type)
		solver.train(training_file)
		print()
		solver.predict(test_file)

	except IOError as e:
		print(Style.BRIGHT + Fore.RED + 'I/O Error: ' + Style.RESET_ALL + Fore.RESET + str(e))
	except ParserException as e:
		print(Style.BRIGHT + Fore.RED + 'ParserException: ' + Style.RESET_ALL + Fore.RESET + str(e))
	except SolverException as e:
		print(Style.BRIGHT + Fore.RED + 'SolverException: ' + Style.RESET_ALL + Fore.RESET + str(e))

if __name__ == '__main__':
	main()
