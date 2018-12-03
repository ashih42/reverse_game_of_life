# (☞ﾟヮﾟ)☞  predict.py

from exceptions import ParserException, SolverException
from colorama import Fore, Back, Style
import sys

from solver import Solver

def main():
	if len(sys.argv) != 8:
		print('usage: ' + Fore.RED + 'python3' + Fore.BLUE + ' predict.py ' + Fore.RESET +
			'( LR | RF ) test_data.csv param_1.dat param_2.dat param_3.dat param_4.dat param_5.dat')
		sys.exit(-1)

	model_type = sys.argv[1]
	test_file = sys.argv[2]
	param_file_1 = sys.argv[3]
	param_file_2 = sys.argv[4]
	param_file_3 = sys.argv[5]
	param_file_4 = sys.argv[6]
	param_file_5 = sys.argv[7]
	
	try:
		param_files = (param_file_1, param_file_2, param_file_3, param_file_4, param_file_5)
		solver = Solver(model_type, param_files)
		solver.predict(test_file)

	except IOError as e:
		print(Style.BRIGHT + Fore.RED + 'I/O Error: ' + Style.RESET_ALL + Fore.RESET + str(e))
	except ParserException as e:
		print(Style.BRIGHT + Fore.RED + 'ParserException: ' + Style.RESET_ALL + Fore.RESET + str(e))
	except LogisticRegressionException as e:
		print(Style.BRIGHT + Fore.RED + 'LogisticRegressionException: ' + Style.RESET_ALL + Fore.RESET + str(e))

if __name__ == '__main__':
	main()
