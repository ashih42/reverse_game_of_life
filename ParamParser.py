from exceptions import ParserException
from colorama import Fore, Back, Style

'''
Format:
442 x 400 floats, separated by commas

442 features
400 labels to predict

'''

class ParamParser:
	
	def __init__(self, filename):
		print('Parsing model parameters in ' + Fore.BLUE + filename + Fore.RESET)
		self.__line_number = 0
		self.data = []
		with open(filename, 'r') as data_file:
			for line in data_file:
				self.__parse_line(line.strip())
		if len(self.data) != 442:
			raise ParserException('Invalid number of rows of parameters')

	def __parse_line(self, line):
		self.__line_number += 1
		tokens = line.split(',')
		if len(tokens) != 400:
			raise ParserException('Invalid number of terms at ' +
				Fore.GREEN + 'line %d' % (self.__line_number) + Fore.RESET + ': ' + Fore.MAGENTA + line + Fore.RESET)
		row_data = []
		for i in range(len(tokens)):
			try:
				row_data.append(float(tokens[i]))
			except ValueError:
				raise ParserException('Invalid cell value at ' +
					Fore.GREEN + 'line %d, column %d' % (self.__line_number, i + 1) + Fore.RESET + ': ' +
					Fore.MAGENTA + tokens[i] + Fore.RESET)
		self.data.append(row_data)
