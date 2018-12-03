from exceptions import ParserException
from colorama import Fore, Back, Style

'''

Parse a matrix of float values, given the matrix dimensions

'''

class MatrixDataParser:
	
	def __init__(self, filename, num_rows, num_cols):
		print('Parsing data in ' + Fore.BLUE + filename + Fore.RESET)
		self.__filename = filename
		self.__num_cols = num_cols
		self.__line_number = 0
		self.data = []
		with open(filename, 'r') as data_file:
			for line in data_file:
				self.__parse_line(line.strip())
		if len(self.data) != num_rows:
			raise ParserException(Fore.BLUE + '[%s] ' % self.__filename + Fore.RESET + 'Invalid number of rows')

	def __parse_line(self, line):
		self.__line_number += 1
		tokens = line.split(',')
		if len(tokens) != self.__num_cols:
			raise ParserException(Fore.BLUE + '[%s] ' % self.__filename + Fore.RESET +
				'Invalid number of terms at ' + Fore.GREEN + 'line %d' % (self.__line_number) + Fore.RESET +
				': ' + Fore.MAGENTA + line + Fore.RESET)
		row_data = []
		for i in range(len(tokens)):
			try:
				row_data.append(float(tokens[i]))
			except ValueError:
				raise ParserException(Fore.BLUE + '[%s] ' % self.__filename + Fore.RESET +
					'Invalid cell value at ' + Fore.GREEN + 'line %d, column %d' % (self.__line_number, i + 1) + Fore.RESET +
					': ' + Fore.MAGENTA + tokens[i] + Fore.RESET)
		self.data.append(row_data)
