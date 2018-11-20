from exceptions import ParserException
from colorama import Fore, Back, Style

class MatrixDataParser:
	
	def __init__(self, filename, num_rows, num_cols):
		print('Parsing data in ' + Fore.BLUE + filename + Fore.RESET)
		self.__num_cols = num_cols
		self.__line_number = 0
		self.data = []
		with open(filename, 'r') as data_file:
			for line in data_file:
				self.__parse_line(line.strip())
		if num_rows is not None and len(self.data) != num_rows:
			raise ParserException('Invalid number of rows')

	def __parse_line(self, line):
		self.__line_number += 1
		tokens = line.split(',')
		if len(tokens) != self.__num_cols:
			print('expected num_cols = %d, got len(tokens) = %d' % (self.__num_cols, len(tokens)))
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
