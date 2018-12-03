from exceptions import ParserException
from colorama import Fore, Back, Style

'''

Format:
- First line:			headers				ignored

(TRAINING DATA) 802 Columns:
- Column 0:				id					ignored
- Column 1:				delta:				must be in [1, 5]
- Column 2 - 401:		start cell value	must be 0 or 1
- Column 402 - 801:		stop cell value		must be 0 or 1

(TEST DATA) 402 Columns:
- Column 0:				id					ignored
- Column 1:				delta:				must be in [1, 5]
- Column 2 - 401:		stop cell value		must be 0 or 1

'''

class DataParser:
	
	def __init__(self, filename, is_training_data):
		print('Parsing data in ' + Fore.BLUE + filename + Fore.RESET)
		self.__filename = filename
		self.__num_cols = 802 if is_training_data else 402
		self.__line_number = 0
		self.data = []
		with open(filename, 'r') as data_file:
			self.__line_number += 1
			first_line = data_file.readline().strip()
			for line in data_file:
				self.__parse_line(line.strip())

	def __parse_line(self, line):
		self.__line_number += 1
		tokens = line.split(',')
		if len(tokens) != self.__num_cols:
			raise ParserException(Fore.BLUE + '[%s] ' % self.__filename + Fore.RESET +
				'Invalid number of terms at ' + Fore.GREEN + 'line %d' % (self.__line_number) + Fore.RESET +
				': ' + Fore.MAGENTA + line + Fore.RESET)
		row_data = []
		self.__parse_delta(row_data, tokens[1])
		for i in range(2, self.__num_cols):
			self.__parse_cell(row_data, tokens[i], i)
		self.data.append(row_data)

	def __parse_delta(self, row_data, expr):
		try:
			delta = int(expr)
		except ValueError:
			raise ParserException(Fore.BLUE + '[%s] ' % self.__filename + Fore.RESET +
				'Invalid delta at ' + Fore.GREEN + 'line %d' % (self.__line_number) + Fore.RESET +
				': ' + Fore.MAGENTA + expr + Fore.RESET)
		if not (1 <= delta <= 5):
			raise ParserException(Fore.BLUE + '[%s] ' % self.__filename + Fore.RESET +
				'Invalid delta at ' + Fore.GREEN + 'line %d' % (self.__line_number) + Fore.RESET +
				': ' + Fore.MAGENTA + expr + Fore.RESET)
		row_data.append(delta)

	def __parse_cell(self, row_data, expr, column_index):
		if expr == '0':
			cell_value = 0
		elif expr == '1':
			cell_value = 1
		else:
			raise ParserException(Fore.BLUE + '[%s] ' % self.__filename + Fore.RESET +
				'Invalid cell value at ' + Fore.GREEN + 'line %d, column %d' % (self.__line_number, column_index + 1) + Fore.RESET +
				': ' + Fore.MAGENTA + expr + Fore.RESET)
		row_data.append(cell_value)
