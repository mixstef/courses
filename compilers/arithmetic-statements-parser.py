"""
Recursive descent parser for arithmetic expressions, using plex as scanner.

Grammar is:
Stmt_list → Stmt Stmt_list | ε
Stmt → id = Expr | print Expr
Expr → Term Term_tail
Term_tail → Addop Term Term_tail | ε
Term → Factor Factor_tail
Factor_tail → Multop Factor Factor_tail | ε
Factor → (Expr) | id | number
Addop → + | -
Multop → * | /
"""

import plex


# parsing error, a user-defined exception
class ParseError(Exception):
	pass


# class of recursive descent parser
class MyParser():

	def __init__(self,scanner):
			
		self.scanner = scanner
		
		# get initial input token
		self.next_token,self.next_lexeme = self.scanner.read()


	def match(self,expected):
	
		if self.next_token==expected:
			# proceed to next token
			self.next_token,self.next_lexeme = self.scanner.read()

		else:
			raise ParseError('Expected {}, found {} instead'.format(expected,self.next_token))
			
			
	def parse(self):

		# call method for starting symbol of grammar
		self.Stmt_list()
		
		# keep the following to match end-of-text
		self.match(None)


	def Stmt_list(self):
				
		if self.next_token in ('id','print'):
			# Stmt_list → Stmt Stmt_list
			self.Stmt()
			self.Stmt_list()
		
		elif self.next_token==None:
			# Stmt_list → e
			return
				
		else:
			raise ParseError("In Stmt_list(), expecting id, print or EOT, found {} instead".format(self.next_token))


	def Stmt(self):
				
		if self.next_token=='id':
			# Stmt → id = Expr
			self.match('id')
			self.match('=')
			self.Expr()

		elif self.next_token=='print':
			# Stmt → print Expr
			self.match('print')
			self.Expr()
				
		else:
			raise ParseError("In Stmt(), expecting id or print, found {} instead".format(self.next_token))
		

	def Expr(self):
				
		if self.next_token in ('(','id','number'):
			# Expr → Term Term_tail
			self.Term()
			self.Term_tail()
				
		else:
			raise ParseError("In Expr(), expecting (, id or number, found {} instead".format(self.next_token))	
			

	def Term_tail(self):
				
		if self.next_token in ('+','-'):
			# Term_tail → Addop Term Term_tail
			self.Addop()
			self.Term()
			self.Term_tail()

		elif self.next_token in ('id','print',')',None):
			# Term_tail → e
			return
				
		else:
			raise ParseError("In Term_tail(), expecting +, -, id, print, ) or EOT , found {} instead".format(self.next_token))	


	def Term(self):
				
		if self.next_token in ('(','id','number'):
			# Term → Factor Factor_tail
			self.Factor()
			self.Factor_tail()
				
		else:
			raise ParseError("In Term(), expecting (, id or number, found {} instead".format(self.next_token))			
			

	def Factor_tail(self):
				
		if self.next_token in ('*','/'):
			# Factor_tail → Multop Factor Factor_tail
			self.Multop()
			self.Factor()
			self.Factor_tail()

		elif self.next_token in ('+','-','id','print',')',None):
			# Factor_tail → e
			return
				
		else:
			raise ParseError("In Factor_tail(), expecting *, /, +, -, id, print, ) or EOT, found {} instead".format(self.next_token))	


	def Factor(self):
				
		if self.next_token=='(':
			# Factor → ( Expr )
			self.match('(')
			self.Expr()
			self.match(')')

		elif self.next_token=='id':
			# Factor → id
			self.match('id')

		elif self.next_token=='number':
			# Factor → number
			self.match('number')
				
		else:
			raise ParseError("In Factor(), expecting (, id or number, found {} instead".format(self.next_token))


	def Addop(self):
				
		if self.next_token=='+':
			# Addop → +
			self.match('+')

		elif self.next_token=='-':
			# Addop → -
			self.match('-')

		else:
			raise ParseError("In Addop(), expecting + or -, found {} instead".format(self.next_token))


	def Multop(self):
				
		if self.next_token=='*':
			# Multop → *
			self.match('*')

		elif self.next_token=='/':
			# Multop → /
			self.match('/')

		else:
			raise ParseError("In Multop(), expecting * or /, found {} instead".format(self.next_token))
		
# main part of program

# pattern definitions
letter = plex.Range("AZaz")
digit = plex.Range("09")
underscore =  plex.Str("_")

number = plex.Rep1(digit) + plex.Opt(plex.Str('.') + plex.Rep1(digit))

name = (letter | underscore) + plex.Rep(letter | digit | underscore)

operator = plex.Any('+-*/=()')

k_print = plex.Str('print')

spaces = plex.Rep1(plex.Any(' \t\n'))

# create plex lexicon
lexicon = plex.Lexicon([
	  (number,'number'),
	  (operator,plex.TEXT),
	  (k_print,'print'),
	  (name,'id'),
	  (spaces,plex.IGNORE)
	])

# open input file for parsing
with open("arithmetic.txt","r") as fp:
	
	# create plex scanner for file fp
	scanner = plex.Scanner(lexicon,fp)
	
	# create recursive descent parser
	parser = MyParser(scanner)
	
	try:
		parser.parse()
		
	except ParseError as e:
		_,lineno,charno = scanner.position()
		print('Syntax error at line:{} char:{}, {}'.format(lineno,charno+1,e))
				
	except plex.errors.PlexError:			
		_,lineno,charno = scanner.position()
		print("Scanner Error at line {} char {}".format(lineno,charno+1))


