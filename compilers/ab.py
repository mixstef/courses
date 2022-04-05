"""
Recursive descent parser, using plex as scanner.

Grammer is:
S -> aB
B -> b | aBb
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
		self.S()
		
		# keep the following to match end-of-text
		self.match(None)


	def S(self):
				
		if self.next_token=='A_TOKEN':
			# S -> a B
			self.match('A_TOKEN')
			self.B()
				
		else:
			raise ParseError("In S(), expecting A_TOKEN, found {} instead".format(self.next_token))
		
	
	def B(self):
			
		if self.next_token=='B_TOKEN':
			# B -> b
			self.match('B_TOKEN')
			
		elif self.next_token=='A_TOKEN':
			# B -> a B b
			self.match('A_TOKEN')
			self.B()
			self.match('B_TOKEN')
			
		else:
			raise ParseError("In B(), expecting A_TOKEN or B_TOKEN, found {} instead".format(self.next_token))
			
			
# main part of program


# create plex lexicon
lexicon = plex.Lexicon([
	(plex.Str("a"),"A_TOKEN"),
	(plex.Str("b"),"B_TOKEN"),
	(plex.Rep1(plex.Any(" \t\n")),plex.IGNORE)
	])

# open input file for parsing
with open("ab.txt","r") as fp:
	
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


