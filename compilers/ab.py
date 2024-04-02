"""
Recursive descent parser example.

Grammar is:
S -> aB
B -> b | aBb
"""

from compilerlabs import Tokenizer,TokenAction,TokenizerError



# parsing error, a user-defined exception
class ParseError(Exception):
    pass


# class of recursive descent parser
class MyParser():

    def __init__(self,scanner):
            
        self.scanner = scanner
        
        # get initial input symbol
        self.next_symbol = next(self.scanner)


    def match(self,expected):
    
        if self.next_symbol.token == expected:
            # proceed to next token, if not at end-of-text
            if self.next_symbol.token is not None:
                self.next_symbol = next(self.scanner)

        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: Expected {expected}, found {self.next_symbol.token} instead')
            
            
    def parse(self):

        # call method for starting symbol of grammar
        self.S()
        
        # keep the following to match end-of-text
        self.match(None)


    def S(self):
                
        if self.next_symbol.token=='A_TOKEN':
            # S -> a B
            self.match('A_TOKEN')
            self.B()
                
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In S(), expecting A_TOKEN, found {self.next_symbol.token} instead')
        
    
    def B(self):
            
        if self.next_symbol.token=='B_TOKEN':
            # B -> b
            self.match('B_TOKEN')
            
        elif self.next_symbol.token=='A_TOKEN':
            # B -> a B b
            self.match('A_TOKEN')
            self.B()
            self.match('B_TOKEN')
            
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In B(), expecting A_TOKEN or B_TOKEN, found {self.next_symbol.token} instead')
            
            
# main part of program


# create tokenizer and define token patterns
tokenizer = Tokenizer()
tokenizer.pattern('a','A_TOKEN')
tokenizer.pattern('b','B_TOKEN')
tokenizer.pattern(r'\s+',TokenAction.IGNORE)
tokenizer.pattern('.',TokenAction.ERROR)


# input text
text = """
aa aabbb b
"""    
    
    
# create scanner for input text
scanner = tokenizer.scan(text)

# create recursive descent parser
parser = MyParser(scanner)

try:
    parser.parse()
    
except (TokenizerError,ParseError) as e:
    print(e)
            


