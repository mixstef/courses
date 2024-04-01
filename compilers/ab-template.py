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
        # ...συμπληρώστε...
        
        # keep the following to match end-of-text
        self.match(None)


    def S(self):
        
        # ...συμπληρώστε...        
        
    
    def B(self):
            
        # ...συμπληρώστε...    


            
# main part of program


# create tokenizer and define token patterns
tokenizer = Tokenizer()

# ...συμπληρώστε patterns για τα Α και Β tokens...

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
            


