"""
Recursive descent LL(1) parser example.

Grammar is:
S -> aB
B -> b | aBb
"""

from compilerlabs import Tokenizer,TokenAction,TokenizerError, \
                         LL1ParserBase,ParseError



# class of recursive descent parser
class MyParser(LL1ParserBase):


    def __init__(self,scanner):
            
        super().__init__(scanner)

            
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
            raise ParseError(f'In S(), expecting A_TOKEN, found {self.next_symbol.token} instead')
        
    
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
            raise ParseError(f'In B(), expecting A_TOKEN or B_TOKEN, found {self.next_symbol.token} instead')
            
            
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
            


