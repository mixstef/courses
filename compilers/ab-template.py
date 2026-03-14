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
    
    

try:
    # create scanner for input text
    scanner = tokenizer.scan(text)

    # create recursive descent parser
    parser = MyParser(scanner)
    
    parser.parse()
    
except (TokenizerError,ParseError) as e:
    print(e)
            


