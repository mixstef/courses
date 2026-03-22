"""
Recursive descent LL(1) parser for a grammar in Logo graphics-style.
Version 1

Grammar is:

CommandList -> [ Command CommandTail 
CommandTail -> , Command CommandTail | ] 
Command     -> forward Intvalue | backward Intvalue
             | left Intvalue | right Intvalue
             | up | down
             | repeat num CommandList             
Intvalue    -> + num | - num | num
"""

from compilerlabs import Tokenizer,TokenAction,TokenizerError, \
                         LL1ParserBase,ParseError



# class of recursive descent parser
class MyParser(LL1ParserBase):


    def __init__(self,scanner):
            
        super().__init__(scanner)

            
    def parse(self):

        # call method for starting symbol of grammar
        self.CommandList()
        
        # keep the following to match end-of-text
        self.match(None)


    def CommandList(self):
        
        # CommandList -> [ Command CommandTail
        self.match('[')
        self.Command()
        self.CommandTail()
        
    
    def CommandTail(self):
    
        if self.next_symbol.token==',':
            # CommandTail -> , Command CommandTail
            self.match(',')
            self.Command()
            self.CommandTail()
            
        elif self.next_symbol.token==']':
            # CommandTail -> ]
            self.match(']')
            
        else:
            self.error(f'In CommandTail(), expecting , or ], found {self.next_symbol.token} instead')

    
    def Command(self):
    
        if self.next_symbol.token=='forward':
            # Command -> forward Intvalue
            self.match('forward')
            self.IntValue()
            
        elif self.next_symbol.token=='backward':
            # Command -> backward Intvalue
            self.match('backward')
            self.IntValue()
            
        elif self.next_symbol.token=='left':
            # Command -> left Intvalue
            self.match('left')
            self.IntValue()

        elif self.next_symbol.token=='right':
            # Command -> right Intvalue
            self.match('right')
            self.IntValue()

        elif self.next_symbol.token=='up':
            # Command -> up
            self.match('up')

        elif self.next_symbol.token=='down':
            # Command -> down
            self.match('down')

        elif self.next_symbol.token=='repeat':
            # Command -> repeat num CommandList
            self.match('repeat')
            self.match('num')
            self.CommandList()

        else:
            self.error(f'In Command(), expecting forward, backward, left, right, up, down or repeat, found {self.next_symbol.token} instead')
    
    
    def IntValue(self):
    
        if self.next_symbol.token=='+':
            # Intvalue -> + num
            self.match('+')
            self.match('num')
            
        elif self.next_symbol.token=='-':
            # IntValue -> - num
            self.match('-')
            self.match('num')
            
        elif self.next_symbol.token=='num':
            # IntValue -> num
            self.match('num')
    
        else:
            self.error(f'In IntValue(), expecting +, -, or num, found {self.next_symbol.token} instead')
    
             
            
# main part of program


# create tokenizer and define token patterns
tokenizer = Tokenizer()

tokenizer.pattern(r'[-+\[\],]',TokenAction.TEXT)
tokenizer.pattern('[0-9]+','num')
tokenizer.pattern('forward|backward|left|right|up|down|repeat',TokenAction.TEXT)
tokenizer.pattern(r'\s+',TokenAction.IGNORE)
tokenizer.pattern('.',TokenAction.ERROR)


# input text
text = """
[ up, forward 50, down, backward -20, left +60, repeat 4 [ right 70 ], forward +120 ]
"""    
    
    

try:
    # create scanner for input text
    scanner = tokenizer.scan(text)

    # create recursive descent parser
    parser = MyParser(scanner)
    
    parser.parse()
    
except (TokenizerError,ParseError) as e:
    print(e)
            


