"""
Recursive descent LL(1) parser for a grammar in Logo graphics-style.
Version 3

Grammar is:

CommandList -> Command CommandList | ε
Command     -> Move | Turn | Pen | repeat num [ CommandList ]
Move        -> forward Intvalue | backward Intvalue
Turn        -> left Intvalue | right Intvalue
Pen         -> up | down
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
        
        if self.next_symbol.token in ('repeat','forward','backward','left','right','up','down'):
            # CommandList -> Command CommandList
            self.Command()
            self.CommandList()
            
        elif self.next_symbol.token in (']',None):  # FOLLOW(CommandList) = { ], EOT }
            # CommandList -> ε
            return

        else:
            self.error(f'In CommandList(), expecting repeat, forward, backward, left, right, up, down, ] or EOT, found {self.next_symbol.token} instead')
            
    
    def Command(self):
    
        if self.next_symbol.token in ('forward','backward'):
            # Command -> Move
            self.Move()
            
        elif self.next_symbol.token in ('left','right'):
            # Command -> Turn
            self.Turn()
            
        elif self.next_symbol.token in ('up','down'):
            # Command -> Pen
            self.Pen()

        elif self.next_symbol.token=='repeat':
            # Command -> repeat num [ CommandList ]
            self.match('repeat')
            self.match('num')
            self.match('[')
            self.CommandList()
            self.match(']')

        else:
            self.error(f'In Command(), expecting forward, backward, left, right, up, down or repeat, found {self.next_symbol.token} instead')
    
    
    def Move(self):

        if self.next_symbol.token=='forward':
            # Move -> forward
            self.match('forward')
            self.IntValue()
            
        elif self.next_symbol.token=='backward':
            # Move -> backward
            self.match('backward')
            self.IntValue()
    
        else:
            self.error(f'In Move(), expecting forward or backward, found {self.next_symbol.token} instead')
    
    
    def Turn(self):

        if self.next_symbol.token=='left':
            # Turn -> left
            self.match('left')
            self.IntValue()
            
        elif self.next_symbol.token=='right':
            # Turn -> right
            self.match('right')
            self.IntValue()
    
        else:
            self.error(f'In Turn(), expecting left or right, found {self.next_symbol.token} instead')    
    
    
    def Pen(self):
    
        if self.next_symbol.token=='up':
            # Pen -> up
            self.match('up')
            
        elif self.next_symbol.token=='down':
            # Pen -> down
            self.match('down')
    
        else:
            self.error(f'In Pen(), expecting up or down, found {self.next_symbol.token} instead')    
    

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

tokenizer.pattern(r'[-+\[\]]',TokenAction.TEXT)
tokenizer.pattern('[0-9]+','num')
tokenizer.pattern('forward|backward|left|right|up|down|repeat',TokenAction.TEXT)
tokenizer.pattern(r'\s+',TokenAction.IGNORE)
tokenizer.pattern('.',TokenAction.ERROR)


# input text
text = """up
forward -120
right +20
down
repeat 72 [
  repeat 5 [
    forward 100
    left 72
  ]
  right 5
]
backward 600
left -20
"""    
    
    

try:
    # create scanner for input text
    scanner = tokenizer.scan(text)

    # create recursive descent parser
    parser = MyParser(scanner)
    
    parser.parse()
    
except (TokenizerError,ParseError) as e:
    print(e)
            


