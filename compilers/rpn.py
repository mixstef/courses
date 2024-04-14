from compilerlabs import Tokenizer,TokenAction,TokenizerError, \
                         Stack,StackError


t = Tokenizer()
t.pattern(r'[0-9]+(\.[0-9]+)?','NUMBER')
t.pattern('[-+*/]','OPERATOR')
t.pattern('print','COMMAND')
t.pattern(r'\s+',TokenAction.IGNORE)
t.pattern('.',TokenAction.ERROR)

# functions of operators

def add(stack):
    b = stack.pop()
    a = stack.pop()
    stack.push(a+b)
    
def sub(stack):
    b = stack.pop()
    a = stack.pop()
    stack.push(a-b)

def mult(stack):
    b = stack.pop()
    a = stack.pop()
    stack.push(a*b)

def div(stack):
    b = stack.pop()
    a = stack.pop()
    stack.push(a/b)
    
    
# functions of commands

def prn(stack):
    a = stack.pop()
    print(a)
    stack.push(a)	# put back item in stack
    

# dict of operator/command functions
fn = {'+':add, '-':sub, '*':mult, '/':div, 'print':prn }


text = """
1.1 10 7 - 6 2 / * + print
"""


stack = Stack()

try:
    for symbol in t.scan(text):
        
        token = symbol.token
        lexeme = symbol.lexeme
            
        if token=='NUMBER':
            stack.push(float(lexeme))	# push into stack arithmetic value
            
        elif token=='OPERATOR' or token=='COMMAND':
            fn[lexeme](stack)	# call operator's/command's function
                            
except TokenizerError as e:
    print(e)            
            
except StackError:        
    print('Input error at line {symbol.lineno} char {symbol.charpos}: stack is empty')
            

