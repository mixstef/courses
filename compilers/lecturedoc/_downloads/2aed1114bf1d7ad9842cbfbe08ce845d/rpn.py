import plex

from mystack import Stack,StackError

# pattern definitions
digit = plex.Range('09')
number = plex.Rep1(digit) + plex.Opt(plex.Str('.') + plex.Rep1(digit))
operator = plex.Any('+-*/?')
spaces = plex.Rep1(plex.Any(' \t\n'))

# the scanner lexicon
lexicon = plex.Lexicon([
      (number,'NUMBER'),
      (operator,'OPERATOR'),
      (spaces,plex.IGNORE)
    ])

# functions per operator
def add(stack):
    b = stack.pop()
    a = stack.pop()
    stack.push(a+b)
    
def sub(stack):
    b = stack.pop()	# a,b order matters!
    a = stack.pop()
    stack.push(a-b)

def mult(stack):
    b = stack.pop()
    a = stack.pop()
    stack.push(a*b)

def div(stack):
    b = stack.pop()	# a,b order matters!
    a = stack.pop()
    stack.push(a/b)
    
def prn(stack):
    a = stack.pop()
    print(a)
    stack.push(a)	# put back item in stack

# dict of operator functions
fn = {'+':add, '-':sub, '*':mult, '/':div, '?':prn }

with open('rpn.txt','r') as fp:

    scanner = plex.Scanner(lexicon,fp)

    stack = Stack()

    while True:
    
        try:       
            token,lexeme = scanner.read()
            
            if token=='NUMBER':
                stack.push(float(lexeme))	# push into stack arithmetic value
            
            elif token=='OPERATOR':
                fn[lexeme](stack)	# call operator's function
            
        except plex.errors.PlexError:            
            _,lineno,charno = scanner.position()
            print("Scanner Error at line {} char {}".format(lineno,charno+1))
            break	# lexical analysis ends after errors
            
        except StackError:        
            _,lineno,charno = scanner.position()
            print("Input error at line {} char {}: stack is empty".format(lineno,charno+1))
            break	# lexical analysis ends after errors
            
        if not token: break	# reached end-of-text (EOT)

