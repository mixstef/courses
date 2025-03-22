"""
Recursive descent parser for arithmetic expressions accompanied by a simple statement interpreter.

NOTE: added support for %, unary +- and **(power)

Grammar is:
Stmt_list → Stmt Stmt_list | ε
Stmt → id = Expr | print Expr
Expr → Term (Addop Term)*
Term → Factor (Multop Factor)*
Factor → Addop Factor | Atom Atom_tail
Atom_tail → pow Factor | ε 
Atom → (Expr) | id | number
Addop → + | -
Multop → * | / | %
"""

from compilerlabs import Tokenizer,TokenAction,TokenizerError


# parsing error, a user-defined exception
class ParseError(Exception):
    pass


# runtime error, a user-defined exception
class RunError(Exception):
    pass

    
# class of recursive descent parser
class MyParserInterpreter():

    def __init__(self,scanner):
            
        self.scanner = scanner
        
        # get initial input token
        self.next_symbol = next(self.scanner)
        
        # dict used as variables' symbol table
        self.symbol_table = {}


    def match(self,expected):
    
        if self.next_symbol.token == expected:
            # proceed to next token, if not at end-of-text
            if self.next_symbol.token is not None:
                self.next_symbol = next(self.scanner)

        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: Expected {expected}, found {self.next_symbol.token} instead')

            
            
    def parse(self):

        # call method for starting symbol of grammar
        self.Stmt_list()
        
        # keep the following to match end-of-text
        self.match(None)


    def Stmt_list(self):
                
        if self.next_symbol.token in ('id','print'):
            # Stmt_list → Stmt Stmt_list
            self.Stmt()
            self.Stmt_list()
        
        elif self.next_symbol.token==None:
            # Stmt_list → e
            return
                
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Stmt_list(), expecting id, print or EOT, found {self.next_symbol.token} instead')


    def Stmt(self):
                
        if self.next_symbol.token=='id':
            # Stmt → id = Expr
            varname = self.next_symbol.lexeme
            self.match('id')
            self.match('=')
            self.symbol_table[varname] = self.Expr()

        elif self.next_symbol.token=='print':
            # Stmt → print Expr
            self.match('print')
            print(self.Expr())
                
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Stmt(), expecting id or print, found {self.next_symbol.token} instead')
        

    def Expr(self):
                
        if self.next_symbol.token in ('+','-','(','id','number'):
            # Expr → Term (Addop Term)*
            t = self.Term()
            while self.next_symbol.token in ('+','-'):
                op = self.Addop()
                t2 = self.Term()
                if op=='+':
                    t += t2
                else:
                    t -= t2
                           
            return t
             
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Expr(), expecting (, id or number, found {self.next_symbol.token} instead')    
            

    def Term(self):
                
        if self.next_symbol.token in ('+','-','(','id','number'):
            # Term → Factor (Multop Factor)*
            f = self.Factor()
            while self.next_symbol.token in ('*','/','%'):
                op = self.Multop()

                # keep these for meaningful error reporting in case of div/mod by 0
                lineno = self.next_symbol.lineno
                charpos = self.next_symbol.charpos
                
                f2 = self.Factor()
                if op=='*':
                    f *= f2
                elif op=='/':
                    if f2==0:
                        raise RunError(f'Runtime error at line {lineno} char {charpos}: division by zero')                
                    f /= f2
                else:
                    if f2==0:
                        raise RunError(f'Runtime error at line {lineno} char {charpos}: modulo by zero')                
                    f %= f2
                           
            return f

        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Term(), expecting (, id or number, found {self.next_symbol.token} instead')            
            

    def Factor(self):
    
        if self.next_symbol.token in ('+','-'):
            # Factor → Addop Factor
            op = self.Addop()
            f = self.Factor()
            if op=='-':
                return -f
                
            return f
            
        elif self.next_symbol.token in ('(','id','number'):
            # Factor → Atom Atom_tail
            a = self.Atom()
            
            # keep these for meaningful error reporting in case of div/mod by 0
            lineno = self.next_symbol.lineno
            charpos = self.next_symbol.charpos
            
            at = self.Atom_tail()
            
            if at is None:
                return a

            try:
                val = float(a**at)
            except:
                raise RunError(f'Runtime error at line {lineno} char {charpos}: invalid exponentation result')
                
            return val            
    
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Factor(), expecting +, -, (, id or number, found {self.next_symbol.token} instead')            
    

    def Atom_tail(self):
    
        if self.next_symbol.token=='pow':
            # Atom_tail → pow Factor
            self.match('pow')
            return self.Factor()
        
        elif self.next_symbol.token in (')','*','/','%','+','-','id','print',None):
            # Atom_tail → ε
            return
            
        else:
            raise RunError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Atom_tail(), expecting **, ), *, /, %, +, -, id, print or None, found {self.next_symbol.token} instead')            
        

    def Atom(self):
                
        if self.next_symbol.token=='(':
            # Atom → ( Expr )
            self.match('(')
            value = self.Expr()
            self.match(')')
            return value

        elif self.next_symbol.token=='id':
            # Atom → id
            varname = self.next_symbol.lexeme
            lineno = self.next_symbol.lineno
            charpos = self.next_symbol.charpos
            self.match('id')
            if varname in self.symbol_table:
                return self.symbol_table[varname]
            raise RunError(f'Runtime error at line {lineno} char {charpos}: Uninitialized variable "{varname}"')

        elif self.next_symbol.token=='number':
            # Atom → number
            value = float(self.next_symbol.lexeme)
            self.match('number')
            return value
                
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Factor(), expecting (, id or number, found {self.next_symbol.token} instead')


    def Addop(self):
                
        if self.next_symbol.token=='+':
            # Addop → +
            self.match('+')
            return '+'

        elif self.next_symbol.token=='-':
            # Addop → -
            self.match('-')
            return '-'

        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Addop(), expecting + or -, found {self.next_symbol.token} instead')


    def Multop(self):
                
        if self.next_symbol.token=='*':
            # Multop → *
            self.match('*')
            return '*'

        elif self.next_symbol.token=='/':
            # Multop → /
            self.match('/')
            return '/'

        elif self.next_symbol.token=='%':
            # Multop → /
            self.match('%')
            return '%'
            
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Multop(), expecting * or /, found {self.next_symbol.token} instead')


        
# main part of program


# create tokenizer and define token patterns
tokenizer = Tokenizer()
tokenizer.pattern(r'[0-9]+(\.[0-9]+)?','number')
tokenizer.pattern(r'\*\*','pow')
tokenizer.pattern('[-+*/=()%]',TokenAction.TEXT)
tokenizer.pattern('[_a-zA-Z][_a-zA-Z0-9]*','id',keywords=['print'])
tokenizer.pattern(r'\s+',TokenAction.IGNORE)
tokenizer.pattern('.',TokenAction.ERROR)

# input text
text = """a = 2 + 7.55*44
print a
b = 3*(a-99.01)
print b*0.23
c = 5-3-2
print c
print 35 % 6 % 2
print -(+27.32/-3.5/---8.0)
print -4**3**-2
"""    
        
# create scanner for input text
scanner = tokenizer.scan(text)

# create recursive descent parser
parser = MyParserInterpreter(scanner)

try:
    parser.parse()
    
except (TokenizerError,ParseError,RunError) as e:
    print(e)
            

