"""
Recursive descent parser for arithmetic expressions
accompanied by a simple statement interpreter.

NOTE: modified for correct calculation for operators
      with left associativity

Grammar is:
Stmt_list → Stmt Stmt_list | ε
Stmt → id = Expr | print Expr
Expr → Term (Addop Term)*
Term → Factor (Multop Factor)*
Factor → (Expr) | id | number
Addop → + | -
Multop → * | /
"""

from compilerlabs import Tokenizer,TokenAction,TokenizerError, \
                         LL1ParserBase,ParseError


# runtime error, a user-defined exception
class RunError(Exception):
    pass


# class of recursive descent parser/interpreter
class MyParserInterpreter(LL1ParserBase):


    def __init__(self,scanner):
            
        super().__init__(scanner)
        
        # dict used as variables' symbol table
        self.symbol_table = {}
        
            
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
            self.error(f'In Stmt_list(), expecting id, print or EOT, found {self.next_symbol.token} instead')


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
            self.error(f'In Stmt(), expecting id or print, found {self.next_symbol.token} instead')
        

    def Expr(self):
                
        if self.next_symbol.token in ('(','id','number'):
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
            self.error(f'In Expr(), expecting (, id or number, found {self.next_symbol.token} instead')    
            

    def Term(self):
                
        if self.next_symbol.token in ('(','id','number'):
            # Term → Factor Factor_tail
            f = self.Factor()
            while self.next_symbol.token in ('*','/'):
                lineno = self.next_symbol.lineno
                charpos = self.next_symbol.charpos
                op = self.Multop()
                f2 = self.Factor()

                if op=='*':
                    f *= f2
                else:
                    if f2==0:
                        raise RunError(f'Runtime error at line {lineno} char {charpos}: Division by zero"')
                    
                    f /= f2
                           
            return f
                                    
        else:
            self.error(f'In Term(), expecting (, id or number, found {self.next_symbol.token} instead')            
            

    def Factor(self):
                
        if self.next_symbol.token=='(':
            # Factor → ( Expr )
            self.match('(')
            value = self.Expr()
            self.match(')')
            return value

        elif self.next_symbol.token=='id':
            # Factor → id
            varname = self.next_symbol.lexeme
            lineno = self.next_symbol.lineno
            charpos = self.next_symbol.charpos
            self.match('id')
            if varname in self.symbol_table:
                return self.symbol_table[varname]
            raise RunError(f'Runtime error at line {lineno} char {charpos}: Uninitialized variable "{varname}"')

        elif self.next_symbol.token=='number':
            # Factor → number
            value = float(self.next_symbol.lexeme)
            self.match('number')
            return value
                
        else:
            self.error(f'In Factor(), expecting (, id or number, found {self.next_symbol.token} instead')


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
            self.error(f'In Addop(), expecting + or -, found {self.next_symbol.token} instead')


    def Multop(self):
                
        if self.next_symbol.token=='*':
            # Multop → *
            self.match('*')
            return '*'

        elif self.next_symbol.token=='/':
            # Multop → /
            self.match('/')
            return '/'

        else:
            self.error(f'In Multop(), expecting * or /, found {self.next_symbol.token} instead')
            
            
            
# main part of program


# create tokenizer and define token patterns
tokenizer = Tokenizer()
tokenizer.pattern(r'[0-9]+(\.[0-9]+)?','number')
tokenizer.pattern('[-+*/=()]',TokenAction.TEXT)
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
"""    
    
try:
    # create scanner for input text
    scanner = tokenizer.scan(text)

    # create recursive descent parser
    parser = MyParserInterpreter(scanner)

    parser.parse()
    
except (TokenizerError,ParseError,RunError) as e:
    print(e)
            


