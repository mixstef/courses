"""
Recursive descent parser for arithmetic expressions accompanied by an AST builder-interpreter.

NOTE: modified for correct calculation for operators with left associativity

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
                         ASTNode


# parsing error, a user-defined exception
class ParseError(Exception):
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
        sl = self.Stmt_list()	# sl holds program's list of statement ASTs
        
        # keep the following to match end-of-text
        self.match(None)

        return sl


    def Stmt_list(self):
                
        if self.next_symbol.token in ('id','print'):
            # Stmt_list → Stmt Stmt_list
            s = self.Stmt()
            sl = self.Stmt_list()
            
            if not sl:	# sl is empty
                return [s]
                
            return [s] + sl
        
        elif self.next_symbol.token==None:
            # Stmt_list → e
            return []
                
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Stmt_list(), expecting id, print or EOT, found {self.next_symbol.token} instead')


    def Stmt(self):
                
        if self.next_symbol.token=='id':
            # Stmt → id = Expr
            varname = self.next_symbol.lexeme
            self.match('id')
            self.match('=')
            e = self.Expr()
            
            return ASTNode(attributes={'type':'ASSIGN','name':varname},
            		   subnodes=[e])

        elif self.next_symbol.token=='print':
            # Stmt → print Expr
            self.match('print')
            e = self.Expr()
            
            return ASTNode(attributes={'type':'PRINT'},
            		   subnodes=[e])
                
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Stmt(), expecting id or print, found {self.next_symbol.token} instead')
        

    def Expr(self):
                
        if self.next_symbol.token in ('(','id','number'):
            # Expr → Term (Addop Term)*
            t = self.Term()
            while self.next_symbol.token in ('+','-'):
                op = self.Addop()
                t2 = self.Term()
                
                t = ASTNode(attributes={'type':'OP','func':op},
                	    subnodes=[t,t2]) 
                           
            return t
             
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Expr(), expecting (, id or number, found {self.next_symbol.token} instead')    
            

    def Term(self):
                
        if self.next_symbol.token in ('(','id','number'):
            # Term → Factor (Multop Factor)*
            f = self.Factor()
            while self.next_symbol.token in ('*','/'):
                # keep op position for future error reporting
                lineno = self.next_symbol.lineno
                charpos = self.next_symbol.charpos
                op = self.Multop()
                f2 = self.Factor()
                
                f = ASTNode(attributes={'type':'OP','func': op,
                		        'lineno':lineno,'charpos':charpos},
                            subnodes=[f,f2])
                                          
            return f

        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Term(), expecting (, id or number, found {self.next_symbol.token} instead')            
            

    def Factor(self):
                
        if self.next_symbol.token=='(':
            # Factor → ( Expr )
            self.match('(')
            e = self.Expr()
            self.match(')')
            return e

        elif self.next_symbol.token=='id':
            # Factor → id
            varname = self.next_symbol.lexeme
            # keep id position for future error reporting
            lineno = self.next_symbol.lineno
            charpos = self.next_symbol.charpos
            self.match('id')
            return ASTNode(attributes={'type':'DEREF','name':varname,
				       'lineno':lineno,'charpos':charpos})
				       
        elif self.next_symbol.token=='number':
            # Factor → number
            value = float(self.next_symbol.lexeme)
            self.match('number')
            return ASTNode(attributes={'type':'NUMBER','value':value})
                
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

        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Multop(), expecting * or /, found {self.next_symbol.token} instead')



# runtime error, a user-defined exception
class RunError(Exception):
    pass

    
# class of AST walking interpreter
class MyInterpreter():

    def __init__(self):
    
        self.symbol_table = {}


    def run(self,ast_list):

        self.execute_statements(ast_list)


    def execute_statements(self,ast_list):
    
        for astnode in ast_list:
            if astnode.attributes['type']=='ASSIGN':
                self.symbol_table[astnode.attributes['name']] = self.evaluate_expression(astnode.subnodes[0])    
        
            elif astnode.attributes['type']=='PRINT':
                print(self.evaluate_expression(astnode.subnodes[0]))
                    

    def evaluate_expression(self,astnode):
    
        if astnode.attributes['type']=='NUMBER':
            return astnode.attributes['value']
        
        elif astnode.attributes['type']=='DEREF':
            varname = astnode.attributes['name']
            if varname in self.symbol_table:
                return self.symbol_table[varname]
            else:
                lineno = astnode.attributes['lineno']
                charpos = astnode.attributes['charpos']
                raise RunError(f'Run error at line {lineno} char {charpos}: Uninitialized variable {varname}')
        
        else:    # a binary operator, visit children first
            a = self.evaluate_expression(astnode.subnodes[0])
            b = self.evaluate_expression(astnode.subnodes[1])
            
            # process after children (post-order)
            if astnode.attributes['func']=='+':
                return a+b
            elif astnode.attributes['func']=='-':
                return a-b
            elif astnode.attributes['func']=='*':
                return a*b
            else:    # func = '/'
                if b==0:
                    lineno = astnode.attributes['lineno']
                    charpos = astnode.attributes['charpos']
                    raise RunError(f'Runtime error at line {lineno} char {charpos}: division by zero')
                    
                return a/b


        
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
        
# create scanner for input text
scanner = tokenizer.scan(text)

# create recursive descent parser
parser = MyParserInterpreter(scanner)

try:
    ast_list = parser.parse()
    
except (TokenizerError,ParseError) as e:
    print(e)
            
else:    # if no lexical or syntax error

    # create AST interpreter
    interpreter = MyInterpreter()
    
    try:
        interpreter.run(ast_list)
    
    except RunError as e:
        print(e)
    
