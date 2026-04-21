"""
Recursive descent parser for arithmetic expressions accompanied by an AST builder-interpreter.

NOTE: modified for correct calculation for operators with left associativity
NOTE: added unary minus/plus (constructs like ----+---+++7 are allowed)

Grammar is:
Stmt_list → Stmt Stmt_list | ε
Stmt → id = Expr | print Expr
Expr → Term (Addop Term)*
Term → Factor (Multop Factor)*
Factor → Addop Factor | Atom
Atom → (Expr) | id | number
Addop → + | -
Multop → * | /
"""

from compilerlabs import Tokenizer,TokenAction,TokenizerError, \
                         LL1ParserBase,ParseError, \
                         ASTNode


# class of recursive descent parser/AST builder
class MyParserASTBuilder(LL1ParserBase):


    def __init__(self,scanner):
            
        super().__init__(scanner)
                
            
    def parse(self):

        # call method for starting symbol of grammar
        sl = self.Stmt_list()
        
        # keep the following to match end-of-text
        self.match(None)

        return sl
        

    def Stmt_list(self):
                
        if self.next_symbol.token in ('id','print'):
            # Stmt_list → Stmt Stmt_list
            s = self.Stmt()
            sl = self.Stmt_list()
            
            return [s] + sl
        
        elif self.next_symbol.token==None:
            # Stmt_list → e
            return []
                
        else:
            self.error(f'In Stmt_list(), expecting id, print or EOT, found {self.next_symbol.token} instead')


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
            self.error(f'In Stmt(), expecting id or print, found {self.next_symbol.token} instead')
        

    def Expr(self):
                
        if self.next_symbol.token in ('+','-','(','id','number'):
            # Expr → Term (Addop Term)*
            t = self.Term()
            while self.next_symbol.token in ('+','-'):
                op = self.Addop()
                t2 = self.Term()

                t = ASTNode(attributes={'type':'OP','func':op},
                	                    subnodes=[t,t2]) 
                           
            return t

        else:
            self.error(f'In Expr(), expecting +, -, (, id or number, found {self.next_symbol.token} instead')    
            

    def Term(self):
                
        if self.next_symbol.token in ('+','-','(','id','number'):
            # Term → Factor Factor_tail
            f = self.Factor()
            while self.next_symbol.token in ('*','/'):
                # keep op position in text for future error reporting
                lineno = self.next_symbol.lineno
                charpos = self.next_symbol.charpos
                op = self.Multop()
                f2 = self.Factor()

                f = ASTNode(attributes={'type':'OP','func': op,
                		                'lineno':lineno,'charpos':charpos},
                                        subnodes=[f,f2])                         
            return f
                                    
        else:
            self.error(f'In Term(), expecting +, -, (, id or number, found {self.next_symbol.token} instead')            
            

    def Factor(self):
    
        if self.next_symbol.token in ('+','-'):
            # Factor → Addop Factor
            op = self.Addop()
            f = self.Factor()
            
            if op=='+':
                return f    # return as is
            
            return ASTNode(attributes={'type':'UMINUS'},subnodes=[f])
            
        elif self.next_symbol.token in ('(','id','number'):
            # Factor → Atom
            return self.Atom()
            
        else:
            self.error(f'In Factor(), expecting +, -, (, id or number, found {self.next_symbol.token} instead')


    def Atom(self):
                
        if self.next_symbol.token=='(':
            # Atom → ( Expr )
            self.match('(')
            e = self.Expr()
            self.match(')')
            return e

        elif self.next_symbol.token=='id':
            # Atom → id
            varname = self.next_symbol.lexeme
            # keep varname position in text for future error reporting
            lineno = self.next_symbol.lineno
            charpos = self.next_symbol.charpos
            self.match('id')
            return ASTNode(attributes={'type':'DEREF','name':varname,
				                       'lineno':lineno,'charpos':charpos}) 

        elif self.next_symbol.token=='number':
            # Atom → number
            value = float(self.next_symbol.lexeme)
            self.match('number')
            return ASTNode(attributes={'type':'NUMBER','value':value})
                
        else:
            self.error(f'In Atom(), expecting (, id or number, found {self.next_symbol.token} instead')


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
            match astnode:
                case ASTNode(attributes={'type':'ASSIGN','name':varname},subnodes=[exprnode]):
                    self.symbol_table[varname] = self.evaluate_expression(exprnode)    
        
                case ASTNode(attributes={'type':'PRINT'},subnodes=[exprnode]):
                    print(self.evaluate_expression(exprnode))
                
                case _:
                    raise RunError(f'Runtime error: Malformed AST {astnode}')    

    def evaluate_expression(self,astnode):
    
        match astnode:
            case ASTNode(attributes={'type':'NUMBER','value':value}):
                return value
                
            case ASTNode(attributes={'type':'DEREF','name':varname}):
                if varname in self.symbol_table:
                    return self.symbol_table[varname]
                else:
                    lineno = astnode.attributes['lineno']
                    charpos = astnode.attributes['charpos']
                    raise RunError(f'Run error at line {lineno} char {charpos}: Uninitialized variable {varname}')

            case ASTNode(attributes={'type':'UMINUS'},subnodes=[snode]):
                return -self.evaluate_expression(snode)
            
            case ASTNode(attributes={'type':'OP','func':func},subnodes=[lnode,rnode]):        
                # a binary operator, visit children first
                a = self.evaluate_expression(lnode)
                b = self.evaluate_expression(rnode)
                
                # process after children (post-order)
                if func=='+':
                    return a+b
                elif func=='-':
                    return a-b
                elif func=='*':
                    return a*b
                else:    # func = '/'
                    if b==0:
                        lineno = astnode.attributes['lineno']
                        charpos = astnode.attributes['charpos']
                        raise RunError(f'Runtime error at line {lineno} char {charpos}: division by zero')
                        
                    return a/b
                    
            case _:
                raise RunError(f'Runtime error: Malformed AST {astnode}')



            
# main part of program


# create tokenizer and define token patterns
tokenizer = Tokenizer()
tokenizer.pattern(r'[0-9]+(\.[0-9]+)?','number')
tokenizer.pattern('[-+*/=()]',TokenAction.TEXT)
tokenizer.pattern('[_a-zA-Z][_a-zA-Z0-9]*','id',keywords=['print'])
tokenizer.pattern(r'\s+',TokenAction.IGNORE)
tokenizer.pattern('.',TokenAction.ERROR)

# input text
text = """a = 2 + 7.55*-44
print a
b = 3*-(a-99.01)
print b*+-+0.23
c = 5-3---2
print +c
"""    
    
try:
    # create scanner for input text
    scanner = tokenizer.scan(text)

    # create recursive descent parser
    parser = MyParserASTBuilder(scanner)

    stmt_asts = parser.parse()
    
except (TokenizerError,ParseError) as e:
    print(e)

else:    # if no lexical or syntax error

    # debug print statements' ASTs
    for ix,ast in enumerate(stmt_asts):
        print(f'{ix+1}:\n{ast}')
    
    # create AST interpreter
    interpreter = MyInterpreter()
    
    try:
        interpreter.run(stmt_asts)
        
    except RunError as e:
        print(e)

