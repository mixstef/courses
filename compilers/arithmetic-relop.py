"""
Recursive descent parser for arithmetic expressions accompanied by an AST builder-interpreter.

NOTE: modified for correct calculation for operators with left associativity
NOTE: added logical expressions support

Grammar is:
Stmt_list → Stmt Stmt_list | ε
Stmt → id = Lexpr | print Lexpr
Lexpr → Lterm Lterm_tail
Lterm_tail → or Lterm Lterm_tail | ε
Lterm → Lfactor Lfactor_tail
Lfactor_tail → and Lfactor Lfactor_tail | ε
Lfactor → not Lfactor | Expr Rest
Rest → Relop Expr | ε
Expr → Term (Addop Term)*
Term → Factor (Multop Factor)*
Factor → (Lexpr) | id | number
Addop → + | -
Multop → * | /
Relop → == | != | > | >= | < | <=
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
            # Stmt → id = Lexpr
            varname = self.next_symbol.lexeme
            self.match('id')
            self.match('=')
            e = self.Lexpr()
            
            return ASTNode(attributes={'type':'ASSIGN','name':varname},
            		                   subnodes=[e])

        elif self.next_symbol.token=='print':
            # Stmt → print Lexpr
            self.match('print')
            e = self.Lexpr()
            
            return ASTNode(attributes={'type':'PRINT'},
            		                   subnodes=[e])
                
        else:
            self.error(f'In Stmt(), expecting id or print, found {self.next_symbol.token} instead')
   
    
    def Lexpr(self):
    
        if self.next_symbol.token in ('not','(','id','number'):
            # Lexpr → Lterm Lterm_tail
            lt = self.Lterm()
            ltt = self.Lterm_tail()

            if ltt is None:
                return lt
                
            return ASTNode(attributes={'type':'OR'},
            		   subnodes=[lt,ltt]) 
            
        else:
            self.error(f'In Lexpr(), expecting not, (, id, number, found {self.next_symbol.token} instead')
    
    
    def Lterm_tail(self):
    
        if self.next_symbol.token=='or':
            # Lterm_tail → or Lterm Lterm_tail
            self.match('or')
            lt = self.Lterm()
            ltt = self.Lterm_tail()
            
            if ltt is None:
                return lt
                
            return ASTNode(attributes={'type':'OR'},
            		   subnodes=[lt,ltt])
             
        elif self.next_symbol.token in (')','id','print',None):
            # Lterm_tail → ε
            return

        else:
            self.error(f'In Lterm_tail(), expecting or, ), id, print, EOT, found {self.next_symbol.token} instead')
    
    
    def Lterm(self):
    
        if self.next_symbol.token in ('not','(','id','number'):
            # Lterm → Lfactor Lfactor_tail
            lf = self.Lfactor()
            lft = self.Lfactor_tail()
            
            if lft is None:
                return lf
                
            return ASTNode(attributes={'type':'AND'},
            		   subnodes=[lf,lft]) 

        else:
            self.error(f'In Lterm(), expecting not, (, id, number, found {self.next_symbol.token} instead')            
    
    
    def Lfactor_tail(self):
    
        if self.next_symbol.token=='and':
            # Lfactor_tail → and Lfactor Lfactor_tail
            self.match('and')
            lf = self.Lfactor()
            lft = self.Lfactor_tail()
            
            if lft is None:
                return lf
                
            return ASTNode(attributes={'type':'AND'},
            		   subnodes=[lf,lft]) 
            
        elif self.next_symbol.token in (')','or','id','print',None):
            # Lfactor_tail → ε
            return

        else:
            self.error(f'In Lfactor_tail(), expecting and, or, ), id, print, EOT, found {self.next_symbol.token} instead')
    
    
    def Lfactor(self):

        if self.next_symbol.token=='not':
            # Lfactor → not Lfactor
            self.match('not')
            f = self.Lfactor()
            
            return ASTNode(attributes={'type':'NOT'},
            		   subnodes=[f])
        
        elif self.next_symbol.token in ('(','id','number'):
            # Lfactor → Expr Rest
            e = self.Expr()
            r = self.Rest()
            
            if r is None:
                return e
                
            return ASTNode(attributes={'type':'CMP','condition':r[0]},
            		   subnodes=[e,r[1]])

        else:
            self.error(f'In Lfactor(), expecting not, (, id, number, found {self.next_symbol.token} instead')            
    
        
    def Rest(self):
    
        if self.next_symbol.token in ('==','!=','>','>=','<','<='):
            # Rest → Relop Expr
            r = self.Relop()
            e = self.Expr()
            
            return r,e
            
        elif self.next_symbol.token in (')','or','and','id','print',None):
            # Lfactor_tail → ε
            return

        else:
            self.error(f'In Rest(), expecting relop and, or, ), id, print, EOT, found {self.next_symbol.token} instead')
    
    
    def Relop(self):
    
        if self.next_symbol.token in ('==','!=','>','>=','<','<='):
        	# one of: ==, !=, >, >=, <, <=
        	relop = self.next_symbol.token
        	self.match(relop)
        	return relop
        	
        else:
            self.error(f'In Relop(), expecting relop, found {self.next_symbol.token} instead')        	
    
    
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
            self.error(f'In Expr(), expecting (, id or number, found {self.next_symbol.token} instead')    
            

    def Term(self):
                
        if self.next_symbol.token in ('(','id','number'):
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
            self.error(f'In Term(), expecting (, id or number, found {self.next_symbol.token} instead')            
            

    def Factor(self):
                
        if self.next_symbol.token=='(':
            # Factor → ( Lexpr )
            self.match('(')
            e = self.Lexpr()
            self.match(')')
            return e

        elif self.next_symbol.token=='id':
            # Factor → id
            varname = self.next_symbol.lexeme
            # keep varname position in text for future error reporting
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

            case ASTNode(attributes={'type':'AND'},subnodes=[lnode,rnode]):
                a = self.evaluate_expression(lnode)
                if a==0.0: return 0.0
                # return value of right child
                return self.evaluate_expression(rnode)
            
            case ASTNode(attributes={'type':'OR'},subnodes=[lnode,rnode]):
                # visit left child first
                a = self.evaluate_expression(lnode)
                if a!=0.0: return a
                # return value of right child
                return self.evaluate_expression(rnode)            

            case ASTNode(attributes={'type':'NOT'},subnodes=[snode]):
                a = self.evaluate_expression(snode)
                if a==0.0: return 1.0
                return 0.0

            case ASTNode(attributes={'type':'CMP','condition':rop},subnodes=[lnode,rnode]):
                # visit children first
                a = self.evaluate_expression(lnode)
                b = self.evaluate_expression(rnode)
            
                if rop=='==' and a==b:
                    return 1.0
                elif rop=='!=' and a!=b:
                    return 1.0
                elif rop=='>' and a>b:
                    return 1.0
                elif rop=='>=' and a>=b:
                    return 1.0
                elif rop=='<' and a<b:
                    return 1.0
                elif rop=='<=' and a<=b:
                    return 1.0
                    
                return 0.0
 
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
tokenizer.pattern('==|!=|>=|<=',TokenAction.TEXT)
tokenizer.pattern('[-+*/=()<>]',TokenAction.TEXT)
tokenizer.pattern('[_a-zA-Z][_a-zA-Z0-9]*','id',keywords=['print','and','or','not'])
tokenizer.pattern(r'\s+',TokenAction.IGNORE)
tokenizer.pattern('.',TokenAction.ERROR)

# input text
text = """a = 2 + 7.55*44
print a
b = 3*(a-99.01)
print b*0.23
c = 5-3-2
print c
print c > a-1255 and not (b == 56*a or c != 0 or a>=b)
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

