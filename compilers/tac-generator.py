"""
Recursive descent parser for arithmetic expressions accompanied by a TAC generator.
Output is to stdout.

NOTE: modified for correct calculation for operators with left associativity
NOTE: added logical expressions support
NOTE: added if/ifelse/while support

Grammar is:
Stmt_list → Stmt Stmt_list | ε
Stmt → id = Lexpr | print Lexpr
     | if Lexpr Block_stmt (else Block_stmt)? 
     | while Lexpr Block_stmt
Block_stmt → Stmt | { Stmt_list }
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
                
        if self.next_symbol.token in ('id','print','if','while'):
            # Stmt_list → Stmt Stmt_list
            s = self.Stmt()
            sl = self.Stmt_list()
            
            return [s] + sl
        
        elif self.next_symbol.token in ('}',None):
            # Stmt_list → e
            return []
                
        else:
            self.error(f'In Stmt_list(), expecting id, print, if, while, }}, or EOT, found {self.next_symbol.token} instead')


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

        elif self.next_symbol.token=='if':
            # Stmt → if Lexpr Block_stmt (else Block_stmt)? 
            self.match('if')
            l = self.Lexpr()
            bs = self.Block_stmt()
            
            #test if optional part (else Block_stmt) follows
            if self.next_symbol.token=='else':
                self.match('else')
                bs2 = self.Block_stmt()
                
                return ASTNode(attributes={'type':'IFELSE'},
                               subnodes=[l,bs,bs2])
                            
            return ASTNode(attributes={'type':'IF'},
                           subnodes=[l,bs])
        
        elif self.next_symbol.token=='while':
            # Stmt → while Lexpr Block_stmt
            self.match('while')
            l = self.Lexpr()
            bs = self.Block_stmt()

            return ASTNode(attributes={'type':'WHILE'},
                           subnodes=[l,bs])

                
        else:
            self.error(f'In Stmt(), expecting id, print, if or while, found {self.next_symbol.token} instead')
   
    
    def Block_stmt(self):
    
        if self.next_symbol.token=='{':
            # Block_stmt → { Stmt_list }
            self.match('{')
            sl = self.Stmt_list()
            self.match('}')
            
            return sl

        elif self.next_symbol.token in ('id','print','if','while'):
            # Block_stmt → Stmt
            s = self.Stmt()
            
            return [s]

        else:
            self.error(f'In Block_stmt(), expecting id, print, if, while or {{, found {self.next_symbol.token} instead')                    
    
    
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
             
        elif self.next_symbol.token in (')','id','print',None,'{','}','if','while','else'):
            # Lterm_tail → ε
            return

        else:
            self.error(f'In Lterm_tail(), expecting or, ), id, print, EOT, {{, }}, if, while, else, found {self.next_symbol.token} instead')
    
    
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
            
        elif self.next_symbol.token in (')','or','id','print',None,'{','}','if','while','else'):
            # Lfactor_tail → ε
            return

        else:
            self.error(f'In Lfactor_tail(), expecting and, or, ), id, print, EOT, {{, }}, if, while, else, found {self.next_symbol.token} instead')
    
    
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
            
        elif self.next_symbol.token in (')','or','and','id','print',None,'{','}','if','while','else'):
            # Lfactor_tail → ε
            return

        else:
            self.error(f'In Rest(), expecting relop and, or, ), id, print, EOT, {{, }}, if, while, else, found {self.next_symbol.token} instead')
    
    
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
     
            
class TACGenError(Exception):
    pass           


# class of AST walking TAC generator
class MyTACGenerator():

    def __init__(self):
    
        # counter for label/temporary names generation
        self.label_count = 0
        self.temp_count = 0
        
        # array of produced TAC
        self.tac = []
        
        
    def new_temp(self):
    
        new_name = f'.T{self.temp_count}'
        self.temp_count += 1
        return new_name


    def new_label(self):
    
        new_name = f'.L{self.label_count}'
        self.label_count += 1
        return new_name


    def emit(self,instr):
    
        self.tac.append(instr)

    def pretty_print_tac(self):
    
        for line in self.tac:
            if line.startswith('label'):
                print(f'{line[6:]}:')
            else:
                print(f'\t{line}')


    def generate_tac(self,ast_list):
    
        self.generate_statements(ast_list)        
        return self.tac    


    def generate_statements(self,ast_list):
    
        for astnode in ast_list:
            match astnode:
                case ASTNode(attributes={'type':'ASSIGN','name':varname},subnodes=[exprnode]):
                    self.generate_expression_ev(exprnode,varname)
        
                case ASTNode(attributes={'type':'PRINT'},subnodes=[exprnode]):
                    tname = self.new_temp()
                    self.generate_expression_ev(exprnode,tname)
                    self.tac.append(f'call @print({tname})')

                case ASTNode(attributes={'type':'IF'},subnodes=[exprnode,ast_list]):
                    ltrue = self.new_label()
                    lfalse = self.new_label()
                    self.generate_expression_cf(exprnode,ltrue,lfalse)
                    self.emit(f'label {ltrue}')
                    self.generate_statements(ast_list)
                    self.emit(f'label {lfalse}')
                    
                case ASTNode(attributes={'type':'WHILE'},subnodes=[exprnode,ast_list]):
                    lentry = self.new_label()
                    self.emit(f'label {lentry}')
                    ltrue = self.new_label()
                    lfalse = self.new_label()
                    self.generate_expression_cf(exprnode,ltrue,lfalse)
                    self.emit(f'label {ltrue}')
                    self.generate_statements(ast_list)
                    self.emit(f'goto {lentry}')
                    self.emit(f'label {lfalse}')
 
                case ASTNode(attributes={'type':'IFELSE'},subnodes=[exprnode,if_ast_list,else_ast_list]):
                    ltrue = self.new_label()
                    lfalse = self.new_label()
                    self.generate_expression_cf(exprnode,ltrue,lfalse)
                    self.emit(f'label {ltrue}')
                    self.generate_statements(if_ast_list)
                    lexit = self.new_label()
                    self.emit(f'goto {lexit}')
                    self.emit(f'label {lfalse}')
                    self.generate_statements(else_ast_list)
                    self.emit(f'label {lexit}')
                
                case _:
                    raise TACGenError(f'TACGen error: Malformed AST {astnode}')
                      

    def generate_expression_ev(self,astnode,dest):
    
        match astnode:
            case ASTNode(attributes={'type':'NUMBER','value':value}):
                self.emit(f'{dest} = {value}')
                
            case ASTNode(attributes={'type':'DEREF','name':varname}):
                self.emit(f'{dest} = {varname}')
                
            case ASTNode(attributes={'type':'AND'},subnodes=[lnode,rnode]):
                tname1 = self.new_temp()
                self.generate_expression_ev(lnode,tname1)
                lfalse = self.new_label()
                self.emit(f'ifzero {tname1} goto {lfalse}')
                tname2 = self.new_temp()
                self.generate_expression_ev(rnode,tname2)
                self.emit(f'ifzero {tname2} goto {lfalse}')
                self.emit(f'{dest} = {tname2}')     # python-style semantics of "and" operator
                lexit = self.new_label()
                self.emit(f'goto {lexit}')
                self.emit(f'label {lfalse}')
                self.emit(f'{dest} = 0.0')
                self.emit(f'label {lexit} ')                
                            
            case ASTNode(attributes={'type':'OR'},subnodes=[lnode,rnode]):
                tname1 = self.new_temp()
                self.generate_expression_ev(lnode,tname1)
                ltrue = self.new_label()
                self.emit(f'ifnotzero {tname1} goto {ltrue}')
                tname2 = self.new_temp()
                self.generate_expression_ev(rnode,tname2)
                self.emit(f'{dest} = {tname2}')     # python-style semantics of "or" operator
                lexit = self.new_label()
                self.emit(f'goto {lexit}')
                self.emit(f'label {ltrue}')
                self.emit(f'{dest} = {tname1}')
                self.emit(f'label {lexit} ')

            case ASTNode(attributes={'type':'NOT'},subnodes=[snode]):
                ltrue = self.new_label()
                lfalse = self.new_label()
                self.generate_expression_cf(snode,ltrue,lfalse)
                tname = self.new_temp()
                self.emit(f'label {lfalse}')
                self.emit(f'{dest} = 1.0')
                lexit = self.new_label()
                self.emit(f'goto {lexit}')
                self.emit(f'label {ltrue}')
                self.emit(f'{dest} = 0.0')
                self.emit(f'label {lexit}')                                

            case ASTNode(attributes={'type':'CMP','condition':rop},subnodes=[lnode,rnode]):
                tname1 = self.new_temp()
                self.generate_expression_ev(lnode,tname1)
                tname2 = self.new_temp()
                self.generate_expression_ev(rnode,tname2)
                lname1 = self.new_label()
                lname2 = self.new_label()
                self.emit(f'if {tname1} {rop} {tname2} goto {lname1}')
                self.emit(f'{dest} = 0.0')
                self.emit(f'goto {lname2}')
                self.emit(f'label {lname1}')
                self.emit(f'{dest} = 1.0')
                self.emit(f'label {lname2}')
             
            case ASTNode(attributes={'type':'OP','func':func},subnodes=[lnode,rnode]):
                tname1 = self.new_temp()
                self.generate_expression_ev(lnode,tname1)
                tname2 = self.new_temp()
                self.generate_expression_ev(rnode,tname2)
                self.emit(f'{dest} = {tname1} {func} {tname2}')
                    
            case _:
                raise TACGenError(f'TACGen error: Malformed AST {astnode}')


    def generate_expression_cf(self,astnode,ltrue,lfalse):
    
        match astnode:
            case ASTNode(attributes={'type':'NUMBER','value':value}):
                # true or false is known here
                if value!=0.0:
                    self.emit(f'goto {ltrue}')
                else:
                    self.emit(f'goto {lfalse}')
                    
            case ASTNode(attributes={'type':'DEREF','name':varname}):
                self.emit(f'ifzero {varname} goto {lfalse}')
                self.emit(f'goto {ltrue}')
                
            case ASTNode(attributes={'type':'AND'},subnodes=[lnode,rnode]):
                l = self.new_label()
                self.generate_expression_cf(lnode,l,lfalse)
                self.emit(f'label {l}')
                self.generate_expression_cf(rnode,ltrue,lfalse)
                            
            case ASTNode(attributes={'type':'OR'},subnodes=[lnode,rnode]):
                l = self.new_label()
                self.generate_expression_cf(lnode,ltrue,l)
                self.emit(f'label {l}')
                self.generate_expression_cf(rnode,ltrue,lfalse)

            case ASTNode(attributes={'type':'NOT'},subnodes=[snode]):
                self.generate_expression_cf(snode,lfalse,ltrue)

            case ASTNode(attributes={'type':'CMP','condition':rop},subnodes=[lnode,rnode]):
                tname1 = self.new_temp()
                self.generate_expression_ev(lnode,tname1)
                tname2 = self.new_temp()
                self.generate_expression_ev(rnode,tname2)
                self.emit(f'if {tname1} {rop} {tname2} goto {ltrue}')
                self.emit(f'goto {lfalse}')
             
            case ASTNode(attributes={'type':'OP','func':func},subnodes=[lnode,rnode]):
                tname1 = self.new_temp()
                self.generate_expression_ev(lnode,tname1)
                tname2 = self.new_temp()
                self.generate_expression_ev(rnode,tname2)
                tresult = self.new_temp()
                self.emit(f'{tresult} = {tname1} {func} {tname2}')
                self.emit(f'ifzero {tresult} goto {lfalse}')
                self.emit(f'goto {ltrue}')
                    
            case _:
                raise TACGenError(f'TACGen error: Malformed AST {astnode}')



            
# main part of program


# create tokenizer and define token patterns
tokenizer = Tokenizer()
tokenizer.pattern(r'[0-9]+(\.[0-9]+)?','number')
tokenizer.pattern('==|!=|>=|<=',TokenAction.TEXT)
tokenizer.pattern('[-+*/=()<>{}]',TokenAction.TEXT)
tokenizer.pattern('[_a-zA-Z][_a-zA-Z0-9]*','id',keywords=['print',
                                                          'and','or','not',
                                                          'if','while','else'])
tokenizer.pattern(r'\s+',TokenAction.IGNORE)
tokenizer.pattern('.',TokenAction.ERROR)

# input text
text = """a = 1
while a<20 {
  b = b+33
  a = a + 1
}
print b
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

    # create TAC generator
    generator = MyTACGenerator()
    
    try:
        generator.generate_tac(stmt_asts)
        generator.pretty_print_tac()
        
    
    except TACGenError as e:
        print(e)

