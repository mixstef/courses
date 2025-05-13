"""
Recursive descent parser for arithmetic/logic expressions accompanied by an AST builder-interpreter. If, ifelse & while statements and functions are supported.

NOTE: modified for correct calculation for arithmetic operators with left associativity

Grammar is:
Program → F_declarations Stmt_list
F_declarations → F_declaration F_declarations | ε
F_declaration → function id ( Param_list ) Block_stmt
Param_list → id Param_tail | ε
Param_tail → , id Param_tail | ε
Stmt_list → Stmt Stmt_list | ε
Stmt → id = Lexpr | print Lexpr
     | if Lexpr Block_stmt (else Block_stmt)? 
     | while Lexpr Block_stmt | return Lexpr
Block_stmt → Stmt | { Stmt_list }
Lexpr → Lterm Lterm_tail
Lterm_tail → or Lexpr | ε		# or Lterm Lterm_tail | ε
Lterm → Lfactor Lfactor_tail
Lfactor_tail → and Lterm | ε		# and Lfactor Lfactor_tail | ε
Lfactor → not Lfactor | Expr Rest
Rest → Relop Expr | ε
Expr → Term (Addop Term)*
Term → Factor (Multop Factor)*
Factor → (Lexpr) | id Id_or_call | number
Id_or_call → ( Arg_list ) | ε
Arg_list → Lexpr Arg_tail | ε
Arg_tail → , Lexpr Arg_tail | ε 
Addop → + | -
Multop → * | /
Relop → == | != | > | >= | < | <=
"""

from enum import Enum

from compilerlabs import Tokenizer,TokenAction,TokenizerError, \
                         ASTNode,RuntimeStack


# parsing error, a user-defined exception
class ParseError(Exception):
    pass


# class of recursive descent parser
class MyParser():

    def __init__(self,scanner):
            
        self.scanner = scanner
        
        # get initial input token
        self.next_symbol = next(self.scanner)
        
        # function table, keyed by func-id (empty str '' indicates main program)
        # values are tuples (list of param ids, function's ASTnodes list)
        self.function_table = {}


    def match(self,expected):
    
        if self.next_symbol.token == expected:
            # proceed to next token, if not at end-of-text
            if self.next_symbol.token is not None:
                self.next_symbol = next(self.scanner)

        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: Expected {expected}, found {self.next_symbol.token} instead')

            
            
    def parse(self):

        # call method for starting symbol of grammar
        self.Program()	# builds function table
        
        # keep the following to match end-of-text
        self.match(None)

        return self.function_table


    def Program(self):
    
        if self.next_symbol.token in ('function','id','print','if',
                                      'while','return',None):
            # Program → F_declarations Stmt_list
            self.F_declarations()
            sl = self.Stmt_list()
    
            # store main program's AST in function table
            self.function_table[''] = ([],sl)	# no parameters in "main"
            
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Program(), expecting id, print,if, while, function, return or EOT, found {self.next_symbol.token} instead')    
             
             
    def F_declarations(self):
    
        if self.next_symbol.token == 'function':
            # F_declarations → F_declaration F_declarations
            self.F_declaration()
            self.F_declarations()
            
        elif self.next_symbol.token in ('id','print','if','while','return',None):
            # F_declarations → ε
            return
            
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In F_declarations(), expecting id, print,if, while, function, return or EOT, found {self.next_symbol.token} instead')    


    def F_declaration(self):
        
        if self.next_symbol.token=='function':
            # F_declaration → function id ( Param_list ) Block_stmt
            self.match('function')

            funcname = self.next_symbol.lexeme
            # check for function re-declaration
            if funcname in self.function_table:
                raise ParseError(f'Function {funcname} redeclaration error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}')
            
            self.match('id')
            self.match('(')
            pl = self.Param_list()
            self.match(')')
            bs = self.Block_stmt()

            # store function's parameters and AST list in function table
            self.function_table[funcname] = (pl,bs.subnodes)             

        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In F_declaration(), expecting function, found {self.next_symbol.token} instead')    


    def Param_list(self):
    
        if self.next_symbol.token=='id':
            # Param_list → id Param_tail
            paramid = self.next_symbol.lexeme
            self.match('id')
            pt = self.Param_tail()
            return [paramid] + pt
            
        elif self.next_symbol.token==')':
            # Param_list → ε
            return []
            
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Param_list(), expecting id or ), found {self.next_symbol.token} instead')            
    
    
    def Param_tail(self):
        
        if self.next_symbol.token==',':
            # Param_tail → , id Param_tail
            self.match(',')
            paramid = self.next_symbol.lexeme
            self.match('id')
            pt = self.Param_tail()
            return [paramid] + pt
            
        elif self.next_symbol.token==')':
            # Param_tail → ε
            return []

        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Param_tail(), expecting , or ), found {self.next_symbol.token} instead')            
             
  
    def Stmt_list(self):
                
        if self.next_symbol.token in ('id','print','if','while','return'):
            # Stmt_list → Stmt Stmt_list
            s = self.Stmt()
            sl = self.Stmt_list()
            
            if not sl:	# sl is empty
                return [s]
                
            return [s] + sl
        
        elif self.next_symbol.token in ('}',None):
            # Stmt_list → e
            return []
                
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Stmt_list(), expecting id, print, if, while, return, }} or EOT, found {self.next_symbol.token} instead')


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
                
        elif self.next_symbol.token=='return':
            # Stmt → return Lexpr
            self.match('return')
            l = self.Lexpr()

            return ASTNode(subnodes=[l],
                           attributes={'type':'RETURN'})

        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Stmt(), expecting id, print, if, while, return found {self.next_symbol.token} instead')


    def Block_stmt(self):
    
        if self.next_symbol.token=='{':
            # Block_stmt → { Stmt_list }
            self.match('{')
            sl = self.Stmt_list()
            self.match('}')
            
            return ASTNode(attributes={'name':'BS'},
                           subnodes=sl)

        elif self.next_symbol.token in ('id','print','if','while','return'):
            # Block_stmt → Stmt
            s = self.Stmt()
            
            return ASTNode(attributes={'type':'BS'},
                           subnodes=[s])

        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Block_stmt(), expecting id, print, if, while, return or {{, found {self.next_symbol.token} instead')                    

    def Lexpr(self):
    
        if self.next_symbol.token in ('not','(','id','number'):
            # Lexpr → Lterm Lterm_tail
            t = self.Lterm()
            tt = self.Lterm_tail()

            if tt is None:
                return t
                
            return ASTNode(attributes={'type':'OR'},
            		   subnodes=[t,tt]) 
            
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Lexpr(), expecting not, (, id, number, found {self.next_symbol.token} instead')


    def Lterm_tail(self):
    
        if self.next_symbol.token=='or':
            # Lterm_tail → or Lexpr		# or Lterm Lterm_tail
            self.match('or')
            return self.Lexpr()
             
        elif self.next_symbol.token in (')','{','}',',','id','print','if',
                                        'else','while','return','function',None):
            # Lterm_tail → ε
            return

        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Lterm_tail(), expecting or, ), {{, }}, id, print, if, else, while, function, return, comma, EOT, found {self.next_symbol.token} instead')


    def Lterm(self):
    
        if self.next_symbol.token in ('not','(','id','number'):
            # Lterm → Lfactor Lfactor_tail
            f = self.Lfactor()
            ft = self.Lfactor_tail()
            
            if ft is None:
                return f
                
            return ASTNode(attributes={'type':'AND'},
            		   subnodes=[f,ft]) 

        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Lterm(), expecting not, (, id, number, found {self.next_symbol.token} instead')            


    def Lfactor_tail(self):
    
        if self.next_symbol.token=='and':
            # Lfactor_tail → and Lterm		# and Lfactor Lfactor_tail
            self.match('and')
            return self.Lterm()
            
        elif self.next_symbol.token in (')','{','}',',','or','id','print','if',
                                        'else','while','function','return',None):
            # Lfactor_tail → ε
            return

        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Lfactor_tail(), expecting and, or, ), {{, }}, id, print, if, else, while, function, return, comma, EOT, found {self.next_symbol.token} instead')


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
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Lfactor(), expecting not, (, id, number, found {self.next_symbol.token} instead')            


    def Rest(self):
    
        if self.next_symbol.token in ('==','!=','>','>=','<','<='):
            # Rest → Relop Expr
            r = self.Relop()
            e = self.Expr()
            
            return r,e
            
        elif self.next_symbol.token in (')','{','}',',','or','and','id','print',
                                        'if','while','function','return','else', 
                                        None):
            # Lfactor_tail → ε
            return

        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Rest(), expecting relop and, or, ), {{, }}, id, print, if, else, while, function, return, comma, EOT, found {self.next_symbol.token} instead')


    def Relop(self):
    
        if self.next_symbol.token in ('==','!=','>','>=','<','<='):
        	# one of: ==, !=, >, >=, <, <=
        	relop = self.next_symbol.token
        	self.match(relop)
        	return relop
        	
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Relop(), expecting relop , found {self.next_symbol.token} instead')        	
     
 
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
            # Factor → ( Lexpr )
            self.match('(')
            l = self.Lexpr()
            self.match(')')
            return l

        elif self.next_symbol.token=='id':
            # Factor → id Id_or_call
            idname = self.next_symbol.lexeme
            # keep id position for future error reporting
            lineno = self.next_symbol.lineno
            charpos = self.next_symbol.charpos
            self.match('id')
            p = self.Id_or_call()
            
            if p is None:	# variable access
                return ASTNode(attributes={'type':'DEREF','name':idname,
                                           'lineno':lineno,'charpos':charpos})
            
            # else, p is a list of expression trees (call arguments)
            return ASTNode(subnodes=p,
                           attributes={'type':'FCALL','name':idname,
                                       'lineno':lineno,'charpos':charpos})
            
        elif self.next_symbol.token=='number':
            # Factor → number
            value = float(self.next_symbol.lexeme)
            self.match('number')
            return ASTNode(attributes={'type':'NUMBER','value':value})
                
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Factor(), expecting (, id or number, found {self.next_symbol.token} instead')


    def Id_or_call(self):
    
        if self.next_symbol.token=='(':
            # Id_or_call → ( Arg_list )
            self.match('(')
            al = self.Arg_list()
            self.match(')')
            return al
            
        elif self.next_symbol.token in ('*','/','+','-','{','}',',',')',
        				'==','!=','>','>=','<','<=','and','or',
                                        'function','id','print','if',
                                        'while','return','else',None):
            # Id_or_call → ε	**NOTE** "else" must be included in follow set here!
            return

        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Id_or_call(), expecting (,  *, /, +, -,  {{, }}, comma, relops, and, or, ), function, id, print, if, while, return, else or EOT, found {self.next_symbol.token} instead')


    def Arg_list(self):
    
        if self.next_symbol.token in ('not','(','id','number'):
            # Arg_list → Lexpr Arg_tail
            e = self.Lexpr()
            al = self.Arg_tail()
            return [e] + al
            
        elif self.next_symbol.token==')':
            # Arg_list → ε
            return []
            
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Arg_list(), expecting not, (, id, number or ), found {self.next_symbol.token} instead')            
    
    
    def Arg_tail(self):
        
        if self.next_symbol.token==',':
            # Arg_tail → , Lexpr Arg_tail
            self.match(',')
            e = self.Lexpr()
            al = self.Arg_tail()
            return [e] + al
            
        elif self.next_symbol.token==')':
            # Arg_tail → ε
            return []

        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Arg_tail(), expecting , or ), found {self.next_symbol.token} instead')            
    


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


ControlFlow = Enum('ControlFlow','NORMAL RETURN')

# runtime error, a user-defined exception
class RunError(Exception):
    pass

    
# class of AST walking interpreter
class MyInterpreter():

    def __init__(self):
    
        # create runitme stack and push "main" frame
        self.rs = RuntimeStack()


    def run(self,function_table):

        self.function_table = function_table
        
        # if an AST for "main" exists in function table, execute its statements
        if '' in self.function_table:
            self.execute_statements(self.function_table[''][1])


    def execute_statements(self,ast_list):
    
        for astnode in ast_list:
            if astnode.attributes['type']=='ASSIGN':
                self.rs.assign(astnode.attributes['name'],
                               self.evaluate_expression(astnode.subnodes[0]))    
        
            elif astnode.attributes['type']=='PRINT':
                print(self.evaluate_expression(astnode.subnodes[0]))

            elif astnode.attributes['type']=='IF':
                if self.evaluate_expression(astnode.subnodes[0])!=0.0:
                    cf = self.execute_statements(astnode.subnodes[1].subnodes)
                    if cf==ControlFlow.RETURN: return cf

            elif astnode.attributes['type']=='IFELSE':
                if self.evaluate_expression(astnode.subnodes[0])!=0.0:
                    cf = self.execute_statements(astnode.subnodes[1].subnodes)
                else:
                    cf = self.execute_statements(astnode.subnodes[2].subnodes)
                if cf==ControlFlow.RETURN: return cf

            elif astnode.attributes['type']=='WHILE':
                while self.evaluate_expression(astnode.subnodes[0])!=0.0:
                    cf = self.execute_statements(astnode.subnodes[1].subnodes)
                    if cf==ControlFlow.RETURN: return cf

            elif astnode.attributes['type']=='RETURN':
                # set return value in current call frame
                self.rs.return_value = self.evaluate_expression(astnode.subnodes[0])
                return ControlFlow.RETURN
              
	
        # loop completed, no break in execution flow requested
        return ControlFlow.NORMAL


    def evaluate_expression(self,astnode):
    
        typ = astnode.attributes['type']
        if typ =='NUMBER':
            return astnode.attributes['value']
        
        elif typ=='DEREF':
            varname = astnode.attributes['name']
            value = self.rs.dereference(varname)
            if value is None:	# symbol not found in current frame
                lineno = astnode.attributes['lineno']
                charpos = astnode.attributes['charpos']
                raise RunError(f'Run error at line {lineno} char {charpos}: Uninitialized variable {varname}')            
            
            return value
        
        elif typ=='CMP':
            rop = astnode.attributes['condition']
            # visit children first
            a = self.evaluate_expression(astnode.subnodes[0])
            b = self.evaluate_expression(astnode.subnodes[1])
        
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
            
        elif typ=='NOT':
            # visit child
            a = self.evaluate_expression(astnode.subnodes[0])
            if a==0.0: return 1.0
            return 0.0
            
        elif typ=='AND':
            # visit left child first
            a = self.evaluate_expression(astnode.subnodes[0])
            if a==0.0: return 0.0
            # return value of right child
            return self.evaluate_expression(astnode.subnodes[1])

        elif typ=='OR':
            # visit left child first
            a = self.evaluate_expression(astnode.subnodes[0])
            if a!=0.0: return 1.0
            # return value of right child
            return self.evaluate_expression(astnode.subnodes[1])

        elif typ=='FCALL':
            rv = self.call_function(astnode)
            if rv is None:	# function returned without setting a return value
                func_name = astnode.attributes['name']
                lineno = astnode.attributes['lineno']
                charpos = astnode.attributes['charpos']
            
                raise RunError(f'Run error at line {lineno} char {charpos}: Call to function {func_name} did not return a value as expected')
        
            return rv
        
        else:    # a binary arithmetic operator, visit children first
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


    def call_function(self,astnode):
        func_name = astnode.attributes['name']
        lineno = astnode.attributes['lineno']
        charpos = astnode.attributes['charpos']

        # check if function name is defined 
        if func_name not in self.function_table:
            raise RunError(f'Run error at line {lineno} char {charpos}: Call to undefined function {func_name}')

        # check if formal parameter and call arg numbers match
        ftable_entry = self.function_table[func_name]
        param_list = ftable_entry[0]
        arg_list = astnode.subnodes
             
        if len(param_list)!=len(arg_list):	
            raise RunError(f'Run error at line {lineno} char {charpos}: Call to function {func_name} with wrong number of arguments')
                
        # calculate expressions in argument list
        arg_values = []
        for lexpr in arg_list:
            arg_values.append(self.evaluate_expression(lexpr))
            
        # push a new call frame with arguments installed
        self.rs.push_frame(zip(param_list,arg_values))
            
        # execute function AST
        self.execute_statements(ftable_entry[1])
        
        # retrieve return value (or None if no return statement)
        rv = self.rs.return_value
            
        # pop call frame
        self.rs.pop_frame()
                
        # send back the return value
        return rv
        
        
        
# main part of program


# create tokenizer and define token patterns
tokenizer = Tokenizer()
tokenizer.pattern(r'[0-9]+(\.[0-9]+)?','number')
tokenizer.pattern('==|!=|>=|<=',TokenAction.TEXT)
tokenizer.pattern('[-+*/=()<>{},]',TokenAction.TEXT)
tokenizer.pattern('[_a-zA-Z][_a-zA-Z0-9]*','id',keywords=['print',
                                                          'and','or','not',
                                                          'if','while','else',
                                                          'function','return'])
tokenizer.pattern(r'\s+',TokenAction.IGNORE)
tokenizer.pattern('.',TokenAction.ERROR)

# input text
text = """
function factorial(n) {
  if (n==1) return 1
  return n*factorial(n-1)
}
function fib(i) {
  if (i<2) return i
  return fib(i-1)+fib(i-2)
}
function um(x)
  return 0-x  
function cube(x) {
  return x*x*x
}
a = 2 + 7.55*44
print a
if a-7!=3 or a<355 {
  b = 3*(a-99.01)
  it = 5
  while it>0 {
    print cube(um(it+b*0.23))
    it = it - 1
  }
}
else
  print a-3.14
print factorial(3)
print fib(13)
"""    
        
# create scanner for input text
scanner = tokenizer.scan(text)

# create recursive descent parser
parser = MyParser(scanner)

try:
    function_table = parser.parse()
    
except (TokenizerError,ParseError) as e:
    print(e)
            
else:    # if no lexical or syntax error

    #for key,(paramlist,astlist) in function_table.items():
    #    print(f"----- {key}{paramlist} :")
    #    for ix,ast in enumerate(astlist):
    #        print(f'{ix+1}:\n{ast}')

        
    # create AST interpreter
    interpreter = MyInterpreter()
    
    try:
        interpreter.run(function_table)
    
    except RunError as e:
        print(e)
    
