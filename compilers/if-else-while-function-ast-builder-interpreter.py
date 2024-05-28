"""
Recursive descent parser and AST builder for arithmetic expressions
and the following statments: assign, print, if (with optional else), while.
Simple functions are supported.
An AST interpreter is included.

Tweaked for correct -,/ associativity, new grammar is:

Program → F_declarations Stmt_list
F_declarations → F_declaration F_declarations | ε
F_declaration → function id ( Param_list ) Block_stmt
Param_list → id Param_tail | ε
Param_tail → , id Param_tail | ε
Stmt_list → Stmt Stmt_list | ε
Stmt → id Assign_or_call | print Expr
     | if Expr Block_stmt (else Block_stmt)? | while Expr Block_stmt
     | return Expr
Assign_or_call → = Expr | ( Arg_list )
Arg_list → Expr Arg_tail | ε
Arg_tail → , Expr Arg_tail | ε     
Block_stmt → Stmt | { Stmt_list }
Expr → Term (Addop Term)*
Term → Factor (Multop Factor)*
Factor → ( Expr ) | id Id_or_call | number
Id_or_call → ( Arg_list ) | ε
Addop → + | -
Multop → * | /

Any expression evaluating to 0 is considered False. If evaluating to non zero
is considered True.
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
        
        # function table, keyed by func-id ("0" indicates main program)
        # values are a dict with a list of param ids and the function's AST
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
        self.Program()
                
        # keep the following to match end-of-text
        self.match(None)
        
        return self.function_table


    def Program(self):
        
        if self.next_symbol.token in ('id','print','if','while','function','return',None):
            # Program → F_declarations Stmt_list
            self.F_declarations()
            sl = self.Stmt_list()

            # store main program's AST in function table
            self.function_table['0'] = {'parameters':[],'ast':sl} 
        
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Program(), expecting id, print,if, while, function, return or EOT, found {self.next_symbol.token} instead')    


    def F_declarations(self):
    
        if self.next_symbol.token=='function':
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

            # store function's AST in function table
            self.function_table[funcname] = {'parameters':pl,'ast':bs} 
            
            

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
            
            if sl is None:
                return s
                
            else:    # sl is a statement or a subtree of statements
                return ASTNode(subnodes=[s,sl],
                               attributes={'type':'.'})
        
        elif self.next_symbol.token in ('}',None):
            # Stmt_list → e
            return
                
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Stmt_list(), expecting id, print,if, while, return, }} or EOT, found {self.next_symbol.token} instead')


    def Stmt(self):
                
        if self.next_symbol.token=='id':
            # id Assign_or_call
            idname = self.next_symbol.lexeme
            # keep these for meaningful error reporting
            lineno = self.next_symbol.lineno
            charpos = self.next_symbol.charpos            
            self.match('id')
            mode,p = self.Assign_or_call()
            
            if mode=='assign':	# p is an expression tree
                return ASTNode(subnodes=[p],
                               attributes={'type':'ASSIGN','name':idname})

            # else, 'call', p is a list of expression trees (call arguments)
            return ASTNode(subnodes=p,
                           attributes={'type':'FCALL','name':idname,
                                       'lineno':lineno,'charpos':charpos})
            
        elif self.next_symbol.token=='print':
            # Stmt → print Expr
            self.match('print')
            e = self.Expr()
            
            return ASTNode(subnodes=[e],
                           attributes={'type':'PRINT'})
                
        elif self.next_symbol.token=='if':
            # Stmt → if Expr Block_stmt (else Block_stmt)?
            self.match('if')
            e = self.Expr()
            bs = self.Block_stmt()
            
            if self.next_symbol.token=='else':
                self.match('else')
                bs2 = self.Block_stmt()
                
                return ASTNode(subnodes=[e,bs,bs2],
                           attributes={'type':'IFELSE'})
            
            return ASTNode(subnodes=[e,bs],
                           attributes={'type':'IF'})
        
        elif self.next_symbol.token=='while':
            # Stmt → while Expr Block_stmt
            self.match('while')
            e = self.Expr()
            bs = self.Block_stmt()

            return ASTNode(subnodes=[e,bs],
                           attributes={'type':'WHILE'})

        elif self.next_symbol.token=='return':
            # Stmt → return Expr
            self.match('return')
            e = self.Expr()

            return ASTNode(subnodes=[e],
                           attributes={'type':'RETURN'})
        
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Stmt(), expecting id, print, if, while or return, found {self.next_symbol.token} instead')


    def Assign_or_call(self):
        
        if self.next_symbol.token=='=':
            # Assign_or_call → = Expr
            self.match('=')
            e = self.Expr()
            return 'assign',e
  
        elif self.next_symbol.token=='(':
            # Assign_or_call → ( Arg_list )
            self.match('(')
            al = self.Arg_list()
            self.match(')')
            return 'fcall',al

        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Assign_or_call(), expecting = or (, found {self.next_symbol.token} instead')


    def Arg_list(self):
    
        if self.next_symbol.token in ('(','id','number'):
            # Arg_list → Expr Arg_tail
            e = self.Expr()
            al = self.Arg_tail()
            return [e] + al
            
        elif self.next_symbol.token==')':
            # Arg_list → ε
            return []
            
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Arg_list(), expecting (, id, number or ), found {self.next_symbol.token} instead')            
    
    
    def Arg_tail(self):
        
        if self.next_symbol.token==',':
            # Arg_tail → , Expr Arg_tail
            self.match(',')
            e = self.Expr()
            al = self.Arg_tail()
            return [e] + al
            
        elif self.next_symbol.token==')':
            # Arg_tail → ε
            return []

        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Arg_tail(), expecting , or ), found {self.next_symbol.token} instead')            
    

    def Block_stmt(self):
    
        if self.next_symbol.token in ('id','print','if','while','return'):
            # Block_stmt → Stmt
            return self.Stmt()
            
        elif self.next_symbol.token=='{':
            # Block_stmt → { Stmt_list }
            self.match('{')
            sl = self.Stmt_list()
            self.match('}')
            
            return sl
        
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Block_stmt(), expecting id, print, if, while, return or {{, found {self.next_symbol.token} instead')
        

    def Expr(self):
                
        if self.next_symbol.token in ('(','id','number'):
            # Expr → Term (Addop Term)*
            t = self.Term()

            while self.next_symbol.token in ('+','-'):
                op = self.Addop()
                t2 = self.Term()
                
                t = ASTNode(subnodes=[t,t2],
                            attributes={'type':'OP','func':op})
            
            return t
                                
        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Expr(), expecting (, id or number, found {self.next_symbol.token} instead')    
            

    def Term(self):
                
        if self.next_symbol.token in ('(','id','number'):
            # Term → Factor (Multop Factor)*
            f = self.Factor()
            
            while self.next_symbol.token in ('*','/'):
                op = self.Multop()
                f2 = self.Factor()
                
                f = ASTNode(subnodes=[f,f2],
                            attributes={'type':'OP','func':op})

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
            # Factor → id Id_or_call
            idname = self.next_symbol.lexeme
            # keep these for meaningful error reporting
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
            
        elif self.next_symbol.token in ('*','/','+','-','{','}',',',')','function','id','print','if','while','return','else',None):
            # Id_or_call → ε	**NOTE** "else" must be included in follow set here!
            return

        else:
            raise ParseError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Id_or_call(), expecting (,  *, /, +, -,  {{, }}, ,, ), function, id, print, if, while, return, else or EOT, found {self.next_symbol.token} instead')


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
        
        # if an AST for "main" exists and is not empty, start execution
        if '0' in self.function_table and self.function_table['0']['ast'] is not None:
            self.execute_statement(self.function_table['0']['ast'])


    def execute_statement(self,astnode):
    
        if astnode is None:	# empty statement AST
            return ControlFlow.NORMAL
    
        if astnode.attributes['type']=='ASSIGN':
            self.rs.assign(astnode.attributes['name'],
                           self.evaluate_expression(astnode.subnodes[0]))
            return ControlFlow.NORMAL  
        
        elif astnode.attributes['type']=='PRINT':
            print(self.evaluate_expression(astnode.subnodes[0]))
            return ControlFlow.NORMAL

        elif astnode.attributes['type']=='IF':
            ce = self.evaluate_expression(astnode.subnodes[0])
            if ce!=0:
                return self.execute_statement(astnode.subnodes[1])
            return ControlFlow.NORMAL

        elif astnode.attributes['type']=='IFELSE':
            ce = self.evaluate_expression(astnode.subnodes[0])
            if ce!=0:
                return self.execute_statement(astnode.subnodes[1])
            else:
                return self.execute_statement(astnode.subnodes[2])

        elif astnode.attributes['type']=='WHILE':
            ce = self.evaluate_expression(astnode.subnodes[0])
            while ce!=0:
                cf = self.execute_statement(astnode.subnodes[1])
                if cf==ControlFlow.RETURN:
                    return cf
                ce = self.evaluate_expression(astnode.subnodes[0])
            
            return ControlFlow.NORMAL
            
        elif astnode.attributes['type']=='FCALL':
            self.call_function(astnode)	# called as "procedure", no rv expected
            return ControlFlow.NORMAL
                                          
        elif astnode.attributes['type']=='RETURN':
            self.rs.return_value = self.evaluate_expression(astnode.subnodes[0])
            return ControlFlow.RETURN   
            
        elif astnode.attributes['type']=='.':
            # a concat node, execute left-right children statements
            cf = self.execute_statement(astnode.subnodes[0])
            if cf==ControlFlow.RETURN:
                return cf
            return self.execute_statement(astnode.subnodes[1])
     
        else:
            raise RunError(f"Run error: Unknown AST node type {astnode.attributes['type']}")  

    def evaluate_expression(self,astnode):
    
        if astnode.attributes['type']=='NUMBER':
            return astnode.attributes['value']
        
        elif astnode.attributes['type']=='DEREF':
            varname = astnode.attributes['name']
            value = self.rs.dereference(varname)
            if value is None:	# symbol not found in current frame
                lineno = astnode.attributes['lineno']
                charpos = astnode.attributes['charpos']
                raise RunError(f'Run error at line {lineno} char {charpos}: Uninitialized variable {varname}')            
            
            return value

        elif astnode.attributes['type']=='FCALL':
            rv = self.call_function(astnode)	# called as "function", rv expected
            if rv is None:	# function returned without setting a return value
                func_name = astnode.attributes['name']
                lineno = astnode.attributes['lineno']
                charpos = astnode.attributes['charpos']
            
                raise RunError(f'Run error at line {lineno} char {charpos}: Call to function {func_name} did not return a value as expected')
        
            return rv
                    
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
        param_list = ftable_entry['parameters']
        arg_list = astnode.subnodes
             
        if len(param_list)!=len(arg_list):	
            raise RunError(f'Run error at line {lineno} char {charpos}: Call to function {func_name} with wrong number of arguments')
                
        # calculate expressions in argument list
        arg_values = []
        for expr in arg_list:
            arg_values.append(self.evaluate_expression(expr))
            
        # push a new call frame with arguments installed
        self.rs.push_frame(zip(param_list,arg_values))
            
        # execute function AST
        self.execute_statement(ftable_entry['ast'])
        
        # retrieve return value (or None if no return statement)
        rv = self.rs.return_value
            
        # pop call frame
        self.rs.pop_frame()
                
        # return value (or None)
        return rv
        
        
        
# main part of program


# create tokenizer and define token patterns
tokenizer = Tokenizer()
tokenizer.pattern(r'[0-9]+(\.[0-9]+)?','number')
tokenizer.pattern('[-+*/=(){},]',TokenAction.TEXT)
tokenizer.pattern('[_a-zA-Z][_a-zA-Z0-9]*','id',
                  keywords=['print','if','while','else','function','return'])
tokenizer.pattern(r'\s+',TokenAction.IGNORE)
tokenizer.pattern('.',TokenAction.ERROR)

# input text
text = """
function um(x)
  return 0-x  

function cube(x) {
  return x*x*x
}

function f(a,b,c) {
  print a-b-c
}

function factorial(i) {
  if i-1 return i*factorial(i-1)  
  return 1.0
}

a = 2 + 7.55*44
print a
if a-7 {
  b = 3*(a-99.01)
  it = 5
  while it {
    print it+b*0.23
    it = it - 1
  }
}
c = 5-3-2
f(a,b,c)
if it+1 if c print c else print cube(um(c+28))
print factorial(4)
if it+1 {}
"""    
        
# create scanner for input text
scanner = tokenizer.scan(text)

# create recursive descent parser
parser = MyParser(scanner)

try:
    function_table = parser.parse()
    for key,value in function_table.items():
        print(f"----- {key}{value['parameters']} :")
        print(value['ast'])
        
    
except(TokenizerError,ParseError) as e:
    print(e)
    
else:    # if no lexical or syntax error
    
    # create AST interpreter
    interpreter = MyInterpreter()
    
    try:
        interpreter.run(function_table)
    
    except RunError as e:
        print(e)
            

