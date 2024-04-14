"""
Recursive descent parser and AST builder for arithmetic expressions,
followed by an AST interpreter.

Grammar is:
Stmt_list → Stmt Stmt_list | ε
Stmt → id = Expr | print Expr
Expr → Term Term_tail
Term_tail → Addop Term Term_tail | ε
Term → Factor Factor_tail
Factor_tail → Multop Factor Factor_tail | ε
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
class MyParser():

    # TODO: τοποθετήστε εδώ όλες τις μεθόδους του parser σας





# runtime error, a user-defined exception
class RunError(Exception):
    pass


# class of AST walking interpreter
class MyInterpreter():

    def __init__(self):
    
        self.symbol_table = {}


    def run(self,astnode):

        if astnode is not None:    # if not an empty AST
            self.execute_statement(astnode)


    def execute_statement(self,astnode):
    
        # TODO: προσθέστε κώδικα για την εκτέλεση των εντολών του AST
        # με διάσχιση DFS και αναδρομικές κλήσεις της execute_statement
        


    def evaluate_expression(self,astnode):
    
        # TODO: προσθέστε κώδικα για τον υπολογισμό ενός AST αριθμητικής 
        # έκφρασης με διάσχιση DFS και αναδρομικές κλήσεις της evaluate_expression


        
# main part of program


# create tokenizer and define token patterns
tokenizer = Tokenizer()

# TODO: προσθέστε τα pattern σας για την αναγνώριση των tokens





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
parser = MyParser(scanner)

try:
    ast = parser.parse()
    print(ast)
    
except (TokenizerError,ParseError) as e:
    print(e)
    
else:    # if no lexical or syntax error
    
    # create AST interpreter
    interpreter = MyInterpreter()
    
    try:
        interpreter.run(ast)
    
    except RunError as e:
        print(e)
            

