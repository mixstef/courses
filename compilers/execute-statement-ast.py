"""
Example of execution of statements contained in a AST.
Statemets are of the form:
1. varname = expression
2. print expression
A concat (.) operator sequences statements in the AST.
Variables are kept in a dictionary (symbol table).
No check is performed for un-initialized variable usage.
"""

from simpleast import ASTNode


# the global symbol table
symbol_table = {}

def visit_expression(node):
	if node.attribute('type')=='NUMBER':
		return node.attribute('value')
	
	elif node.attribute('type')=='VAR':
		return symbol_table[node.attribute('name')]
	
	# else, an operator
	a = visit_expression(node.subnode(0))
	b = visit_expression(node.subnode(1))
	if node.attribute('func')=='+':
		return a+b
	elif node.attribute('func')=='-':
		return a-b
	elif node.attribute('func')=='*':
		return a*b
	
	return a/b


def visit_statement(node):
	if node.attribute('type')=='ASSIGN':
		symbol_table[node.attribute('name')] = visit_expression(node.subnode(0))	
	
	elif node.attribute('type')=='PRINT':
		print(visit_expression(node.subnode(0)))
		
	else:	# a concat node
		visit_statement(node.subnode(0))
		visit_statement(node.subnode(1))



# 1st statement: a = 5-(3-2*1)
f1 = ASTNode(attributes={'type':'NUMBER','value':2})
g1 = ASTNode(attributes={'type':'NUMBER','value':1})
a1 = ASTNode(attributes={'type':'NUMBER','value':3})
b1 = ASTNode(subnodes=[f1,g1],attributes={'type':'OP','func':'*'})
d1 = ASTNode(attributes={'type':'NUMBER','value':5})
e1 = ASTNode(subnodes=[a1,b1],attributes={'type':'OP','func':'-'})
c1 = ASTNode(subnodes=[d1,e1],attributes={'type':'OP','func':'-'})
h1 = ASTNode(subnodes=[c1],attributes={'type':'ASSIGN','name':'a'})

# 2nd statement: b = a/2
a2 = ASTNode(attributes={'type':'VAR','name':'a'})
b2 = ASTNode(attributes={'type':'NUMBER','value':2})
c2 = ASTNode(subnodes=[a2,b2],attributes={'type':'OP','func':'/'})
d2 = ASTNode(subnodes=[c2],attributes={'type':'ASSIGN','name':'b'})

# 3rd statement: print b+8
a3 = ASTNode(attributes={'type':'VAR','name':'b'})
b3 = ASTNode(attributes={'type':'NUMBER','value':8})
c3 = ASTNode(subnodes=[a3,b3],attributes={'type':'OP','func':'+'})
d3 = ASTNode(subnodes=[c3],attributes={'type':'PRINT'})

# the concatenated full AST
a0 = ASTNode(subnodes=[d2,d3],attributes={'type':'.'})
b0 = ASTNode(subnodes=[h1,a0],attributes={'type':'.'})	# AST root


print(b0)
visit_statement(b0)
