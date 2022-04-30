"""
Example of arithmetic expression AST evaluation.
Basic operators (+, -, *, /) are supported
"""

from simpleast import ASTNode



def visit(node):
	if node.attribute('type')=='NUMBER':
		return node.attribute('value')
	
	a = visit(node.subnode(0))
	b = visit(node.subnode(1))
	if node.attribute('func')=='+':
		return a+b
	elif node.attribute('func')=='-':
		return a-b
	elif node.attribute('func')=='*':
		return a*b
	
	return a/b
	


a = ASTNode(attributes={'type':'NUMBER','value':3.14})
b = ASTNode(attributes={'type':'NUMBER','value':5.0})
c = ASTNode(subnodes=[a,b],attributes={'type':'OP','func':'+'})
d = ASTNode(attributes={'type':'NUMBER','value':11.23})
e = ASTNode(subnodes=[c,d],attributes={'type':'OP','func':'*'})

print(e)
print(visit(e))

print('=========================')

f = ASTNode(attributes={'type':'NUMBER','value':2})
g = ASTNode(attributes={'type':'NUMBER','value':1})
a = ASTNode(attributes={'type':'NUMBER','value':3})
b = ASTNode(subnodes=[f,g],attributes={'type':'OP','func':'*'})
d = ASTNode(attributes={'type':'NUMBER','value':5})
e = ASTNode(subnodes=[a,b],attributes={'type':'OP','func':'-'})
c = ASTNode(subnodes=[d,e],attributes={'type':'OP','func':'-'})

print(c)
print(visit(c))
