"""
Example of arithmetic expression AST traversal.
Produces textual output: the arithmetic expression in postfix notation
"""

from simpleast import ASTNode



def visit(node):
	if node.attribute('type')=='NUMBER':
		return str(node.attribute('value'))
	
	a = visit(node.subnode(0))
	b = visit(node.subnode(1))
	
	return('{} {} {}'.format(a,b,node.attribute('func')))
	


a = ASTNode(attributes={'type':'NUMBER','value':3.14})
b = ASTNode(attributes={'type':'NUMBER','value':5.0})
c = ASTNode(subnodes=[a,b],attributes={'type':'OP','func':'+'})
d = ASTNode(attributes={'type':'NUMBER','value':11.23})
e = ASTNode(subnodes=[c,d],attributes={'type':'OP','func':'*'})

print(e)
print(visit(e))
