"""
Example of arithmetic expression AST traversal.
Produces textual output: the arithmetic expression in infix notation (with parentheses ONLY when needed)
"""

from simpleast import ASTNode



# these operators MUST be evaluated from left to right
left_assoc = {'-','/'}

# priorities of operators and numbers, 0 is highest priority
priority = {'NUMBER':0,'*':1,'/':1,'+':2,'-':2 }


def visit(node):
	if node.attribute('type')=='NUMBER':
		return 'NUMBER',str(node.attribute('value'))
	
	func_a,a = visit(node.subnode(0))
	func_b,b = visit(node.subnode(1))
	f = node.attribute('func')

	if priority[f]<priority[func_a]: fmt_a = '({})'
	else: fmt_a='{}'
	
	if priority[f]<priority[func_b]: fmt_b = '({})'
	elif priority[f]==priority[func_b] and f in left_assoc: fmt_b = '({})'
	else: fmt_b='{}'
	
	return f,(fmt_a+'{}'+fmt_b).format(a,f,b)


	
f = ASTNode(attributes={'type':'NUMBER','value':2})
g = ASTNode(attributes={'type':'NUMBER','value':1})
a = ASTNode(attributes={'type':'NUMBER','value':3})
b = ASTNode(subnodes=[f,g],attributes={'type':'OP','func':'*'})
d = ASTNode(attributes={'type':'NUMBER','value':5})
e = ASTNode(subnodes=[a,b],attributes={'type':'OP','func':'-'})
c = ASTNode(subnodes=[d,e],attributes={'type':'OP','func':'-'})

print(c)
print(visit(c)[1])

print('=========================')

f = ASTNode(attributes={'type':'NUMBER','value':2})
g = ASTNode(attributes={'type':'NUMBER','value':1})
a = ASTNode(attributes={'type':'NUMBER','value':3})
b = ASTNode(subnodes=[f,g],attributes={'type':'OP','func':'*'})
d = ASTNode(attributes={'type':'NUMBER','value':5})
e = ASTNode(subnodes=[d,a],attributes={'type':'OP','func':'-'})
c = ASTNode(subnodes=[e,b],attributes={'type':'OP','func':'-'})

print(c)
print(visit(c)[1])
