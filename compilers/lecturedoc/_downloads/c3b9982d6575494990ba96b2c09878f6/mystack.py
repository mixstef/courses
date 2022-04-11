"""
A simple module implementing a stack class
"""

# stack error, a user-defined exception
class StackError(Exception):
    pass
    
    

class Stack():
    
    def __init__(self):
        
        self.stacklist = []	# initially stack is empty
        
        
    def push(self,item):
    
        self.stacklist.append(item)	# add to the end of list
        
        
    def pop(self):
    
        if self.stacklist:	# if stack not empty
            return self.stacklist.pop()	# return last item of list
            
        raise StackError("Stack is empty")

        


