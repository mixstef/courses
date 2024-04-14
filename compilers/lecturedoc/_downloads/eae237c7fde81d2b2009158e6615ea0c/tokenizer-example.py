from compilerlabs import Tokenizer,TokenAction,TokenizerError

t = Tokenizer()

t.pattern('[a-zA-Z]+','word',('print','int','def'))
t.pattern('[0-9]+','number')
t.pattern('[-+*/=]',TokenAction.TEXT)
t.pattern(r'\s+',TokenAction.IGNORE)
t.pattern('.',TokenAction.ERROR)

text = """hi 123-7 
 
   666
b + m 5 int/ 92
so78*6"""

try:
    for s in t.scan(text,eot=False):
        print(s)
except TokenizerError as e:
    print(e)
    

