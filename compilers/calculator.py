"""
Recursive descent parser and direct calculator
for arithmetic expressions.

Manually fixed for correct - and / associativity.

Grammar used:

Expr → Term (Addop Term)*
Term → Factor (Multop Factor)*
Factor → (Expr) | number
Addop → + | -
Multop → * | /
"""

from compilerlabs import Tokenizer,TokenAction,TokenizerError



# calculator error, a user-defined exception
class CalculatorError(Exception):
    pass


# class of recursive descent parser/calculator
class MyCalculator():

    def __init__(self):

        self.tokenizer_setup()
            

    def match(self,expected):
    
        if self.next_symbol.token == expected:
            # proceed to next token, if not at end-of-text
            if self.next_symbol.token is not None:
                self.next_symbol = next(self.scanner)

        else:
            raise CalculatorError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: Expected {expected}, found {self.next_symbol.token} instead')

    
    def tokenizer_setup(self):
            
        # create tokenizer and define token patterns
        self.tokenizer = Tokenizer()
        self.tokenizer.pattern(r'[0-9]+(\.[0-9]+)?','number')
        self.tokenizer.pattern('[-+*/=()]',TokenAction.TEXT)
        self.tokenizer.pattern(r'\s+',TokenAction.IGNORE)
        self.tokenizer.pattern('.',TokenAction.ERROR)

            
    def calculate(self,text):

        # create scanner for input text
        self.scanner = self.tokenizer.scan(text)
        
        # get initial input token
        self.next_symbol = next(self.scanner)

        # call method for starting symbol of grammar
        e = self.Expr()
        
        # keep the following to match end-of-text
        self.match(None)
        
        return e
        

    def Expr(self):
                
        if self.next_symbol.token in ('(','number'):
            # Expr → Term (Addop Term)*
            t = self.Term()

            while self.next_symbol.token in ('+','-'):
                op = self.Addop()
                t2 = self.Term()
                
                if op=='+':
                    t = t + t2
                else:
                    t = t - t2
            
            return t
                                
        else:
            raise CalculatorError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Expr(), expecting ( or number, found {self.next_symbol.token} instead')    
            

    def Term(self):
                
        if self.next_symbol.token in ('(','number'):
            # Term → Factor (Multop Factor)*
            f = self.Factor()
            
            while self.next_symbol.token in ('*','/'):
                
                # keep these for meaningful error reporting in case of div by 0
                lineno = self.next_symbol.lineno
                charpos = self.next_symbol.charpos
                
                op = self.Multop()
                f2 = self.Factor()
                
                if op=='*':
                    f = f * f2
                else:
                    if f2==0:
                        raise CalculatorError(f'Calculation error at line {lineno} char {charpos}: division by zero')
                        
                    f = f / f2

            return f
                                
        else:
            raise CalculatorError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Term(), expecting (, id or number, found {self.next_symbol.token} instead')            
            

    def Factor(self):
                
        if self.next_symbol.token=='(':
            # Factor → ( Expr )
            self.match('(')
            e = self.Expr()
            self.match(')')
            return e

        elif self.next_symbol.token=='number':
            # Factor → number
            value = float(self.next_symbol.lexeme)
            self.match('number')
            return value
                
        else:
            raise CalculatorError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Factor(), expecting ( or number, found {self.next_symbol.token} instead')


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
            raise CalculatorError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Addop(), expecting + or -, found {self.next_symbol.token} instead')


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
            raise CalculatorError(f'Syntax error at line {self.next_symbol.lineno} char {self.next_symbol.charpos}: In Multop(), expecting * or /, found {self.next_symbol.token} instead')


        
# main part of program
if __name__ == "__main__":
        
    def tester(text):

        """ A helper function for testing. Creates parser and calculates/prints
        the value of expression in `text`.
    
        Test generic expression with +, - , *, / operators
        >>> tester("(3.14159+4/5)*(2-(0.45-8))")
        37.6421845
        
        Test division by 0 exception generation
        >>> tester("(3.14159+4/0)*(2-(0.45-8))") # doctest: +ELLIPSIS
        Traceback (most recent call last):
        CalculatorError: Calculation error...
        
        Test left associativity of -
        >>> tester("5-3-2")
        0.0
        
        Test left associativity of /
        >>> tester("16/4/2")
        2.0
        
        Test new operators' functionality
        >>> tester("-(105%10**2)")
        -5.0
        
        Test all operators' precedence
        >>> tester("2**-3-45%4*7")
        -6.875
        
        Test right associativity of pow
        >>> tester("4**3**2")
        262144.0
        
        Test exception generation for particular cases of exponentation
        >>> tester("(-2)**0.5") # doctest: +ELLIPSIS
        Traceback (most recent call last):
        CalculatorError: Calculation error...
        >>> tester("0**-1") # doctest: +ELLIPSIS
        Traceback (most recent call last):
        CalculatorError: Calculation error...
    
        """
    
        # create recursive descent parser/calculator
        calc = MyCalculator()
        print(calc.calculate(text))

        
    import doctest
    doctest.testmod(verbose=True,exclude_empty=True)

    
        


