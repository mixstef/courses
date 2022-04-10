import plex


# define patterns
letter = plex.Range("AZaz")
digit = plex.Range("09")

word = plex.Rep1(letter)
number = plex.Rep1(digit)
spaces = plex.Rep1(plex.Any(' \t\n'))

# the scanner lexicon - a list of (pattern,action ) tuples
lexicon = plex.Lexicon([
    (word,"WORD_TOKEN"),
    (number,"NUMBER_TOKEN"),
    (spaces,plex.IGNORE)
    ])

with open("plex-words-numbers.txt","r") as fp:

    scanner = plex.Scanner(lexicon,fp)

    while True:
        try:
            token,lexeme = scanner.read()
            
        except plex.errors.PlexError:
            _,lineno,charno = scanner.position()	# lineno is 1-based, charno is 0-based
            print("Scanner Error at line {} char {}".format(lineno,charno+1))
            break	# lexical analysis ends after error
    
        if not token: break	# reached end-of-text (EOT)
    
        print(token,lexeme)	
        

