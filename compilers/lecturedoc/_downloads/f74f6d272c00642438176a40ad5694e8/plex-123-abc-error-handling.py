import plex


text_abc = plex.Str('abc')
text_123 = plex.Str('123')
spaces = plex.Rep1(plex.Any(' \t\n'))

lexicon = plex.Lexicon([
    (text_abc,'ABC_TOKEN'),
    (text_123,'123_TOKEN'),
    (spaces,plex.IGNORE)
    ])

with open("plex-123-abc-other.txt","r") as fp:

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
        

