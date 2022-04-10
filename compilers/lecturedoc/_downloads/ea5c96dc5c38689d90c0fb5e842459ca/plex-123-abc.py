import plex


text_abc = plex.Str('abc')
text_123 = plex.Str('123')
spaces = plex.Rep1(plex.Any(' \t\n'))

lexicon = plex.Lexicon([
        (text_abc,'ABC_TOKEN'),
        (text_123,'123_TOKEN'),
        (spaces,plex.IGNORE)
      ])

with open("plex-123-abc.txt","r") as fp:

    scanner = plex.Scanner(lexicon,fp)

    while True:
        token,lexeme = scanner.read()
    
        if not token: break	# reached end-of-text (EOT)
    
        print(token,lexeme)	
        

