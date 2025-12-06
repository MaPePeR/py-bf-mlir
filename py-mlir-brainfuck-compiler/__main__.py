import sys

from .lexer import BrainfuckLexer


def main():
    if len(sys.argv) < 2:
        print("Missing arg.")
        return 1
    lexer = BrainfuckLexer()
    with open(sys.argv[1]) as h:
        lexer.readFrom(h)
        while True:
            tok = lexer.token()
            if not tok:
                break
            print(tok)


sys.exit(main())
