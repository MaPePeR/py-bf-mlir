import sys

from .lexer import BrainfuckLexer
from .parser import BrainfuckParser


def main():
    if len(sys.argv) < 2:
        print("Missing arg.")
        return 1
    lexer = BrainfuckLexer()
    parser = BrainfuckParser().build()
    with open(sys.argv[1]) as h:
        lexer.readFrom(h)
        parsed = parser.parse(lexer=lexer)
        print(parsed)


sys.exit(main())
