import sys

from .parser import BrainfuckParser


def main():
    if len(sys.argv) < 2:
        print("Missing arg.")
        return 1
    parser = BrainfuckParser()

    with open(sys.argv[1]) as h:
        print(parser.parse(h.read()))


sys.exit(main())
