import sys

import lark
from xdsl.printer import Printer

from .gen_mlir import GenMLIR
from .parser import BrainfuckParser


def main():
    if len(sys.argv) < 2:
        print("Missing arg.")
        return 1
    parser = BrainfuckParser()

    with open(sys.argv[1]) as h:
        ast = parser.parse(h.read())
    assert isinstance(ast, lark.Tree)
    assert (
        isinstance(ast.data, lark.Token)
        and ast.data.type == "RULE"
        and ast.data.value == "start"
    )
    gen = GenMLIR()
    gen.gen_main_func(ast.children)

    printer = Printer()
    printer.print_op(gen.module)


sys.exit(main())
