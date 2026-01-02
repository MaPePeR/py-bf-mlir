import lark


def BrainfuckParser():
    return lark.Lark.open_from_package(
        __name__, "brainfuck.lark", parser="lalr", strict=True
    )
