from ply import yacc

from .lexer import BrainfuckLexer


class BrainfuckParser:
    tokens = BrainfuckLexer.tokens
    literals = BrainfuckLexer.literals

    start = "program"

    def p_program(self, p):
        "program : code"
        p[0] = p[1]

    def p_code(self, p):
        """
        code : instruction code
             | empty
        """
        if len(p) == 3:
            # TODO: This creates an awful lot of lists
            p[0] = [p[1]] + p[2]
        elif len(p) == 2:
            p[0] = []
        else:
            raise Exception("Invalid number of arguments")

    def p_instruction_block(self, p):
        "instruction : '[' code ']'"
        p[0] = ("loop", p[2])

    def p_instruction(self, p):
        """
        instruction : MOVE_RIGHT
                     | MOVE_LEFT
                     | INCREMENT
                     | DECREMENT
                     | OUTPUT
                     | INPUT
        """
        p[0] = p[1]

    def p_empty(self, p):
        "empty :"
        pass

    def build(self) -> yacc.LRParser:
        return yacc.yacc(module=self)
