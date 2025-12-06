import typing
from typing import Optional

from ply import lex


class BrainfuckLexer:
    data: Optional[typing.TextIO]

    tokens = (
        "MOVE_RIGHT",
        "MOVE_LEFT",
        "INCREMENT",
        "DECREMENT",
        "OUTPUT",
        "INPUT",
    )
    literals = "[]"

    t_MOVE_RIGHT = r">"
    t_MOVE_LEFT = r"<"
    t_INCREMENT = r"\+"
    t_DECREMENT = r"\-"
    t_OUTPUT = r"\."
    t_INPUT = r","

    t_ignore_COMMENT = r"[^<>+.,\[\]\n-]+"

    def t_eof(self, t):
        if not self.data:
            return None
        more = self.data.read(1024)
        if more == "":
            self.data = None
            return None
        t.lexer.input(more)
        return t.lexer.token()

    def t_newline(self, t):
        r"\n+"
        t.lexer.lineno += len(t.value)

    def __init__(self, **kwargs) -> None:
        self._build(**kwargs)

    def _build(self, **kwargs):
        self.lexer = lex.lex(module=self, **kwargs)

    def t_error(self, t):
        # TODO: Calculate line column...
        raise Exception(
            f"Illegal character '{ t.value[0] }' at line { t.lineno } column ? index { t.lexpos }"
        )

    def readFrom(self, data: typing.TextIO):
        self.data = data

    def token(self) -> lex.LexToken:
        return self.lexer.token()
