from typing import TypeAlias

import lark
from mlir.dialects import builtin, func
from mlir.ir import InsertionPoint, Location, Module, Operation

AST: TypeAlias = list[lark.Tree | lark.Token]


class GenMLIR:
    module: Module
    filename: str

    def __init__(self, filename) -> None:
        self.module = Module.create()
        self.filename = filename

    def gen_main_func(self, ast: AST):
        with InsertionPoint(self.module.body):
            func_type = builtin.FunctionType.get([], [])

            def build_body(_):
                self.gen_instructions(ast)
                func.ReturnOp([])

            func.FuncOp("main", func_type, body_builder=build_body)

    def gen_instructions(self, ast: AST):
        for op in ast:
            match op:
                case lark.Token(line=int() as line, column=int() as column):
                    loc = Location.file(self.filename, line, column)
                case lark.Tree(
                    lark.Token("RULE", "loop"),
                    [
                        lark.Token(
                            "LOOP_START",
                            line=int() as start_line,
                            column=int() as start_column,
                        ),
                        *_,
                        lark.Token(
                            "LOOP_END",
                            line=int() as end_line,
                            column=int() as end_column,
                        ),
                    ],
                ):
                    loc = Location.file(
                        self.filename, start_line, start_column, end_line, end_column
                    )
                case _:
                    loc = Location.unknown()
            with loc:
                match op:
                    case lark.Token("MOVE_LEFT"):
                        Operation.create("bf_free.left")
                    case lark.Token("MOVE_RIGHT"):
                        Operation.create("bf_free.right")
                    case lark.Token("INCREMENT"):
                        Operation.create("bf_free.inc")
                    case lark.Token("DECREMENT"):
                        Operation.create("bf_free.dec")
                    case lark.Token("OUTPUT"):
                        Operation.create("bf_free.output")
                    case lark.Token("INPUT"):
                        Operation.create("bf_free.input")
                    case lark.Tree(
                        lark.Token("RULE", "loop"),
                        [
                            lark.Token("LOOP_START") as loop_tok,
                            *children,
                            lark.Token("LOOP_END"),
                        ],
                    ):
                        op = Operation.create("bf_free.loop", regions=1)
                        with InsertionPoint(op.regions[0].blocks.append()):
                            self.gen_instructions(children)
                    case other:
                        raise Exception(f"Invalid Token in AST: {other!r}")
