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
                case lark.Token():
                    if op.line is not None and op.column is not None:
                        loc = Location.file(self.filename, op.line, op.column)
                    else:
                        loc = Location.unknown()
                case lark.Tree(
                    lark.Token("RULE", "loop"),
                    [
                        lark.Token("LOOP_START") as loop_start_tok,
                        *_,
                        lark.Token("LOOP_END") as loop_end_tok,
                    ],
                ):
                    if (
                        loop_start_tok.line is not None
                        and loop_start_tok.column is not None
                    ):
                        if (
                            loop_end_tok.line is not None
                            and loop_end_tok.column is not None
                        ):
                            loc = Location.file(
                                self.filename,
                                loop_start_tok.line,
                                loop_start_tok.column,
                                loop_end_tok.line,
                                loop_end_tok.column,
                            )
                        else:
                            loc = Location.file(
                                self.filename,
                                loop_start_tok.line,
                                loop_start_tok.column,
                            )
                    else:
                        loc = Location.unknown()
                case other:
                    raise Exception(f"Invalid Element in AST: {other!r}")
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
