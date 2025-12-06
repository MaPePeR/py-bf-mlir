from typing import TypeAlias

import lark
from xdsl.builder import Builder
from xdsl.dialects.builtin import (
    FunctionType,
    ModuleOp,
)
from xdsl.dialects.func import FuncOp
from xdsl.ir import (
    Block,
    Region,
)
from xdsl.rewriter import InsertPoint

from .dialects.brainfuck import (
    DecrementOp,
    IncrementOp,
    InputOp,
    LoopOp,
    MoveLeftOp,
    MoveRightOp,
    OutputOp,
)

AST: TypeAlias = list[lark.Tree | lark.Token]


class GenMLIR:
    builder: Builder
    module: ModuleOp

    def __init__(self) -> None:
        self.module = ModuleOp([])
        self.builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))

    def gen_main_func(self, ast: AST):
        body = Block()
        body_builder = Builder(InsertPoint.at_end(body))
        self.gen_instructions(body_builder, ast)
        func_type = FunctionType.from_lists([], [])
        self.builder.insert(FuncOp("main", func_type, Region(body)))

    def gen_instructions(self, builder: Builder, ast: AST):
        for op in ast:
            match op:
                case lark.Token("MOVE_LEFT"):
                    builder.insert(MoveLeftOp())
                case lark.Token("MOVE_RIGHT"):
                    builder.insert(MoveRightOp())
                case lark.Token("INCREMENT"):
                    builder.insert(IncrementOp())
                case lark.Token("DECREMENT"):
                    builder.insert(DecrementOp())
                case lark.Token("OUTPUT"):
                    builder.insert(OutputOp())
                case lark.Token("INPUT"):
                    builder.insert(InputOp())
                case lark.Tree(lark.Token("RULE", "loop"), children):
                    body = Block()
                    body_builder = Builder(InsertPoint.at_end(body))
                    self.gen_instructions(body_builder, children)
                    builder.insert(LoopOp(regions=[Region(body)]))
                case other:
                    raise Exception(f"Invalid Token in AST: {other!r}")
