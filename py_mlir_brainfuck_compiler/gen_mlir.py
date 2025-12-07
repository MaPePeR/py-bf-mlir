from typing import TypeAlias

import lark
from xdsl.builder import Builder
from xdsl.dialects import builtin, func
from xdsl.ir import (
    Block,
    Region,
)
from xdsl.rewriter import InsertPoint

from .dialects import free_brainfuck as bf

AST: TypeAlias = list[lark.Tree | lark.Token]


class GenMLIR:
    builder: Builder
    module: builtin.ModuleOp

    def __init__(self) -> None:
        self.module = builtin.ModuleOp([])
        self.builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))

    def gen_main_func(self, ast: AST):
        body = Block()
        body_builder = Builder(InsertPoint.at_end(body))
        self.gen_instructions(body_builder, ast)
        body_builder.insert(func.ReturnOp())
        func_type = builtin.FunctionType.from_lists([], [])
        self.builder.insert(func.FuncOp("main", func_type, Region(body)))

    def gen_instructions(self, builder: Builder, ast: AST):
        for op in ast:
            match op:
                case lark.Token("MOVE_LEFT"):
                    builder.insert(bf.MoveLeftOp())
                case lark.Token("MOVE_RIGHT"):
                    builder.insert(bf.MoveRightOp())
                case lark.Token("INCREMENT"):
                    builder.insert(bf.IncrementOp())
                case lark.Token("DECREMENT"):
                    builder.insert(bf.DecrementOp())
                case lark.Token("OUTPUT"):
                    builder.insert(bf.OutputOp())
                case lark.Token("INPUT"):
                    builder.insert(bf.InputOp())
                case lark.Tree(lark.Token("RULE", "loop"), children):
                    body = Block()
                    body_builder = Builder(InsertPoint.at_end(body))
                    self.gen_instructions(body_builder, children)
                    builder.insert(bf.LoopOp(regions=[Region(body)]))
                case other:
                    raise Exception(f"Invalid Token in AST: {other!r}")
