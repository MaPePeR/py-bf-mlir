from typing import Protocol

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint, Rewriter

from ..dialects import free_brainfuck as free_bf, linked_brainfuck as linked_bf


class Visitor(Protocol):
    def enter(self, op: Operation) -> bool: ...
    def leave(self, op: Operation) -> None: ...


def walk(op: Operation, visitor: Visitor):
    if visitor.enter(op):
        for region in op.regions:
            for block in region.blocks:
                for block_op in block.ops:
                    walk(block_op, visitor)
        visitor.leave(op)


class FreeToLinkedVisitor(Visitor):
    index: list[SSAValue] = []

    def enter(self, op: Operation) -> bool:
        match op:
            case func.FuncOp(body=body):
                new_op = arith.ConstantOp(builtin.IntegerAttr(0, builtin.IndexType()))
                Rewriter.insert_op(new_op, InsertPoint.at_start(body.block))
                assert len(self.index) == 0
                self.index = [new_op.results[0]]
                return True
            case free_bf.LoopOp(body=body):
                block = body.detach_block(body.block)
                block_index_arg = block.insert_arg(linked_bf.PositionType(), 0)
                new_op = linked_bf.LoopOp(self.index[-1], Region(block))
                Rewriter.replace_op(op, new_op, [])
                self.index[-1] = new_op.results[0]
                self.index.append(block_index_arg)
                walk(new_op, self)
                self.index.pop()
                return False
            case free_bf.MoveLeftOp():
                new_op = linked_bf.MoveLeftOp(self.index[-1])
                Rewriter.replace_op(op, new_op, [])
                self.index[-1] = new_op.results[0]
            case free_bf.MoveRightOp():
                new_op = linked_bf.MoveRightOp(self.index[-1])
                Rewriter.replace_op(op, new_op, [])
                self.index[-1] = new_op.results[0]
            case free_bf.IncrementOp():
                Rewriter.replace_op(op, linked_bf.IncrementOp(self.index[-1]))
            case free_bf.DecrementOp():
                Rewriter.replace_op(op, linked_bf.DecrementOp(self.index[-1]))
            case free_bf.OutputOp():
                Rewriter.replace_op(op, linked_bf.OutputOp(self.index[-1]))
            case free_bf.InputOp():
                Rewriter.replace_op(op, linked_bf.InputOp(self.index[-1]))
            case linked_bf.LoopOp():
                return True
            case ModuleOp():
                return True
        return False

    def leave(self, op: Operation):
        match op:
            case func.FuncOp():
                self.index.pop()
            case linked_bf.LoopOp(body=body):
                body.block.add_op(linked_bf.LoopEndOp(self.index[-1]))


class LowerFreeToLinkedBfPass(ModulePass):
    """
    A pass for lowering operations in the Toy dialect to built-in dialects.
    """

    name = "lower-free-to-linked"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        walk(op, FreeToLinkedVisitor())
