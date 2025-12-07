from xdsl.context import Context
from xdsl.dialects import arith, builtin, func
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)
from xdsl.rewriter import InsertPoint

from ..dialects import free_brainfuck as free_bf, linked_brainfuck as linked_bf


class FreeToLinked(RewritePattern):
    index: list

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        new_op = None
        match op:
            case func.FuncOp(body=body):
                print("New stack")
                if (
                    not isinstance(body.block.first_op, arith.ConstantOp)
                    or body.block.first_op.results[0].name_hint != "initial_index"
                ):
                    new_op = arith.ConstantOp(
                        builtin.IntegerAttr(0, builtin.IndexType())
                    )
                    new_op.results[0].name_hint = "initial_index"
                    rewriter.insert_op(new_op, InsertPoint.at_start(body.block))
                    self.index = [new_op.results[0]]
                else:
                    self.index = [body.block.first_op.results[0]]
            case free_bf.LoopOp(body=body):
                print("Enter old loop")
                block_index_arg = rewriter.insert_block_argument(
                    body.block, 0, linked_bf.PositionType()
                )
                new_op = linked_bf.LoopOp(self.index[-1], body.clone())
                rewriter.replace_matched_op(new_op, [])
                self.index[-1] = new_op.results[0]
                self.index.append(block_index_arg)
                if new_op.body.block.is_empty:
                    rewriter.insert_op(
                        linked_bf.LoopEndOp(block_index_arg),
                        InsertPoint.at_end(new_op.body.block),
                    )
            case linked_bf.LoopOp(body=body):
                print("Enter new loop")
                self.index[-1] = op.results[0]
                self.index.append(body.block.args[0])

            case free_bf.MoveLeftOp():
                new_op = linked_bf.MoveLeftOp(self.index[-1])
                self.index[-1] = new_op.results[0]
                rewriter.replace_matched_op(new_op, [])
            case free_bf.MoveRightOp():
                new_op = linked_bf.MoveRightOp(self.index[-1])
                self.index[-1] = new_op.results[0]
                rewriter.replace_matched_op(new_op, [])
        if new_op:
            op = new_op
        if (
            not op.next_op
            and not isinstance(op, free_bf.LoopOp)
            and not isinstance(op, linked_bf.LoopOp)
            and not isinstance(op, func.FuncOp)
        ):
            if isinstance(op.parent_op(), linked_bf.LoopOp) and not isinstance(
                op, linked_bf.LoopEndOp
            ):
                rewriter.insert_op(
                    linked_bf.LoopEndOp(self.index[-1]), InsertPoint.after(op)
                )
            else:
                self.index.pop()
                print("Leaving one")


class LowerFreeToLinkedBfPass(ModulePass):
    """
    A pass for lowering operations in the Toy dialect to built-in dialects.
    """

    name = "lower-free-to-linked"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            FreeToLinked(),
        ).rewrite_module(op)
