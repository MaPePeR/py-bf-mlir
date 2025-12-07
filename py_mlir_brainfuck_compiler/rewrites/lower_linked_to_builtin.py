from xdsl.context import Context
from xdsl.dialects import arith, builtin, func
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint, Rewriter

from ..dialects import linked_brainfuck as linked_bf


class MoveOpLowering(RewritePattern):
    def __init__(self, const_one, const_size) -> None:
        self.const_one = const_one
        self.const_size = const_size

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: linked_bf.MoveLeftOp | linked_bf.MoveRightOp,
        rewriter: PatternRewriter,
    ):
        if isinstance(op, linked_bf.MoveLeftOp):
            direction_op = arith.SubiOp
        elif isinstance(op, linked_bf.MoveRightOp):
            direction_op = arith.AddiOp
        else:
            raise AssertionError("op was not of the expected type.")
        rewriter.replace_matched_op(
            [
                add_op := direction_op(op.index, self.const_one.result),
                and_op := arith.AndIOp(add_op.result, self.const_size.result),
            ],
            [and_op.result],
        )


class LowerLinkedToBuiltinBfPass(ModulePass):
    """
    A pass for lowering operations in the Toy dialect to built-in dialects.
    """

    name = "lower-linked-to-builtin"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        assert isinstance(op.body.block.first_op, func.FuncOp)
        Rewriter.insert_op(
            [
                const_one := arith.ConstantOp(
                    builtin.IntegerAttr(1, linked_bf.PositionType())
                ),
                const_size := arith.ConstantOp(
                    builtin.IntegerAttr(255, linked_bf.PositionType())
                ),
            ],
            InsertPoint.at_start(op.body.block.first_op.body.block),
        )
        const_one.result.name_hint = "const_one"
        const_size.result.name_hint = "const_size"
        PatternRewriteWalker(
            GreedyRewritePatternApplier([MoveOpLowering(const_one, const_size)])
        ).rewrite_module(op)
