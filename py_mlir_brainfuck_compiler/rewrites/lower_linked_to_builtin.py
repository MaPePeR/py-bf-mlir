from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, Region, SSAValue
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

MEMORY_SIZE = 1 << 15
MEMORY_TYPE = builtin.IntegerType(8, builtin.Signedness.SIGNLESS)


class MoveOpLowering(RewritePattern):
    def __init__(self, const_one, const_index_mask) -> None:
        self.const_one = const_one
        self.const_size = const_index_mask

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


class IncDecOpLowering(RewritePattern):
    def __init__(self, const_one, memref: SSAValue) -> None:
        self.const_one = const_one
        self.memref = memref

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: linked_bf.IncrementOp | linked_bf.DecrementOp,
        rewriter: PatternRewriter,
    ):
        match op:
            case linked_bf.IncrementOp():
                new_op = arith.AddiOp
            case linked_bf.DecrementOp():
                new_op = arith.SubiOp
            case _:
                raise AssertionError("op has wrong type")
        rewriter.replace_matched_op(
            [
                load_op := memref.LoadOp(
                    operands=[self.memref, op.operands[0]], result_types=[MEMORY_TYPE]
                ),
                change_op := new_op(
                    load_op.results[0], self.const_one.result, MEMORY_TYPE
                ),
                store_op := memref.StoreOp(
                    operands=[change_op.result, self.memref, op.operands[0]]
                ),
            ],
            [],
        )


class LoopOpLowering(RewritePattern):
    def __init__(self, memref: SSAValue) -> None:
        self.memref = memref

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: linked_bf.LoopOp,
        rewriter: PatternRewriter,
    ):
        before_block = Block(arg_types=[linked_bf.PositionType()])
        before_block.add_ops(
            [
                val := memref.LoadOp(
                    operands=[self.memref, before_block.args[0]],
                    result_types=[MEMORY_TYPE],
                ),
                zero := arith.ConstantOp(builtin.IntegerAttr(0, MEMORY_TYPE)),
                cmp := arith.CmpiOp(val, zero, "ugt"),
                scf.ConditionOp(cmp.result, before_block.args[0]),
            ]
        )
        rewriter.replace_matched_op(
            scf.WhileOp(
                [op.index],
                [linked_bf.PositionType()],
                Region(before_block),
                Region(op.body.detach_block(0)),
            )
        )


class LoopEndOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: linked_bf.LoopEndOp,
        rewriter: PatternRewriter,
    ):
        rewriter.replace_matched_op(scf.YieldOp(op.index))


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
                const_one_ui8 := arith.ConstantOp(
                    builtin.IntegerAttr(1, MEMORY_TYPE),
                ),
                const_index_mask := arith.ConstantOp(
                    builtin.IntegerAttr(MEMORY_SIZE - 1, linked_bf.PositionType())
                ),
                const_size := arith.ConstantOp(
                    builtin.IntegerAttr(MEMORY_SIZE, linked_bf.PositionType())
                ),
                memref_op := memref.AllocOp(
                    [], [], builtin.MemRefType(MEMORY_TYPE, [MEMORY_SIZE])
                ),
            ],
            InsertPoint.at_start(op.body.block.first_op.body.block),
        )
        const_one.result.name_hint = "const_one"
        const_one_ui8.result.name_hint = "const_one_ui8"
        const_index_mask.result.name_hint = "index_mask"
        const_size.result.name_hint = "const_size"
        memref_op.results[0].name_hint = "memory"
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    MoveOpLowering(const_one, const_index_mask),
                    IncDecOpLowering(const_one_ui8, memref_op.results[0]),
                    LoopOpLowering(memref_op.results[0]),
                    LoopEndOpLowering(),
                ]
            ),
        ).rewrite_module(op)
