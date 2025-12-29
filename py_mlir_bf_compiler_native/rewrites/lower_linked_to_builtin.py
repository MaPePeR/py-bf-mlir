from mlir.dialects import arith, builtin, func, llvm, memref, scf
from mlir.ir import InsertionPoint, Operation
from mlir.rewrite import (
    PatternRewriter,
    RewritePatternSet,
    apply_patterns_and_fold_greedily,
)

from ..dialects import linked_brainfuck as linked_bf


class _Patterns:

    def __init__(self, const_one, const_index_mask, memref) -> None:
        self.const_one = const_one
        self.const_size = const_index_mask
        self.memref = memref

    def getPatternSet(self):
        set = RewritePatternSet()
        set.add(Operation.create("bf_linked.left"), self.lower_move_ops)
        set.add(Operation.create("bf_linked.right"), self.lower_move_ops)
        return set.freeze()

    def lower_move_ops(
        self,
        op: Operation,
        rewriter,
    ):
        if op.name == "bf_linked.left":
            direction_op = arith.SubIOp
        elif op.name == "bf_linked.right":
            direction_op = arith.AddIOp
        else:
            raise AssertionError("op was not of the expected type.")
        rewriter.replace_matched_op(
            [
                add_op := direction_op(op.index, self.const_one.result),
                and_op := arith.AndIOp(add_op.result, self.const_size.result),
            ],
            [and_op.result],
        )

    def lower_inc_dec_ops(
        self,
        op: Operation,
        rewriter,
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
                    load_op.results[0], self.const_one_ui8.result, MEMORY_TYPE
                ),
                store_op := memref.StoreOp(
                    operands=[change_op.result, self.memref, op.operands[0]]
                ),
            ],
            [],
        )

    def lower_loop_op(
        self,
        op: Operation,
        rewriter,
    ):
        while_op = scf.WhileOp(
            [op.index],
            [linked_bf.PositionType()],
            Region(Block([], arg_types=[linked_bf.PositionType()])),
            Region(op.body.detach_block(0)),
        )
        with ImplicitBuilder(while_op.before_region.block) as (index_arg,):
            val = memref.LoadOp(
                operands=[self.memref, index_arg],
                result_types=[MEMORY_TYPE],
            )
            zero = arith.ConstantOp(builtin.IntegerAttr(0, MEMORY_TYPE))
            cmp = arith.CmpiOp(val, zero, "ugt")
            scf.ConditionOp(cmp.result, index_arg)

        rewriter.replace_matched_op(while_op)

    def lower_loop_end_op(
        self,
        op: Operation,
        rewriter,
    ):
        rewriter.replace_matched_op(scf.YieldOp(op.index))

    def lower_output_input_ops(self, op: Operation, rewriter):
        if not isinstance(op, linked_bf.OutputOp) and not isinstance(
            op, linked_bf.InputOp
        ):
            raise AssertionError("Invalid op")

        memref_llvm_struct = llvm.LLVMStructType(
            builtin.StringAttr(""),
            builtin.ArrayAttr(
                [
                    llvm.LLVMPointerType(),
                    llvm.LLVMPointerType(),
                    builtin.i64,
                    llvm.LLVMArrayType(builtin.IntAttr(1), builtin.i64),
                    llvm.LLVMArrayType(builtin.IntAttr(1), builtin.i64),
                ]
            ),
        )
        rewriter.replace_matched_op(
            [
                zero := arith.ConstantOp(
                    builtin.IntegerAttr(0, builtin.IntegerType(64))
                ),
                one := arith.ConstantOp(
                    builtin.IntegerAttr(1, builtin.IntegerType(64))
                ),
                cast_memref_op := builtin.UnrealizedConversionCastOp(
                    operands=[self.memref], result_types=[memref_llvm_struct]
                ),
                cast_index_op := builtin.UnrealizedConversionCastOp(
                    operands=[op.index], result_types=[builtin.i64]
                ),
                val_op := llvm.ExtractValueOp(
                    builtin.DenseArrayBase.from_list(builtin.i64, [1]),
                    cast_memref_op.results[0],
                    result_type=llvm.LLVMPointerType(),
                ),
                elementptr_op := llvm.GEPOp(
                    val_op.results[0],
                    [llvm.GEP_USE_SSA_VAL],
                    MEMORY_TYPE,
                    ssa_indices=[cast_index_op.results[0]],
                ),
                ptr_to_int_op := llvm.PtrToIntOp(elementptr_op.result),
                llvm.InlineAsmOp(
                    "syscall",
                    "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11}",
                    (
                        [one, one, ptr_to_int_op.results[0], one]
                        if isinstance(op, linked_bf.OutputOp)
                        else [zero, zero, ptr_to_int_op.results[0], one]
                    ),
                    has_side_effects=True,
                    res_types=[builtin.i64],
                ),
            ],
            [],
        )


def LowerLinkedToBuiltinBfPass(op: Operation, pass_):
    """
    A pass for lowering operations in the linked dialect to built-in dialects.
    """
    MEMORY_SIZE = 1 << 15
    MEMORY_TYPE = builtin.IntegerType.get_signless(8)

    assert isinstance(op.regions[0].blocks[0].operations[0], func.FuncOp)

    with InsertionPoint.at_block_begin(
        op.regions[0].blocks[0].operations[0].regions[0].blocks[0]
    ):
        const_zero = arith.ConstantOp(builtin.IndexType.get(), 0)
        const_one = arith.ConstantOp(builtin.IndexType.get(), 1)
        const_zero_ui8 = arith.ConstantOp(MEMORY_TYPE, 0)
        arith.ConstantOp(MEMORY_TYPE, 1)
        const_index_mask = arith.ConstantOp(builtin.IndexType.get(), MEMORY_SIZE - 1)
        const_size = arith.ConstantOp(builtin.IndexType.get(), MEMORY_SIZE)
        memref_op = memref.AllocOp(
            builtin.MemRefType.get(
                [MEMORY_SIZE],
                MEMORY_TYPE,
            ),
            [],
            [],
        )

        init_zero_for = scf.ForOp(
            const_zero.result,
            const_size.result,
            const_one.result,
        )
        with InsertionPoint(init_zero_block := init_zero_for.regions[0].blocks[0]):
            memref.StoreOp(
                const_zero_ui8.result,
                memref_op.results[0],
                [init_zero_block.arguments[0]],
            )
            scf.YieldOp([])

    patterns = _Patterns(const_one, const_index_mask, memref_op).getPatternSet()
    apply_patterns_and_fold_greedily(op, patterns)
