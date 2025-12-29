from mlir.dialects import arith, builtin, func, llvm, memref, scf
from mlir.ir import InsertionPoint, Operation, OpView
from mlir.rewrite import (
    PatternRewriter,
    RewritePatternSet,
    apply_patterns_and_fold_greedily,
)

from ..dialects import linked_brainfuck as linked_bf

MEMORY_SIZE = 1 << 15
MEMORY_TYPE = lambda: builtin.IntegerType.get_signless(8)


class _Patterns:

    def __init__(self, const_one, const_index_mask, memref) -> None:
        self.const_one = const_one
        self.const_size = const_index_mask
        self.memref = memref

    def getPatternSet(self):
        def make_op_pattern(opname: str):
            class OpPattern:
                OPERATION_NAME = opname

            return OpPattern

        set = RewritePatternSet()
        set.add(make_op_pattern("bf_linked.left"), self.lower_move_ops)
        set.add(make_op_pattern("bf_linked.right"), self.lower_move_ops)
        set.add(make_op_pattern("bf_linked.inc"), self.lower_inc_dec_ops)
        set.add(make_op_pattern("bf_linked.dec"), self.lower_inc_dec_ops)
        set.add(make_op_pattern("bf_linked.loop"), self.lower_loop_op)
        set.add(make_op_pattern("bf_linked.loop_end"), self.lower_loop_end_op)
        set.add(make_op_pattern("bf_linked.output"), self.lower_output_input_ops)
        set.add(make_op_pattern("bf_linked.input"), self.lower_output_input_ops)
        return set.freeze()

    def lower_move_ops(
        self,
        op: OpView,
        rewriter: PatternRewriter,
    ):
        if op.name == "bf_linked.left":
            direction_op = arith.SubIOp
        elif op.name == "bf_linked.right":
            direction_op = arith.AddIOp
        else:
            raise AssertionError("op was not of the expected type.")
        with rewriter.ip:
            add_op = direction_op(op.operands[0], self.const_one.result)
            and_op = arith.AndIOp(add_op.result, self.const_size.result)

        rewriter.replace_op(op, and_op)

    def lower_inc_dec_ops(
        self,
        op: OpView,
        rewriter: PatternRewriter,
    ):
        match op.operation:
            case Operation(name="bf_linked.inc"):
                new_op = arith.AddIOp
            case Operation(name="bf_linked.dec"):
                new_op = arith.SubIOp
            case _:
                raise AssertionError("op has wrong type")
        with rewriter.ip:
            one = arith.ConstantOp(builtin.IntegerType.get_signless(8), 1)
            load_op = memref.LoadOp(self.memref, [op.operands[0]])
            change_op = new_op(load_op.results[0], one)
            memref.StoreOp(change_op.result, self.memref, [op.operands[0]])
        rewriter.erase_op(op)

    def lower_loop_op(
        self,
        op: OpView,
        rewriter: PatternRewriter,
    ):
        with rewriter.ip:
            while_op = scf.WhileOp(
                [builtin.IndexType.get()],
                [op.operands[0]],
            )
            op.operation.regions[0].blocks[0].append_to(while_op.regions[1])
            with InsertionPoint(before_block := while_op.regions[0].blocks.append()):
                index_arg = before_block.add_argument(
                    builtin.IndexType.get(), op.location
                )
                val = memref.LoadOp(
                    self.memref,
                    [index_arg],
                )
                zero = arith.ConstantOp(MEMORY_TYPE(), 0)
                cmp = arith.cmpi(arith.CmpIPredicate.ugt, val, zero)
                scf.ConditionOp(cmp, [index_arg])

        rewriter.replace_op(op, while_op)

    def lower_loop_end_op(
        self,
        op: OpView,
        rewriter: PatternRewriter,
    ):
        with rewriter.ip:
            yield_op = scf.YieldOp([op.operands[0]])
        rewriter.replace_op(op, yield_op)

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


def LowerLinkedToBuiltinBfPass(op: OpView, pass_):
    """
    A pass for lowering operations in the linked dialect to built-in dialects.
    """
    assert isinstance(op.regions[0].blocks[0].operations[0], func.FuncOp)

    with InsertionPoint.at_block_begin(
        op.regions[0].blocks[0].operations[0].regions[0].blocks[0]
    ):
        const_zero = arith.ConstantOp(builtin.IndexType.get(), 0)
        const_one = arith.ConstantOp(builtin.IndexType.get(), 1)
        const_zero_ui8 = arith.ConstantOp(MEMORY_TYPE(), 0)
        arith.ConstantOp(MEMORY_TYPE(), 1)
        const_index_mask = arith.ConstantOp(builtin.IndexType.get(), MEMORY_SIZE - 1)
        const_size = arith.ConstantOp(builtin.IndexType.get(), MEMORY_SIZE)
        memref_op = memref.AllocOp(
            builtin.MemRefType.get(
                [MEMORY_SIZE],
                MEMORY_TYPE(),
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
