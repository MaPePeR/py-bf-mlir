from typing import TypeAlias

from xdsl.dialects import builtin
from xdsl.ir import Dialect, Region, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    region_def,
    result_def,
    traits_def,
)
from xdsl.traits import IsTerminator

PositionType: TypeAlias = builtin.IndexType


@irdl_op_definition
class MoveLeftOp(IRDLOperation):
    name = "bf.linked.left"
    index = operand_def(PositionType())
    new_index = result_def(PositionType())

    def __init__(self, index: SSAValue):
        super().__init__(operands=[index], result_types=[PositionType()])


@irdl_op_definition
class MoveRightOp(IRDLOperation):
    name = "bf.linked.right"
    index = operand_def(PositionType())
    new_index = result_def(PositionType())

    def __init__(self, index: SSAValue):
        super().__init__(operands=[index], result_types=[PositionType()])


@irdl_op_definition
class IncrementOp(IRDLOperation):
    name = "bf.linked.inc"
    index = operand_def(PositionType())

    def __init__(self, index: SSAValue):
        super().__init__(operands=[index])


@irdl_op_definition
class DecrementOp(IRDLOperation):
    name = "bf.linked.dec"
    index = operand_def(PositionType())

    def __init__(self, index: SSAValue):
        super().__init__(operands=[index])


@irdl_op_definition
class OutputOp(IRDLOperation):
    name = "bf.linked.output"
    index = operand_def(PositionType())

    def __init__(self, index: SSAValue):
        super().__init__(operands=[index])


@irdl_op_definition
class InputOp(IRDLOperation):
    name = "bf.linked.input"
    index = operand_def(PositionType())

    def __init__(self, index: SSAValue):
        super().__init__(operands=[index])


@irdl_op_definition
class LoopOp(IRDLOperation):
    name = "bf.linked.loop"
    index = operand_def(PositionType())
    new_index = result_def(PositionType())
    body = region_def("single_block")

    def __init__(self, index: SSAValue, body: Region):
        super().__init__(
            operands=[index], result_types=[PositionType()], regions=[body]
        )


@irdl_op_definition
class LoopEndOp(IRDLOperation):
    name = "bf.linked.loop.end"
    index = operand_def(PositionType())
    traits = traits_def(IsTerminator())

    def __init__(self, index: SSAValue):
        super().__init__(operands=[index])


LinkedBrainFuck = Dialect(
    "bf.linked",
    [
        MoveLeftOp,
        MoveRightOp,
        IncrementOp,
        DecrementOp,
        OutputOp,
        InputOp,
        LoopOp,
        LoopEndOp,
    ],
    [],
)
