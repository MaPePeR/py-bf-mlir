from xdsl.irdl import IRDLOperation, irdl_op_definition, region_def


@irdl_op_definition
class MoveLeftOp(IRDLOperation):
    name = "bf.left"


@irdl_op_definition
class MoveRightOp(IRDLOperation):
    name = "bf.right"


@irdl_op_definition
class IncrementOp(IRDLOperation):
    name = "bf.inc"


@irdl_op_definition
class DecrementOp(IRDLOperation):
    name = "bf.dec"


@irdl_op_definition
class OutputOp(IRDLOperation):
    name = "bf.output"


@irdl_op_definition
class InputOp(IRDLOperation):
    name = "bf.input"


@irdl_op_definition
class LoopOp(IRDLOperation):
    name = "bf.loop"
    body = region_def("single_block")
