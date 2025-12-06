from xdsl.irdl import IRDLOperation, irdl_op_definition, region_def


@irdl_op_definition
class MoveLeftOp(IRDLOperation):
    name = "bf.free.left"


@irdl_op_definition
class MoveRightOp(IRDLOperation):
    name = "bf.free.right"


@irdl_op_definition
class IncrementOp(IRDLOperation):
    name = "bf.free.inc"


@irdl_op_definition
class DecrementOp(IRDLOperation):
    name = "bf.free.dec"


@irdl_op_definition
class OutputOp(IRDLOperation):
    name = "bf.free.output"


@irdl_op_definition
class InputOp(IRDLOperation):
    name = "bf.free.input"


@irdl_op_definition
class LoopOp(IRDLOperation):
    name = "bf.free.loop"
    body = region_def("single_block")
