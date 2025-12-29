from typing import Protocol

from mlir.dialects import arith, builtin
from mlir.ir import InsertionPoint, Location, Operation, OpView


class Visitor(Protocol):
    def enter(self, op: OpView) -> bool: ...
    def leave(self, op: OpView) -> None: ...


def walk(op: OpView, visitor: Visitor):
    if visitor.enter(op):
        for region in op.regions:
            for block in region.blocks:
                for block_op in block.operations:
                    walk(block_op, visitor)
        visitor.leave(op)


class FreeToLinkedVisitor(Visitor):
    index: list = []

    def enter(self, op: OpView) -> bool:
        match op.operation:
            case Operation(name="func.func"):
                with InsertionPoint.at_block_begin(op.operation.regions[0].blocks[0]):
                    new_op = arith.ConstantOp(builtin.IndexType.get(), 0)
                assert len(self.index) == 0
                self.index = [new_op.results[0]]
                return True
            case Operation(name="bf_free.loop"):
                with InsertionPoint.after(op.operation):
                    new_op = Operation.create(
                        "bf_linked.loop",
                        operands=[self.index[-1]],
                        results=[builtin.IndexType.get()],
                        regions=1,
                    )
                op.operation.regions[0].blocks[0].append_to(new_op.regions[0])
                new_op.regions[0].blocks[0].add_argument(
                    builtin.IndexType.get(), Location.unknown()
                )
                op.operation.detach_from_parent()

                self.index[-1] = new_op.results[0]
                self.index.append(new_op.regions[0].blocks[0].arguments[0])
                walk(OpView(new_op), self)
                self.index.pop()

                return False
            case Operation(name="bf_free.left"):
                with InsertionPoint.after(op.operation):
                    new_op = Operation.create(
                        "bf_linked.left",
                        operands=[self.index[-1]],
                        results=[builtin.IndexType.get()],
                    )
                op.operation.detach_from_parent()
                self.index[-1] = new_op.results[0]
            case Operation(name="bf_free.right"):
                with InsertionPoint.after(op.operation):
                    new_op = Operation.create(
                        "bf_linked.right",
                        operands=[self.index[-1]],
                        results=[builtin.IndexType.get()],
                    )
                op.operation.detach_from_parent()
                self.index[-1] = new_op.results[0]
            case Operation(name="bf_free.inc"):
                with InsertionPoint.after(op.operation):
                    new_op = Operation.create(
                        "bf_linked.inc", operands=[self.index[-1]]
                    )
                op.operation.detach_from_parent()
            case Operation(name="bf_free.dec"):
                with InsertionPoint.after(op.operation):
                    new_op = Operation.create(
                        "bf_linked.dec", operands=[self.index[-1]]
                    )
                op.operation.detach_from_parent()
            case Operation(name="bf_free.output"):
                with InsertionPoint.after(op.operation):
                    new_op = Operation.create(
                        "bf_linked.output", operands=[self.index[-1]]
                    )
                op.operation.detach_from_parent()
            case Operation(name="bf_free.input"):
                with InsertionPoint.after(op.operation):
                    new_op = Operation.create(
                        "bf_linked.input", operands=[self.index[-1]]
                    )
                op.operation.detach_from_parent()
            case Operation(name="bf_linked.loop"):
                return True
            case Operation(name="builtin.module"):
                return True

        return False

    def leave(self, op: OpView):

        match op.operation:
            case Operation(name="func.func"):
                self.index.pop()
            case Operation(name="bf_linked.loop"):
                op.operation.regions[0].blocks[0].append(
                    Operation.create("bf_linked.loop_end", operands=[self.index[-1]])
                )


def LowerFreeToLinkedBfPass(op, pass_):
    """
    A pass for lowering operations in the free dialect to linked dialect.
    """
    walk(op, FreeToLinkedVisitor())
