from functools import cache

from mlir.dialects import builtin, irdl
from mlir.ir import InsertionPoint, Location, Module


@cache
def _create_linked_dialect() -> Module:
    with Location.unknown(), InsertionPoint((module := Module.create()).body):
        dialect = irdl.dialect("bf_linked")
        with InsertionPoint(dialect.body):

            for op in ["left", "right"]:
                with InsertionPoint(irdl.operation_(op).body):
                    t = irdl.is_(builtin.TypeAttr.get(builtin.IndexType.get()))
                    irdl.operands_([t], ["pos"], [irdl.Variadicity.single])
                    irdl.results_([t], ["new_pos"], [irdl.Variadicity.single])
            for op in ["inc", "dec", "output", "input"]:
                with InsertionPoint(irdl.operation_(op).body):
                    t = irdl.is_(builtin.TypeAttr.get(builtin.IndexType.get()))
                    irdl.operands_([t], ["pos"], [irdl.Variadicity.single])
            with InsertionPoint(irdl.operation_("loop").body):
                t = irdl.is_(builtin.TypeAttr.get(builtin.IndexType.get()))
                irdl.operands_([t], ["pos"], [irdl.Variadicity.single])
                irdl.results_([t], ["new_pos"], [irdl.Variadicity.single])
                body = irdl.region([], number_of_blocks=1)
                irdl.regions_([body], ["body"])
            with InsertionPoint(irdl.operation_("loop_end").body):
                t = irdl.is_(builtin.TypeAttr.get(builtin.IndexType.get()))
                irdl.operands_([t], ["pos"], [irdl.Variadicity.single])

        return module


LinkedBrainFuck = _create_linked_dialect
