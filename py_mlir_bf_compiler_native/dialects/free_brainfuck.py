from functools import cache

from mlir.dialects import irdl
from mlir.ir import InsertionPoint, Location, Module


@cache
def _create_free_dialect() -> Module:
    with Location.unknown(), InsertionPoint((module := Module.create()).body):
        dialect = irdl.dialect("bf_free")
        with InsertionPoint(dialect.body):
            irdl.operation_("left")
            irdl.operation_("right")
            irdl.operation_("inc")
            irdl.operation_("dec")
            irdl.operation_("output")
            irdl.operation_("input")
            with InsertionPoint(irdl.operation_("loop").body):
                body = irdl.region([], number_of_blocks=1)
                irdl.regions_([body], ["body"])

        return module


FreeBrainFuck = _create_free_dialect
