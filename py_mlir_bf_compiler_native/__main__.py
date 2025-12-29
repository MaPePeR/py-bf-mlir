import argparse
import pathlib
import sys
import typing

import lark
from mlir.dialects import irdl
from mlir.ir import Context, Location
from mlir.passmanager import PassManager

from .dialects.free_brainfuck import FreeBrainFuck
from .dialects.linked_brainfuck import LinkedBrainFuck
from .gen_mlir import GenMLIR
from .parser import BrainfuckParser
from .rewrites.lower_free_to_linked_bf import LowerFreeToLinkedBfPass


def main(
    sourcefile: pathlib.Path,
    target: typing.Literal["ast", "free", "linked", "builtin"],
    output: typing.TextIO,
):
    parser = BrainfuckParser()

    with sourcefile.open("r") as h:
        ast = parser.parse(h.read())
    assert isinstance(ast, lark.Tree)
    assert (
        isinstance(ast.data, lark.Token)
        and ast.data.type == "RULE"
        and ast.data.value == "start"
    )
    if target == "ast":
        output.write(str(ast))
        return 0
    with Context(), Location.unknown():
        irdl.load_dialects(FreeBrainFuck())
        if target != "free":
            irdl.load_dialects(LinkedBrainFuck())
        gen = GenMLIR()
        gen.gen_main_func(ast.children)

        pm = PassManager()
        pm.enable_verifier(False)
        if target == "linked" or target == "builtin":
            pm.add(LowerFreeToLinkedBfPass)
        if target == "builtin":
            # LowerLinkedToBuiltinBfPass().apply(ctx, gen.module)
            pass
        pm.run(gen.module.operation)

    output.write(str(gen.module))
    gen.module.operation.verify()


parser = argparse.ArgumentParser(description="Process Toy file")
parser.add_argument("source", type=pathlib.Path, help="Brainfuck Source File")
parser.add_argument(
    "--target",
    dest="target",
    choices=[
        "ast",
        "free",
        "linked",
        "builtin",
    ],
    default="builtin",
    help="What MLIR to generate (default: builtin)",
)
parser.add_argument(
    "--output",
    "-o",
    type=pathlib.Path,
    default=None,
    help="Output destination (default: stdout)",
)

args = parser.parse_args()
output = sys.stdout
if args.output:
    assert isinstance(args.output, pathlib.Path)
    output = args.output.open("w")
try:
    ret = main(args.source, args.target, output)
finally:
    output.close()
sys.exit(ret)
