import argparse
import pathlib
import sys
import typing

import lark
from xdsl.context import Context
from xdsl.dialects import affine, arith, builtin, func, memref, printf, scf
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

from .dialects.free_brainfuck import FreeBrainFuck
from .dialects.linked_brainfuck import LinkedBrainFuck
from .gen_mlir import GenMLIR
from .parser import BrainfuckParser
from .rewrites.lower_free_to_linked_bf import LowerFreeToLinkedBfPass
from .rewrites.lower_linked_to_builtin import LowerLinkedToBuiltinBfPass


def context():
    ctx = Context()
    ctx.load_dialect(affine.Affine)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(memref.MemRef)
    ctx.load_dialect(printf.Printf)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(FreeBrainFuck)
    ctx.load_dialect(LinkedBrainFuck)
    return ctx


def main(
    sourcefile: pathlib.Path,
    target: typing.Literal["ast", "free", "linked", "builtin"],
    output: typing.TextIO,
):
    parser = BrainfuckParser()

    ctx = context()

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
    gen = GenMLIR()
    gen.gen_main_func(ast.children)

    if target == "linked" or target == "builtin":
        LowerFreeToLinkedBfPass().apply(ctx, gen.module)
    if target == "builtin":
        LowerLinkedToBuiltinBfPass().apply(ctx, gen.module)

    verify_error = None
    try:
        gen.module.verify()
    except VerifyException as e:
        verify_error = e
        print("Verification failed:", file=sys.stderr)
        print(verify_error, file=sys.stderr)
        # raise e

    printer = Printer(stream=output)
    printer.print_op(gen.module)

    if verify_error:
        print("\nVerification failed:", file=sys.stderr)
        print(verify_error, file=sys.stderr)
        return 1


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
