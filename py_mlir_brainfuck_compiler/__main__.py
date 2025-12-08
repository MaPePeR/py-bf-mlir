import sys

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


def main():
    if len(sys.argv) < 2:
        print("Missing arg.")
        return 1
    parser = BrainfuckParser()

    ctx = context()

    with open(sys.argv[1]) as h:
        ast = parser.parse(h.read())
    assert isinstance(ast, lark.Tree)
    assert (
        isinstance(ast.data, lark.Token)
        and ast.data.type == "RULE"
        and ast.data.value == "start"
    )
    gen = GenMLIR()
    gen.gen_main_func(ast.children)

    LowerFreeToLinkedBfPass().apply(ctx, gen.module)
    LowerLinkedToBuiltinBfPass().apply(ctx, gen.module)

    verify_error = None
    try:
        gen.module.verify()
    except VerifyException as e:
        verify_error = e
        print("Verification failed:", file=sys.stderr)
        print(verify_error, file=sys.stderr)
        # raise e

    printer = Printer()
    printer.print_op(gen.module)

    if verify_error:
        print("\nVerification failed:", file=sys.stderr)
        print(verify_error, file=sys.stderr)
        return 1


sys.exit(main())
