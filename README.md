# MLIR based Brainfuck Compiler in Python

This project is an experiment to implement a Compiler using [MLIR](https://mlir.llvm.org/).
[Brainfuck](https://en.wikipedia.org/wiki/Brainfuck) was choosen as a programming language for its simplicity.

To keep more true to the stack I would use for a real programming language a [Lark](https://github.com/lark-parser/lark) based [grammar](py_mlir_brainfuck_compiler/brainfuck.lark) is used to lex and parse the Brainfuck code. To create the MLIR i tried both, the [native LLVM MLIR Python](https://mlir.llvm.org/docs/Bindings/Python/) bindings, and the Python [xDSL](https://github.com/xdslproject/xdsl/) package. This branch is the version using the xDSL package.

## Syntax Tree
So a simple Brainfuck code like `>+>+[<]` is parsed as this Syntax Tree:

```
Tree(Token('RULE', 'start'), [
    Token('MOVE_RIGHT', '>'), 
    Token('INCREMENT', '+'),
    Token('MOVE_RIGHT', '>'),
    Token('INCREMENT', '+'),
    Tree(Token('RULE', 'loop'), [
        Token('MOVE_LEFT', '<')
    ])
])
```

## Initial "free" Dialect
This Tree is then [converted](py_mlir_brainfuck_compiler/gen_mlir.py) to the [first MLIR dialect](py_mlir_brainfuck_compiler/dialects/free_brainfuck.py).
This first dialect does not contain any SSA values, yet, but mimics the syntax tree very closely:

```mlir
builtin.module {
  func.func @main() {
    "bf.free.right"() : () -> ()
    "bf.free.inc"() : () -> ()
    "bf.free.right"() : () -> ()
    "bf.free.inc"() : () -> ()
    "bf.free.loop"() ({
      "bf.free.left"() : () -> ()
    }) : () -> ()
    func.return
  }
}
```

## Lowering to linked Dialect
In a [first lowering step](py_mlir_brainfuck_compiler/rewrites/lower_free_to_linked_bf.py) this "free" dialect is lowered to the ["linked" dialect](py_mlir_brainfuck_compiler/dialects/linked_brainfuck.py).
In this second dialect every Operation that uses the memory index receives the index as an operand. Every operation, that modifies the index will produce a new index as a result. (This lowering was accomplished with a custom Tree-Walk that seems to be unconvential for xDSL?)

```
builtin.module {
  func.func @main() {
    %0 = arith.constant 0 : index
    %1 = "bf.linked.right"(%0) : (index) -> index
    "bf.linked.inc"(%1) : (index) -> ()
    %2 = "bf.linked.right"(%1) : (index) -> index
    "bf.linked.inc"(%2) : (index) -> ()
    %3 = "bf.linked.loop"(%2) ({
    ^bb0(%4 : index):
      %5 = "bf.linked.left"(%4) : (index) -> index
      "bf.linked.loop.end"(%5) : (index) -> ()
    }) : (index) -> index
    func.return
  }
}
```

## Lowering to builtins
Using a [second lowering step](py_mlir_brainfuck_compiler/rewrites/lower_linked_to_builtin.py) these linked ops are converted to the builtin dialects. Namely, `arith`, `scf`, and `memref`. The output and input operands (not displayed here, though arguably the most interesting) are converted to `llvm.inline_asm` to get a read/write syscall.

```
builtin.module {
  func.func @main() {
    %0 = arith.constant 0 : index
    %const_one = arith.constant 1 : index
    %1 = arith.constant 0 : i8
    %const_one_ui8 = arith.constant 1 : i8
    %index_mask = arith.constant 32767 : index
    %const_size = arith.constant 32768 : index
    %memory = memref.alloc() : memref<32768xi8>
    scf.for %2 = %0 to %const_size step %const_one {
      memref.store %1, %memory[%2] : memref<32768xi8>
    }
    %3 = arith.constant 0 : index
    %4 = arith.addi %3, %const_one : index
    %5 = arith.andi %4, %index_mask : index
    %6 = memref.load %memory[%5] : memref<32768xi8>
    %7 = arith.addi %6, %const_one_ui8 : i8
    memref.store %7, %memory[%5] : memref<32768xi8>
    %8 = arith.addi %5, %const_one : index
    %9 = arith.andi %8, %index_mask : index
    %10 = memref.load %memory[%9] : memref<32768xi8>
    %11 = arith.addi %10, %const_one_ui8 : i8
    memref.store %11, %memory[%9] : memref<32768xi8>
    %12 = scf.while (%13 = %9) : (index) -> index {
      %14 = memref.load %memory[%13] : memref<32768xi8>
      %15 = arith.constant 0 : i8
      %16 = arith.cmpi ugt, %14, %15 : i8
      scf.condition(%16) %13 : index
    } do {
    ^bb0(%17 : index):
      %18 = arith.subi %17, %const_one : index
      %19 = arith.andi %18, %index_mask : index
      scf.yield %19 : index
    }
    func.return
  }
}
```

This MLIR can then be further lowered/optimized using the `mlir-opt` tool, be translated to LLVM-IR with `mlir-translate`, converted to assembly with `llc` and then compiled using `clang`. See the [Makefile](Makefile), that can be used to compile `.bf` code to `.out` exceutables, for the exact commands.

## Devcontainer

This project contains a Devcontainer Configuration, however it compiles the [llvm-project](https://github.com/llvm/llvm-project/) to get the `mlir` executables, so first startup can take a while and it is not very optimized in general.
