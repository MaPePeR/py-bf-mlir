# Keep intermediate files
.SECONDARY:

%.mlir :: %.bf
	python -m py_mlir_brainfuck_compiler $< -o $@

%.opt.mlir : %.mlir
	mlir-opt --convert-scf-to-cf --convert-cf-to-llvm --convert-func-to-llvm \
		--convert-arith-to-llvm --expand-strided-metadata --normalize-memrefs \
		--memref-expand --fold-memref-alias-ops --finalize-memref-to-llvm \
		--reconcile-unrealized-casts \
		$< -o $@

%.ll : %.opt.mlir
	mlir-translate --mlir-to-llvmir $< -o $@

%.s : %.ll
	llc $< -o $@

%.out : %.s
	clang -g $< -o $@