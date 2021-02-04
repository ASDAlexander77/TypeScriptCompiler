add_mlir_dialect(TypeScriptOps typescript)
add_mlir_doc(TypeScriptDialect -gen-dialect-doc Dialect TypeScript/)
add_mlir_doc(TypeScriptOps -gen-op-doc TypeScriptOps TypeScript/)

set(LLVM_TARGET_DEFINITIONS TypeScriptTypes.td)
mlir_tablegen(TypeScriptTypeDefs.h.inc -gen-typedef-decls)
mlir_tablegen(TypeScriptTypeDefs.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRTypeScriptDefIncGen)
