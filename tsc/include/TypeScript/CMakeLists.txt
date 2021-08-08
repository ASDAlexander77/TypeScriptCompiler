add_mlir_dialect(TypeScriptOps ts)
add_mlir_doc(TypeScriptDialect -gen-dialect-doc Dialect TypeScript/)
add_mlir_doc(TypeScriptOps -gen-op-doc TypeScriptOps TypeScript/)

set(LLVM_TARGET_DEFINITIONS LoadBoundPropertiesInterface.td)
mlir_tablegen(LoadBoundPropertiesOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(LoadBoundPropertiesOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRTypeScriptLoadBoundPropertiesInterfaceIncGen)
