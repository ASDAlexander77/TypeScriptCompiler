add_mlir_dialect_library(MLIRTypeScript
        TypeScriptDialect.cpp
        TypeScriptOps.cpp
  	AST.cpp
  	MLIRGen.cpp
  	LowerToAffineLoops.cpp
  	LowerToLLVM.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/TypeScript

        DEPENDS
        MLIRTypeScriptOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
