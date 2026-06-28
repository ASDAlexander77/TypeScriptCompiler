#include <typeinfo>

// The runtime symbols required by JIT-compiled code are exported under their
// JIT names via TypeScriptRuntime.def. We deliberately do NOT export the MLIR
// '__mlir_execution_engine_init/destroy' callbacks: when those are absent, the
// MLIR ExecutionEngine loads this library as a plain JITDylib and resolves
// symbols directly from its export table. That avoids the cross-heap free that
// happens when the DLL populates a StringMap owned (and later freed) by the EXE
// while both link the static CRT (separate heaps).

#ifdef _WIN32
#pragma comment(linker, "/EXPORT:??_7type_info@@6B@")
#endif
