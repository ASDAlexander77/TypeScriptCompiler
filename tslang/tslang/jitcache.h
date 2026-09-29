#ifndef TSLANG_JITCACHE_H_
#define TSLANG_JITCACHE_H_

#include "TypeScript/DataStructs.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/Support/MemoryBuffer.h"

#include <memory>
#include <string>
#include <vector>

// One .ts file compiled for the JIT: the program, or a module it imports.
struct JitUnit
{
    std::string sourcePath;
    bool isMain = false;
    std::unique_ptr<llvm::MemoryBuffer> object;
    // what the file's global constructors became: the JIT runs no object's llvm.global_ctors, so
    // they are called from one function, which the JIT calls before the program starts
    std::string initFunction;
    // `--gctors-as-method`: the program's __mlir_gctors
    bool hasGCtorsMethod = false;
};

// The program in `mainFile` and every .ts module it imports, directly or through another, each an
// object: read from the cache if its sources and options have not changed since it was compiled,
// otherwise compiled and put in the cache. In `units` an imported module comes before its
// importer, the program last. Non-zero if a file does not compile.
int buildJitUnits(const char *argv0, llvm::StringRef mainFile, const CompileOptions &compileOptions,
                  llvm::orc::JITTargetMachineBuilder &tmBuilder, std::vector<JitUnit> &units);

#endif // TSLANG_JITCACHE_H_
