#ifndef TSLANG_JITCACHE_H_
#define TSLANG_JITCACHE_H_

#include "TypeScript/DataStructs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Target/TargetMachine.h"

#include <memory>
#include <string>
#include <vector>

// One .ts file compiled for the JIT: the program, or a module it imports.
struct JitUnit
{
    std::string sourcePath;
    bool isMain = false;
    std::unique_ptr<llvm::MemoryBuffer> object;
    // where the object is in the cache folder; empty if it could not be written there
    std::string objectPath;
    // the hash of the object, as its manifest has it
    std::string objectHash;
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

// Which build of tslang this is (its executable's size and time), or empty if that cannot be told:
// what the cache keys its objects on, so that another build compiles them again.
std::string jitCompilerId(const char *argv0);

// --jit-native (jitnative.cpp): the units' objects linked into one DLL, with its PDB, in the cache
// folder beside the program's object; reused while the objects and options are the same. `dllPath`
// is set to the DLL. Non-zero if an object is not on disk or the link fails.
int linkJitUnits(const char *argv0, llvm::ArrayRef<JitUnit> units, CompileOptions &compileOptions,
                 llvm::TargetMachine &tm, std::string &dllPath);

// The entry point the JIT calls, `int (int argc, char **argv)`, whatever `main` lowered to (see
// addEntryThunk in jit.cpp)
#define JIT_ENTRY_THUNK_NAME "__tslang_jit_main"

// The functions the DLL linkJitUnits makes exports beside the program's own: called before the
// program's initializers and after the program, they reach the C runtime linked into the DLL.
#define JIT_NATIVE_INIT_NAME "__tslang_jit_native_init"
#define JIT_NATIVE_FLUSH_NAME "__tslang_jit_native_flush"

#endif // TSLANG_JITCACHE_H_
