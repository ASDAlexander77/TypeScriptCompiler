#ifndef TYPESCRIPT_DATASTRUCT_H_
#define TYPESCRIPT_DATASTRUCT_H_

#include "TypeScript/TypeScriptCompiler/Defines.h"
#include "TypeScript/TargetInfo.h"

#include <string>
#include <vector>

struct CompileOptions
{
    bool isJit;
    enum MemoryModel memoryModel;
    bool enableBuiltins;
    bool noDefaultLib;
    std::string defaultDeclarationTSFile;
    bool disableWarnings;
    bool generateDebugInfo;
    bool lldbDebugInfo;
    std::string moduleTargetTriple;
    TargetInfo targetInfo;

    // Pointer/size width in bits. Kept as an accessor rather than a field so it cannot drift
    // away from targetInfo, which is the single place the target's width is decided.
    int sizeBits() const
    {
        return static_cast<int>(targetInfo.pointerBits);
    }
    bool isWasm;
    bool isWindows;
    bool isExecutable;
    bool isDLL;

    // Whether this compilation is building the module that holds the program's entry point, and
    // so has to have a `main` even when the root holds no code to put in one. `--emit=jit` and
    // `--emit=exe` answer that by themselves. `--emit=obj` cannot: the same action compiles the
    // program and every library linked beside it, so a program built that way says so with
    // `--entry-point`. Without that, a library whose root merely initializes a variable would
    // define `main` too, and two of them fail to link with "duplicate symbol: main".
    bool generateEntryPoint;
    // --export filters: `all`, `none`, names or globs, `!name` to exclude; empty means the
    // `export` keyword decides (see MLIRExportFilter.h)
    std::vector<std::string> exportFilters;
    bool embedExportDeclarations;
    std::string outputFolder;
    bool appendGCtorsToMethod;
    bool strictNullChecks;
    bool enableFastMath;

    // Whether the module imports a tslang shared library (`import './lib'` resolving to a DLL).
    // Set once the module is generated, and read when linking: under `-mm=gc` such a program must
    // take its collector from gc.dll, like the library does, because two statically linked
    // collectors in one process each free what only the other's memory references.
    // See docs/single-gc-collector-design.md.
    bool importsSharedLibrary = false;

    // Under the JIT: every .ts module is compiled into an object of its own, kept in a cache
    // folder (`__jit` beside the source), and the objects are loaded side by side - so an import
    // pointing to a .ts file is a declaration, as it is compiled. Off (`--jit-cache=false`), the
    // imported file is included with its bodies into the one module the JIT runs.
    bool jitCache = false;

    // Filled while the module is generated, for the JIT cache: the .ts files imported, directly
    // or through another import, each after the ones it imports (canonical paths), and the
    // shared libraries imported, whose declarations the module was generated against.
    std::vector<std::string> sourceImports;
    std::vector<std::string> sharedLibraryImports;
    // an import led back to a file still being generated: the modules in the cycle cannot be
    // compiled one by one (the JIT cache then compiles the program as one module)
    bool importCycle = false;

    // Whether the Boehm runtime has to be present. Only `gc` needs it: it is the model whose
    // reclamation *is* the collector. `rc` frees through the reference counts it maintains and
    // `none` frees nothing, so both allocate straight from `malloc` and neither links libgc.
    //
    // This was true for `rc` too while the retain/release insertion points were being built
    // (§9.6 through §9.27). Boehm collecting behind them made a missing release invisible, which
    // was the point at the time - but it also meant no memory measurement taken under `rc` said
    // anything about reference counting, since the collector was doing the reclaiming either
    // way. See docs/reference-counting-evaluation.md §9.28.
    bool needsGCRuntime() const
    {
        return memoryModel == MemoryModelGC;
    }

    // Whether allocations maintain a reference count in the block header.
    bool isRefCounted() const
    {
        return memoryModel == MemoryModelRC;
    }

    // Whether the ownership operations are live and lower to code - retains and releases under
    // `rc`, releases that destroy under `own`. Where only the count itself matters, the
    // question is isRefCounted().
    bool tracksOwnership() const
    {
        return memoryModel == MemoryModelRC || memoryModel == MemoryModelOwn;
    }
};

#endif // TYPESCRIPT_DATASTRUCT_H_